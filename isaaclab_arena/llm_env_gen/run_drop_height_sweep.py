# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep initial drop heights for one object on the GR1 table.

This diagnosis tool fixes an object's XY pose and orientation from the normal
placement result, then repeatedly raises only its Z position by each requested
height and lets physics settle. The expected vertical fall from the raised spawn
height is not counted as falling off; the fall-off check compares the settled
pose against the original table-placement baseline.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.llm_env_gen.stability_utils import (
    classify_object,
    get_rigid_pose,
    get_rigid_velocity,
    tilt_angle_rad,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def _add_cli_args(parser) -> None:
    group = parser.add_argument_group("Drop Height Sweep")
    group.add_argument("--object", type=str, default="mustard_bottle_hope_robolab", help="Object to drop.")
    group.add_argument(
        "--heights_m",
        type=str,
        default="0.0,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.10",
        help="Comma-separated initial Z offsets above the baseline placement pose, in meters.",
    )
    group.add_argument("--env_id", type=int, default=0, help="Environment index to inspect.")
    group.add_argument("--settle_steps", type=int, default=120, help="Steps to run before metric readout.")
    group.add_argument("--dwell_steps", type=int, default=0, help="Extra steps after each readout for visual inspection.")
    group.add_argument("--tilt_thresh_deg", type=float, default=20.0, help="Stable tilt threshold.")
    group.add_argument("--xy_drift_thresh", type=float, default=0.05, help="Stable XY drift threshold in meters.")
    group.add_argument(
        "--z_below_baseline_thresh",
        type=float,
        default=0.05,
        help="Allowed settled Z below the baseline table-placement pose.",
    )
    group.add_argument("--vel_thresh_lin", type=float, default=0.05, help="Stable linear velocity threshold.")
    group.add_argument("--vel_thresh_ang", type=float, default=0.20, help="Stable angular velocity threshold.")


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    _add_cli_args(parser)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = parser.parse_args()

        if args_cli.seed is not None:
            set_seed(args_cli.seed)

        if not getattr(args_cli, "objects", None):
            args_cli.objects = [args_cli.object]
        elif args_cli.object not in args_cli.objects:
            raise ValueError(f"--object {args_cli.object!r} must be included in --objects {args_cli.objects!r}")

        arena_builder = get_arena_builder_from_cli(args_cli)
        env, _ = arena_builder.make_registered_and_return_cfg()
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)

        try:
            _run_sweep(env, args_cli)
        finally:
            teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
            env.close()

    return 0


def _run_sweep(env, args_cli) -> None:
    object_name = args_cli.object
    env_id = int(args_cli.env_id)
    assert 0 <= env_id < int(args_cli.num_envs)

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env_ids = torch.tensor([env_id], device=env.unwrapped.device)
    obj = env.unwrapped.scene.rigid_objects[object_name]

    env.reset()
    baseline_pos, baseline_quat = get_rigid_pose(env, object_name, env_id)
    baseline_pose = torch.cat([baseline_pos, baseline_quat]).clone()

    thresholds = {
        "first_step_jump_thresh": 0.02,
        "z_drop_thresh": float(args_cli.z_below_baseline_thresh),
        "tilt_thresh_rad": math.radians(float(args_cli.tilt_thresh_deg)),
        "xy_drift_thresh": float(args_cli.xy_drift_thresh),
        "vel_thresh_lin": float(args_cli.vel_thresh_lin),
        "vel_thresh_ang": float(args_cli.vel_thresh_ang),
    }

    print(
        f"[drop-sweep] object={object_name} baseline_xyz="
        f"({baseline_pos[0].item():.4f}, {baseline_pos[1].item():.4f}, {baseline_pos[2].item():.4f})",
        flush=True,
    )
    print(
        f"{'height_m':>8s} {'status':>12s} {'tilt_deg':>9s} {'below_base_m':>12s} "
        f"{'xy_m':>8s} {'settle_drop_m':>13s} {'lin_vel':>8s} {'ang_vel':>8s}",
        flush=True,
    )

    stable_heights = []
    heights_m = [float(value) for value in args_cli.heights_m.split(",") if value]
    for height_m in heights_m:
        env.reset()

        spawn_pose = baseline_pose.clone()
        spawn_pose[2] += float(height_m)
        obj.write_root_pose_to_sim(spawn_pose.unsqueeze(0), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.unwrapped.device), env_ids=env_ids)

        spawn_pos, spawn_quat = get_rigid_pose(env, object_name, env_id)
        env.step(zero_action)
        first_step_pos, _ = get_rigid_pose(env, object_name, env_id)

        for _ in range(int(args_cli.settle_steps)):
            env.step(zero_action)

        now_pos, now_quat = get_rigid_pose(env, object_name, env_id)
        lin_vel, ang_vel = get_rigid_velocity(env, object_name, env_id)

        metrics = {
            "first_step_jump_m": float(torch.linalg.norm(first_step_pos - spawn_pos).item()),
            "xy_drift_m": float(torch.linalg.norm((now_pos - baseline_pos)[:2]).item()),
            "z_drop_m": float(max(0.0, (baseline_pos[2] - now_pos[2]).item())),
            "tilt_rad": tilt_angle_rad(baseline_quat, now_quat),
            "lin_vel_norm": float(torch.linalg.norm(lin_vel).item()),
            "ang_vel_norm": float(torch.linalg.norm(ang_vel).item()),
            "aabb_overlap_with": [],
        }
        status = classify_object(metrics, thresholds)
        if status == "stable":
            stable_heights.append(float(height_m))

        print(
            f"{height_m:8.4f} {status:>12s} {math.degrees(metrics['tilt_rad']):9.2f} "
            f"{metrics['z_drop_m']:12.4f} {metrics['xy_drift_m']:8.4f} "
            f"{max(0.0, (spawn_pos[2] - now_pos[2]).item()):13.4f} "
            f"{metrics['lin_vel_norm']:8.4f} {metrics['ang_vel_norm']:8.4f}",
            flush=True,
        )

        for _ in range(int(args_cli.dwell_steps)):
            env.step(zero_action)

    if stable_heights:
        print(f"[drop-sweep] max_stable_height_m={max(stable_heights):.4f}", flush=True)
    else:
        print("[drop-sweep] max_stable_height_m=None", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
