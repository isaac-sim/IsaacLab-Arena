# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay an exact subset from one solved multi-object layout.

This complements ``run_fixed_layout_prefix_viz.py``. The prefix tool is useful
for adding objects in order; this tool is useful for targeted checks such as
``spoon`` alone at the pose it had in the failing full env1 layout.
"""

from __future__ import annotations

import math
import sys

import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.llm_env_gen.run_fixed_layout_prefix_viz import (
    _add_cli_args as _add_prefix_diagnostic_args,
    _apply_diagnostic_overrides,
    _capture_root_poses,
    _overlap_partners,
    _remap_poses_to_target_env,
    _write_prefix_layout,
)
from isaaclab_arena.llm_env_gen.stability_utils import (
    classify_object,
    compute_aabb_overlap_pairs,
    get_rigid_pose,
    get_rigid_velocity,
    tilt_angle_rad,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def _add_cli_args(parser) -> None:
    _add_prefix_diagnostic_args(parser, target_env_default=1, include_start_count=False)
    group = parser.add_argument_group("Fixed Layout Subset Replay")
    group.add_argument(
        "--active_objects",
        type=str,
        default=None,
        help="Comma-separated object subset to replay at the positions captured from the full layout.",
    )
    group.add_argument("--settle_steps", type=int, default=60, help="Steps to run before the first metric readout.")
    group.add_argument("--dwell_steps", type=int, default=500, help="Additional steps to run for visual inspection.")


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    _add_cli_args(parser)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = parser.parse_args()

        if args_cli.seed is not None:
            set_seed(args_cli.seed)

        _apply_diagnostic_overrides(args_cli)
        arena_builder = get_arena_builder_from_cli(args_cli)
        env, _ = arena_builder.make_registered_and_return_cfg()
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)

        try:
            _run_replay(env, arena_builder.arena_env, args_cli)
        finally:
            teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
            env.close()

    return 0


def _run_replay(env, arena_env, args_cli) -> None:
    names = list(args_cli.objects)
    assert args_cli.active_objects, "--active_objects must be provided before the environment name."
    active_names = [name.strip() for name in args_cli.active_objects.split(",") if name.strip()]
    unknown = sorted(set(active_names) - set(names))
    assert not unknown, f"--active_objects contains objects not present in --objects: {unknown}"

    source_env_id = int(args_cli.source_env_id)
    target_env_id = int(args_cli.target_env_id)
    num_envs = int(args_cli.num_envs)
    assert 0 <= source_env_id < num_envs
    assert 0 <= target_env_id < num_envs

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env_ids = torch.arange(num_envs, device=env.unwrapped.device)
    target_env_ids = torch.tensor([target_env_id], device=env.unwrapped.device)
    thresholds = {
        "first_step_jump_thresh": 0.02,
        "z_drop_thresh": 0.30,
        "tilt_thresh_rad": math.radians(20.0),
        "xy_drift_thresh": 0.05,
        "vel_thresh_lin": 0.05,
        "vel_thresh_ang": 0.20,
    }

    env.reset()
    source_poses = _capture_root_poses(env, names, source_env_id)
    replay_poses = _remap_poses_to_target_env(env, names, source_poses, source_env_id, target_env_id)

    print(
        f"[subset-viz] captured full layout from env={source_env_id:02d}; "
        f"replaying active={active_names} in env={target_env_id:02d}",
        flush=True,
    )
    for name in active_names:
        rel = source_poses[name][:3] - env.unwrapped.scene.env_origins[source_env_id]
        pos = replay_poses[name][:3]
        print(
            f"[subset-viz] {name}: rel=({rel[0].item():.3f}, {rel[1].item():.3f}, {rel[2].item():.3f}) "
            f"target=({pos[0].item():.3f}, {pos[1].item():.3f}, {pos[2].item():.3f})",
            flush=True,
        )

    env.reset()
    _write_prefix_layout(env, names, active_names, replay_poses, target_env_id, env_ids, target_env_ids)

    overlap_pairs = compute_aabb_overlap_pairs(env, arena_env, active_names, target_env_id)
    overlap_partners = _overlap_partners(active_names, overlap_pairs)
    print(f"[subset-viz] initial_overlaps={overlap_pairs}", flush=True)

    spawn_pose = {name: get_rigid_pose(env, name, target_env_id) for name in active_names}
    env.step(zero_action)
    first_step_pose = {name: get_rigid_pose(env, name, target_env_id) for name in active_names}

    for _ in range(int(args_cli.settle_steps)):
        env.step(zero_action)
    _print_metrics("settle", env, active_names, target_env_id, spawn_pose, first_step_pose, overlap_partners, thresholds)

    for _ in range(int(args_cli.dwell_steps)):
        env.step(zero_action)
    _print_metrics("dwell", env, active_names, target_env_id, spawn_pose, first_step_pose, overlap_partners, thresholds)


def _print_metrics(
    stage: str,
    env,
    active_names: list[str],
    env_id: int,
    spawn_pose: dict[str, tuple[torch.Tensor, torch.Tensor]],
    first_step_pose: dict[str, tuple[torch.Tensor, torch.Tensor]],
    overlap_partners: dict[str, list[str]],
    thresholds: dict,
) -> None:
    statuses = []
    chunks = []
    for name in active_names:
        spawn_pos, spawn_quat = spawn_pose[name]
        t1_pos, _ = first_step_pose[name]
        now_pos, now_quat = get_rigid_pose(env, name, env_id)
        lin_vel, ang_vel = get_rigid_velocity(env, name, env_id)
        metrics = {
            "first_step_jump_m": float(torch.linalg.norm(t1_pos - spawn_pos).item()),
            "xy_drift_m": float(torch.linalg.norm((now_pos - spawn_pos)[:2]).item()),
            "z_drop_m": float(max(0.0, (spawn_pos[2] - now_pos[2]).item())),
            "tilt_rad": tilt_angle_rad(spawn_quat, now_quat),
            "lin_vel_norm": float(torch.linalg.norm(lin_vel).item()),
            "ang_vel_norm": float(torch.linalg.norm(ang_vel).item()),
            "aabb_overlap_with": overlap_partners.get(name, []),
        }
        status = classify_object(metrics, thresholds)
        statuses.append(status)
        chunks.append(
            f"{name}={status}(tilt={math.degrees(metrics['tilt_rad']):.1f},"
            f"drop={metrics['z_drop_m']:.3f},xy={metrics['xy_drift_m']:.3f},"
            f"jump={metrics['first_step_jump_m']:.3f})"
        )
    overall = "stable" if all(status == "stable" for status in statuses) else next(
        status for status in statuses if status != "stable"
    )
    print(f"[subset-viz] {stage} overall={overall} | " + ", ".join(chunks), flush=True)


if __name__ == "__main__":
    sys.exit(main())
