# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the object stability issue fixed by the placement MR.

This script is intentionally small and CLI-oriented so reviewers can compare the
old behavior with the proposed fixes in the same scene:

* ``--origin_main_solver`` emulates the origin/main solver behavior: 3D
  no-collision loss (``xy_only=False``) and anchor/table comparisons.
* The fixed run omits ``--origin_main_solver`` so no-collision is XY-only and
  anchors are excluded (``xy_only=True``), then adds ``--force_convex_hull`` to
  use simpler collision geometry for fragile scanned objects.

The baseline command should report ``overall=fell_off``. The fixed command
should report ``overall=stable``.

Visual baseline repro::

    /isaac-sim/python.sh isaaclab_arena/evaluation/object_stability_repro.py \
        --viz kit --num_envs 4 --env_spacing 4.0 --seed 123 --placement_seed 123 \
        --all_envs --settle_steps 60 --dwell_steps 3000 --origin_main_solver --xy_only false \
        gr1_table_multi_object_no_collision --embodiment gr1_joint \
        --objects parmesan_cheese_canister_hope_robolab mustard_bottle_hope_robolab \
            bbq_sauce_bottle_hope_robolab milk_carton_hope_robolab

Visual fixed run::

    /isaac-sim/python.sh isaaclab_arena/evaluation/object_stability_repro.py \
        --viz kit --num_envs 4 --env_spacing 4.0 --seed 123 --placement_seed 123 \
        --all_envs --settle_steps 60 --dwell_steps 3000 --xy_only true --force_convex_hull \
        gr1_table_multi_object_no_collision --embodiment gr1_joint \
        --objects parmesan_cheese_canister_hope_robolab mustard_bottle_hope_robolab \
            bbq_sauce_bottle_hope_robolab milk_carton_hope_robolab
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import torch
from typing import Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def _parse_bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in ("1", "true", "yes", "on"):
        return True
    if value_lower in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0")


def _add_repro_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Object stability repro")
    group.add_argument("--env_id", type=int, default=0, help="Environment index to check.")
    group.add_argument("--all_envs", action="store_true", default=False, help="Check every environment.")
    group.add_argument("--settle_steps", type=int, default=60, help="Zero-action steps before final metrics.")
    group.add_argument(
        "--dwell_steps", type=int, default=0, help="Extra zero-action steps after metrics for viewer inspection."
    )
    group.add_argument(
        "--placement_pool_size",
        type=int,
        default=None,
        help="Number of placement layouts to pre-solve. Default is --num_envs for this repro.",
    )
    group.add_argument(
        "--placement_max_attempts",
        type=int,
        default=1,
        help="Placement attempts per requested layout. Default 1 keeps the visual repro fast.",
    )
    group.add_argument("--json", action="store_true", default=False, help="Emit final machine-readable JSON.")
    group.add_argument(
        "--origin_main_solver",
        action="store_true",
        default=False,
        help=(
            "Emulate origin/main no-collision behavior: use 3D overlap loss and include anchor/table objects in"
            " no-collision comparisons. If omitted, the proposed solver behavior is used: xy_only=True and"
            " anchors excluded."
        ),
    )
    group.add_argument(
        "--xy_only",
        type=_parse_bool,
        default=None,
        metavar="{true,false}",
        help=(
            "Set no-collision XY-only mode explicitly. Use '--xy_only false' for the origin/main baseline and"
            " '--xy_only true' for the proposed tabletop solver behavior. If omitted, this defaults from"
            " --origin_main_solver."
        ),
    )
    group.add_argument(
        "--force_convex_hull",
        action="store_true",
        default=False,
        help=(
            "Replace convexDecomposition mesh collisions with convexHull after scene creation. This isolates the"
            " scanned-object collision mesh fix."
        ),
    )
    group.add_argument("--first_step_jump_thresh", type=float, default=0.02)
    group.add_argument("--z_drop_thresh", type=float, default=0.30)
    group.add_argument("--xy_drift_thresh", type=float, default=0.05)
    group.add_argument("--tilt_thresh_deg", type=float, default=20.0)
    group.add_argument("--vel_thresh_lin", type=float, default=0.05)
    group.add_argument("--vel_thresh_ang", type=float, default=0.20)


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    _add_repro_args(parser)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = parser.parse_args()

        # The baseline keeps origin/main semantics. The fixed run treats tabletop
        # no-collision as 2D packing and lets On(table) control height.
        args_cli.no_collision_xy_only = (
            not args_cli.origin_main_solver if args_cli.xy_only is None else bool(args_cli.xy_only)
        )
        args_cli.no_collision_include_anchors = args_cli.origin_main_solver
        args_cli.resolve_on_reset = False
        if args_cli.placement_pool_size is None:
            args_cli.placement_pool_size = int(args_cli.num_envs)

        if getattr(args_cli, "seed", None) is not None:
            set_seed(args_cli.seed)

        print(
            "[stability-repro] building env with placement_pool_size={} placement_max_attempts={}".format(
                args_cli.placement_pool_size,
                args_cli.placement_max_attempts,
            ),
            flush=True,
        )
        arena_builder = get_arena_builder_from_cli(args_cli)
        arena_builder.arena_env.force_convex_hull = bool(args_cli.force_convex_hull)
        env, _ = arena_builder.make_registered_and_return_cfg()
        if getattr(args_cli, "seed", None) is not None:
            set_seed(args_cli.seed, env)

        try:
            return _run_check(env, arena_builder.arena_env, args_cli)
        finally:
            env.close()


def _run_check(env, arena_env, args_cli: argparse.Namespace) -> int:
    names = _collect_checkable_objects(arena_env)
    assert names, "No non-anchor rigid objects found to check."
    env_ids = list(range(int(args_cli.num_envs))) if args_cli.all_envs else [int(args_cli.env_id)]
    solver_mode = "origin_main_baseline" if args_cli.origin_main_solver else "proposed_xy_only"
    print(
        "[stability-repro] solver_mode={} no_collision_xy_only={} no_collision_include_anchors={} "
        "force_convex_hull={}".format(
            solver_mode,
            args_cli.no_collision_xy_only,
            args_cli.no_collision_include_anchors,
            args_cli.force_convex_hull,
        ),
        flush=True,
    )
    print(
        "[stability-repro] objects={} envs={}".format(
            names,
            env_ids,
        ),
        flush=True,
    )

    # Measure both immediate spawn response and post-settle behavior so the
    # output catches contact explosions, tipping, sliding, and falling.
    env.reset()
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    spawn_poses = _snapshot_poses(env, names, env_ids)
    env.step(zero_action)
    first_step_poses = _snapshot_poses(env, names, env_ids)
    for _ in range(int(args_cli.settle_steps)):
        env.step(zero_action)

    thresholds = _thresholds(args_cli)
    per_env = {
        env_id: _compute_env_metrics(env, names, env_id, spawn_poses[env_id], first_step_poses[env_id], thresholds)
        for env_id in env_ids
    }
    overall = _overall_status(per_env)

    for env_id, metrics_by_name in per_env.items():
        for name, metrics in metrics_by_name.items():
            print(
                "[stability-repro] env={} {}: {} | jump1={:.4f}m xy_drift={:.4f}m z_drop={:.4f}m "
                "tilt={:.1f}deg |v|={:.4f}m/s |w|={:.4f}rad/s".format(
                    env_id,
                    name,
                    metrics["status"],
                    metrics["first_step_jump_m"],
                    metrics["xy_drift_m"],
                    metrics["z_drop_m"],
                    math.degrees(metrics["tilt_rad"]),
                    metrics["lin_vel_norm"],
                    metrics["ang_vel_norm"],
                ),
                flush=True,
            )
    print(f"[stability-repro] overall={overall}", flush=True)

    if args_cli.json:
        print(
            json.dumps({
                "overall_status": overall,
                "origin_main_solver": args_cli.origin_main_solver,
                "force_convex_hull": args_cli.force_convex_hull,
                "objects": per_env,
            }),
            flush=True,
        )
        sys.stdout.flush()

    for _ in range(int(args_cli.dwell_steps)):
        env.step(zero_action)
    return 0 if overall == "stable" else 4


def _collect_checkable_objects(arena_env) -> list[str]:
    # Keep pxr-backed imports after SimulationApp starts. Importing them at
    # module load time can break Kit extension initialization.
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.relations.relations import IsAnchor

    names = []
    embodiment_name = getattr(arena_env.embodiment, "name", None)
    for asset in arena_env.scene.assets.values():
        if getattr(asset, "object_type", None) != ObjectType.RIGID:
            continue
        if asset.name == embodiment_name:
            continue
        if any(isinstance(relation, IsAnchor) for relation in getattr(asset, "get_relations", lambda: [])()):
            continue
        names.append(asset.name)
    return names


def _snapshot_poses(
    env, names: list[str], env_ids: list[int]
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    return {env_id: {name: _get_rigid_pose(env, name, env_id) for name in names} for env_id in env_ids}


def _get_rigid_pose(env, name: str, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    import warp as wp

    obj = env.unwrapped.scene.rigid_objects[name]
    return wp.to_torch(obj.data.root_pos_w)[env_id].clone(), wp.to_torch(obj.data.root_quat_w)[env_id].clone()


def _get_rigid_velocity(env, name: str, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    import warp as wp

    obj = env.unwrapped.scene.rigid_objects[name]
    return wp.to_torch(obj.data.root_lin_vel_w)[env_id].clone(), wp.to_torch(obj.data.root_ang_vel_w)[env_id].clone()


def _compute_env_metrics(env, names: list[str], env_id: int, spawn_poses, first_step_poses, thresholds: dict) -> dict:
    metrics_by_name = {}
    for name in names:
        spawn_pos, spawn_quat = spawn_poses[name]
        first_pos, _ = first_step_poses[name]
        settled_pos, settled_quat = _get_rigid_pose(env, name, env_id)
        lin_vel, ang_vel = _get_rigid_velocity(env, name, env_id)
        metrics: dict[str, Any] = {
            "first_step_jump_m": float(torch.linalg.norm(first_pos - spawn_pos).item()),
            "xy_drift_m": float(torch.linalg.norm(settled_pos[:2] - spawn_pos[:2]).item()),
            "z_drop_m": float(max(0.0, (spawn_pos[2] - settled_pos[2]).item())),
            "tilt_rad": _tilt_angle_rad(spawn_quat, settled_quat),
            "lin_vel_norm": float(torch.linalg.norm(lin_vel).item()),
            "ang_vel_norm": float(torch.linalg.norm(ang_vel).item()),
        }
        metrics["status"] = _classify(metrics, thresholds)
        metrics_by_name[name] = metrics
    return metrics_by_name


def _tilt_angle_rad(quat_init_wxyz: torch.Tensor, quat_now_wxyz: torch.Tensor) -> float:
    import isaaclab.utils.math as pose_utils

    z_init = pose_utils.matrix_from_quat(quat_init_wxyz)[:, 2]
    z_now = pose_utils.matrix_from_quat(quat_now_wxyz)[:, 2]
    return float(torch.acos(torch.clamp(torch.dot(z_init, z_now), -1.0, 1.0)).item())


def _thresholds(args_cli: argparse.Namespace) -> dict:
    return {
        "first_step_jump_thresh": float(args_cli.first_step_jump_thresh),
        "z_drop_thresh": float(args_cli.z_drop_thresh),
        "xy_drift_thresh": float(args_cli.xy_drift_thresh),
        "tilt_thresh_rad": math.radians(float(args_cli.tilt_thresh_deg)),
        "vel_thresh_lin": float(args_cli.vel_thresh_lin),
        "vel_thresh_ang": float(args_cli.vel_thresh_ang),
    }


def _classify(metrics: dict, thresholds: dict) -> str:
    if metrics["first_step_jump_m"] > thresholds["first_step_jump_thresh"]:
        return "spawn_collision"
    if metrics["z_drop_m"] > thresholds["z_drop_thresh"]:
        return "fell_off"
    if metrics["tilt_rad"] > thresholds["tilt_thresh_rad"]:
        return "tipped"
    if metrics["xy_drift_m"] > thresholds["xy_drift_thresh"]:
        return "slid"
    if metrics["lin_vel_norm"] > thresholds["vel_thresh_lin"] or metrics["ang_vel_norm"] > thresholds["vel_thresh_ang"]:
        return "unsettled"
    return "stable"


def _overall_status(per_env: dict[int, dict[str, dict]]) -> str:
    priority = ["spawn_collision", "fell_off", "tipped", "slid", "unsettled", "stable"]
    statuses = [metrics["status"] for metrics_by_name in per_env.values() for metrics in metrics_by_name.values()]
    return min(statuses, key=priority.index)


if __name__ == "__main__":
    sys.exit(main())
