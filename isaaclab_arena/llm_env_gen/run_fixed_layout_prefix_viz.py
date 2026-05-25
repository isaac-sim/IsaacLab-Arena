# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay one solved multi-object layout while adding objects incrementally.

This is a diagnosis tool for separating solver layout quality from object-level
stability. It solves/builds the requested environment once, captures the full
layout from ``--source_env_id``, remaps those relative poses into
``--target_env_id``, and then replays prefixes of the object list without
re-solving.
"""

from __future__ import annotations

import math
import sys

import torch
import warp as wp

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
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


def _add_cli_args(parser, target_env_default: int = 0, include_start_count: bool = True) -> None:
    group = parser.add_argument_group("Fixed Layout Prefix Replay")
    group.add_argument("--source_env_id", type=int, default=1, help="Env index to capture the solved full layout from.")
    group.add_argument(
        "--target_env_id",
        type=int,
        default=target_env_default,
        help="Env index to replay the captured layout into.",
    )
    if include_start_count:
        group.add_argument("--start_count", type=int, default=2, help="First prefix size to replay.")
    group.add_argument("--settle_steps", type=int, default=60, help="Steps to run before the first metric readout.")
    group.add_argument("--dwell_steps", type=int, default=500, help="Additional steps to run for visual inspection.")
    group.add_argument(
        "--table_as_base",
        action="store_true",
        default=False,
        help="Spawn office_table as a BASE asset instead of a RIGID object for table-setup ablation.",
    )
    group.add_argument(
        "--object_max_depenetration_velocity",
        type=float,
        default=None,
        help="Override max_depenetration_velocity for replayed object assets.",
    )
    group.add_argument(
        "--object_solver_position_iterations",
        type=int,
        default=None,
        help="Override solver_position_iteration_count for replayed object assets.",
    )
    group.add_argument(
        "--object_contact_offset",
        type=float,
        default=None,
        help="Override contact_offset for replayed object assets.",
    )
    group.add_argument(
        "--object_rest_offset",
        type=float,
        default=None,
        help="Override rest_offset for replayed object assets.",
    )
    group.add_argument(
        "--object_restitution",
        type=float,
        default=None,
        help="Override rigid-body material restitution for replayed object assets.",
    )


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
        _apply_runtime_restitution_override(env, args_cli)

        try:
            _run_replay(env, arena_builder.arena_env, args_cli)
        finally:
            teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
            env.close()

    return 0


def _apply_diagnostic_overrides(args_cli) -> None:
    """Apply temporary asset config overrides for physics ablations.

    These overrides intentionally avoid convex hull. They test whether the fixed
    layout failure is sensitive to table spawning or common PhysX contact knobs.
    """
    if not any(
        [
            args_cli.table_as_base,
            args_cli.object_max_depenetration_velocity is not None,
            args_cli.object_solver_position_iterations is not None,
            args_cli.object_contact_offset is not None,
            args_cli.object_rest_offset is not None,
            args_cli.object_restitution is not None,
        ]
    ):
        return

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()

    if args_cli.table_as_base:
        office_table_cls = registry.get_asset_by_name("office_table")
        office_table_cls.object_type = ObjectType.BASE
        print("[prefix-viz] override: office_table object_type=BASE", flush=True)

    rigid_kwargs = {}
    if args_cli.object_max_depenetration_velocity is not None:
        rigid_kwargs["max_depenetration_velocity"] = float(args_cli.object_max_depenetration_velocity)
    if args_cli.object_solver_position_iterations is not None:
        rigid_kwargs["solver_position_iteration_count"] = int(args_cli.object_solver_position_iterations)

    collision_kwargs = {}
    if args_cli.object_contact_offset is not None:
        collision_kwargs["contact_offset"] = float(args_cli.object_contact_offset)
    if args_cli.object_rest_offset is not None:
        collision_kwargs["rest_offset"] = float(args_cli.object_rest_offset)

    for object_name in args_cli.objects:
        object_cls = registry.get_asset_by_name(object_name)
        spawn_cfg_addon = dict(getattr(object_cls, "spawn_cfg_addon", {}) or {})
        if rigid_kwargs:
            spawn_cfg_addon["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(**rigid_kwargs)
        if collision_kwargs:
            spawn_cfg_addon["collision_props"] = sim_utils.CollisionPropertiesCfg(**collision_kwargs)
        object_cls.spawn_cfg_addon = spawn_cfg_addon

    print(
        "[prefix-viz] object overrides: rigid={} collision={} restitution={}".format(
            rigid_kwargs,
            collision_kwargs,
            args_cli.object_restitution,
        ),
        flush=True,
    )


def _apply_runtime_restitution_override(env, args_cli) -> None:
    if args_cli.object_restitution is None:
        return

    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    material_path = "/World/diagnostic_object_physics_material"
    material_cfg = sim_utils.RigidBodyMaterialCfg(restitution=float(args_cli.object_restitution))
    material_cfg.func(material_path, material_cfg)

    bind_count = 0
    for object_name in args_cli.objects:
        for env_id in range(int(args_cli.num_envs)):
            root_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{object_name}")
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    if sim_utils.bind_physics_material(prim.GetPath(), material_path, stage=stage):
                        bind_count += 1

    print(
        f"[prefix-viz] runtime restitution override: restitution={args_cli.object_restitution} "
        f"material={material_path} bound_colliders={bind_count}",
        flush=True,
    )


def _run_replay(env, arena_env, args_cli) -> None:
    names = list(args_cli.objects)
    source_env_id = int(args_cli.source_env_id)
    target_env_id = int(args_cli.target_env_id)
    num_envs = int(args_cli.num_envs)
    assert 0 <= source_env_id < num_envs
    assert 0 <= target_env_id < num_envs
    assert 1 <= int(args_cli.start_count) <= len(names)

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
        f"[prefix-viz] captured full layout from env={source_env_id:02d}; "
        f"replaying in env={target_env_id:02d}",
        flush=True,
    )
    for name in names:
        rel = source_poses[name][:3] - env.unwrapped.scene.env_origins[source_env_id]
        pos = replay_poses[name][:3]
        print(
            f"[prefix-viz] {name}: rel=({rel[0].item():.3f}, {rel[1].item():.3f}, {rel[2].item():.3f}) "
            f"target=({pos[0].item():.3f}, {pos[1].item():.3f}, {pos[2].item():.3f})",
            flush=True,
        )

    for count in range(int(args_cli.start_count), len(names) + 1):
        active_names = names[:count]
        env.reset()
        _write_prefix_layout(env, names, active_names, replay_poses, target_env_id, env_ids, target_env_ids)

        overlap_pairs = compute_aabb_overlap_pairs(env, arena_env, active_names, target_env_id)
        overlap_partners = _overlap_partners(active_names, overlap_pairs)
        print(f"\n[prefix-viz] active_count={count} active={active_names}", flush=True)
        print(f"[prefix-viz] initial_overlaps={overlap_pairs}", flush=True)

        spawn_pose = {name: get_rigid_pose(env, name, target_env_id) for name in active_names}
        env.step(zero_action)
        first_step_pose = {name: get_rigid_pose(env, name, target_env_id) for name in active_names}

        for _ in range(int(args_cli.settle_steps)):
            env.step(zero_action)
        _print_metrics("settle", env, active_names, target_env_id, spawn_pose, first_step_pose, overlap_partners, thresholds)

        for _ in range(int(args_cli.dwell_steps)):
            env.step(zero_action)
        _print_metrics("dwell", env, active_names, target_env_id, spawn_pose, first_step_pose, overlap_partners, thresholds)


def _capture_root_poses(env, names: list[str], env_id: int) -> dict[str, torch.Tensor]:
    return {
        name: wp.to_torch(env.unwrapped.scene.rigid_objects[name].data.root_pose_w)[env_id].clone()
        for name in names
    }


def _remap_poses_to_target_env(
    env,
    names: list[str],
    source_poses: dict[str, torch.Tensor],
    source_env_id: int,
    target_env_id: int,
) -> dict[str, torch.Tensor]:
    origins = env.unwrapped.scene.env_origins
    source_origin = origins[source_env_id]
    target_origin = origins[target_env_id]
    replay_poses = {}
    for name in names:
        pose = source_poses[name].clone()
        relative_xyz = pose[:3] - source_origin
        pose[:3] = target_origin + relative_xyz
        replay_poses[name] = pose
    return replay_poses


def _write_prefix_layout(
    env,
    names: list[str],
    active_names: list[str],
    replay_poses: dict[str, torch.Tensor],
    target_env_id: int,
    env_ids: torch.Tensor,
    target_env_ids: torch.Tensor,
) -> None:
    active = set(active_names)
    origins = env.unwrapped.scene.env_origins
    for idx, name in enumerate(names):
        asset = env.unwrapped.scene.rigid_objects[name]
        if name in active:
            asset.write_root_pose_to_sim(replay_poses[name].unsqueeze(0), env_ids=target_env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.unwrapped.device), env_ids=target_env_ids)
        else:
            inactive_pose = replay_poses[name].repeat(len(target_env_ids), 1)
            inactive_pose[:, :3] = origins[target_env_id] + torch.tensor(
                [20.0 + 2.0 * idx, 20.0, 2.0], device=env.unwrapped.device
            )
            asset.write_root_pose_to_sim(inactive_pose, env_ids=target_env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.unwrapped.device), env_ids=target_env_ids)

        # Move all non-target env copies away to keep the viewer focused on one replay.
        non_target_env_ids = env_ids[env_ids != target_env_id]
        if len(non_target_env_ids) == 0:
            continue
        far_pose = replay_poses[name].repeat(len(non_target_env_ids), 1)
        far_pose[:, :3] = origins[non_target_env_ids] + torch.tensor(
            [20.0 + 2.0 * idx, 20.0, 2.0], device=env.unwrapped.device
        )
        asset.write_root_pose_to_sim(far_pose, env_ids=non_target_env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros(len(non_target_env_ids), 6, device=env.unwrapped.device),
            env_ids=non_target_env_ids,
        )


def _overlap_partners(active_names: list[str], overlap_pairs: list[dict]) -> dict[str, list[str]]:
    partners = {name: [] for name in active_names}
    for pair in overlap_pairs:
        partners[pair["a"]].append(pair["b"])
        partners[pair["b"]].append(pair["a"])
    return partners


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
    print(f"[prefix-viz] {stage} overall={overall} | " + ", ".join(chunks), flush=True)


if __name__ == "__main__":
    sys.exit(main())
