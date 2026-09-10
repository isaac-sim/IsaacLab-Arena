# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics preparation and validation of replayable clutter layouts."""

from __future__ import annotations

import math
import torch
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.utils.math import quat_error_magnitude

from isaaclab_arena.relations.clutter_groups import ClutterGroup, get_clutter_groups
from isaaclab_arena.relations.clutter_pour import region_above_support, resting_extents
from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, check_resting_poses
from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_pool_validation import get_pose_capture_assets, validate_pool_layouts
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.relations.placement_result import PlacementResult


def prepare_clutter_layouts(
    env: IsaacLabArenaManagerBasedRLEnv, placement_pool: PooledObjectPlacer, params: ClutterSettleParams | None = None
) -> None:
    """Settle, validate and retain replayable layouts outside the episode/reset lifecycle.

    Args:
        env: Constructed Arena environment whose scene matches the pool. Preparation uses
            configured scene defaults and steps all environments together; caller state is restored.
        placement_pool: Explicit pool to prepare, independent of any reset event registration.
        params: Physical-time budget and validation thresholds. Configure these before the first
            preparation; an already-prepared pool cannot be prepared again.
    """
    from isaaclab.envs.mdp.events import reset_scene_to_default

    env = env.unwrapped
    params = replace(params) if params is not None else ClutterSettleParams()
    groups = get_clutter_groups(placement_pool.objects)
    if not groups:
        return
    assert not placement_pool.recycle_layouts, (
        "Clutter pool is already prepared. Configure clutter_settle_params before building, "
        "or set settle_clutter_on_build=False and prepare once before reset."
    )
    assert placement_pool.num_envs == env.num_envs, "Pool and scene environment counts must match"
    original_state = env.scene.get_state()
    original_targets = {
        key: (
            asset.data.joint_pos_target.torch.clone(),
            asset.data.joint_vel_target.torch.clone(),
            asset.data.joint_effort_target.torch.clone(),
        )
        for key, asset in env.scene.articulations.items()
    }
    try:
        # Before the first reset, articulation state may still be the USD rest pose rather
        # than the configured Arena pose. Use the same default-state operation as reset.
        reset_scene_to_default(env, torch.arange(env.num_envs, device=env.device), reset_joint_targets=True)
        _settle_clutter_layouts(env, placement_pool, groups, params)
    finally:
        env.scene.reset_to(original_state)
        for key, (position, velocity, effort) in original_targets.items():
            asset = env.scene.articulations[key]
            asset.set_joint_position_target_index(target=position)
            asset.set_joint_velocity_target_index(target=velocity)
            asset.set_joint_effort_target_index(target=effort)
        env.scene.write_data_to_sim()
        env.sim.forward()


def _settle_clutter_layouts(
    env: IsaacLabArenaManagerBasedRLEnv,
    placement_pool: PooledObjectPlacer,
    groups: list[ClutterGroup],
    params: ClutterSettleParams,
) -> None:
    """Settle and filter candidate layouts against configured scene defaults."""
    gravity = env.cfg.sim.gravity
    assert gravity[0] == 0 and gravity[1] == 0 and gravity[2] < 0, "Clutter pouring requires downward world-Z gravity"
    env.scene.write_data_to_sim()
    env.sim.forward()
    world = env.arena_world
    for group in groups:
        assert world.is_geometry_fixed(group.support.get_scene_key()), (
            f"Support {group.support.name!r} has dynamic spawned geometry. "
            "Prepared piles require a static or kinematic support."
        )
        for member in group.members:
            key = member.get_scene_key()
            assert (
                key in env.scene.rigid_objects
                and not world.is_geometry_fixed(key)
                and world.rigid_body_has_gravity(key)
            ), (
                f"Clutter member {member.name!r} must be a dynamic rigid object with gravity enabled. "
                "A prepared pose cannot represent articulated or deformable member state."
            )
    allowed_contacts = set()
    for group in groups:
        allowed_contacts.update(frozenset(pair) for pair in combinations(group.members, 2))
        allowed_contacts.update(frozenset((member, group.support)) for member in group.members)
    assets = placement_pool.objects
    # Use live geometry in the same frames as the captured poses, including nested support references.
    boxes = {
        asset: (
            world.get_aabb_in_local_frame(asset.get_scene_key())
            if asset.get_scene_key() not in env.scene.articulations
            else asset.get_bounding_box()
        )
        for asset in assets
    }
    captured_keys = {asset.get_scene_key() for asset in get_pose_capture_assets(assets)}
    passive_keys = (set(env.scene.rigid_objects) | set(env.scene.articulations)) - captured_keys
    initial_passive = {key: world.get_pose_e(key).clone() for key in passive_keys}
    initial_links = {
        key: wp.to_torch(articulation.data.body_link_pose_w).clone()
        for key, articulation in env.scene.articulations.items()
    }

    def validate_final(env_id: int, layout: PlacementResult) -> None:
        poses = {}
        local_boxes = {}
        for asset in assets:
            state = world.get_pose_e(asset.get_scene_key())[env_id].cpu()
            poses[asset] = Pose(tuple(state[:3].tolist()), tuple(state[3:7].tolist()))
            box = boxes[asset]
            row = env_id if box.min_point.shape[0] > 1 else 0
            local_boxes[asset] = AxisAlignedBoundingBox(
                box.min_point[row : row + 1].cpu(), box.max_point[row : row + 1].cpu()
            )
        checks = placement_pool.validate_poses(poses, local_boxes, allowed_contacts)
        # A fixed obstacle that got pushed cannot be replayed from its original reset pose.
        passive_unchanged = True
        for key, initial in initial_passive.items():
            passive_unchanged &= _poses_unchanged(
                initial[env_id], world.get_pose_e(key)[env_id], params, f"env {env_id}, {key}"
            )
        for key, initial in initial_links.items():
            current = wp.to_torch(env.scene.articulations[key].data.body_link_pose_w)[env_id]
            passive_unchanged &= _poses_unchanged(initial[env_id], current, params, f"env {env_id}, {key} links")
        checks[PlacementCheck.PASSIVE_POSES_UNCHANGED] = passive_unchanged
        contained = True
        for group in groups:
            pose = poses[group.support]
            region = region_above_support(
                pose.position_xyz, local_boxes[group.support], support_rotation_xyzw=pose.rotation_xyzw
            )
            positions = torch.tensor([layout.positions[member] for member in group.members])
            extents = [resting_extents(member, layout, local_boxes[member]) for member in group.members]
            verdict = check_resting_poses(positions, region, params, extents)
            if not verdict.ok:
                print(
                    f"[clutter] env {env_id}, group {group.name!r}: {verdict.describe([m.name for m in group.members])}"
                )
            contained &= verdict.ok
        checks[PlacementCheck.CLUTTER_CONTAINED] = bool(contained)
        layout.validation_results.validation_results.update(checks)
        required = layout.validation_results.required_checks
        accepted = all(passed for check, passed in checks.items() if required is None or check in required)
        layout.validation_results.validation_results[PlacementCheck.FINAL_POSES_VALIDATED] = (
            accepted and checks[PlacementCheck.POSITION_CONSTRAINTS] and contained and passive_unchanged
        )

    dt = env.sim.get_physics_dt()
    validate_pool_layouts(
        env,
        placement_pool,
        PhysicsSettleParams(num_steps=math.ceil(params.timeout_s / (dt * env.cfg.decimation))),
        capture_settled_poses=True,
        pose_settle_params=params,
        poll_every=math.ceil(params.poll_interval_s / dt),
        validate_captured_layout=validate_final,
    )

    def keep(env_id: int, layout: PlacementResult) -> bool:
        if not layout.is_prepared:
            print(f"[clutter] rejected layout in env {env_id}: {layout.validation_results.report()}")
        return layout.is_prepared

    kept, rejected = placement_pool.retain_layouts(keep, include_consumed=True)
    placement_pool.recycle_layouts = True
    print(f"[clutter] prepared {kept} layout(s); rejected {rejected}")


def _poses_unchanged(initial: torch.Tensor, current: torch.Tensor, params: ClutterSettleParams, label: str) -> bool:
    """Check total passive drift for poses shaped (..., 7), ordered xyz/xyzw."""
    if not torch.isfinite(current).all():
        print(f"[clutter] {label}: non-finite passive pose")
        return False
    distance = float((current[..., :3] - initial[..., :3]).norm(dim=-1).max())
    angle = float(torch.rad2deg(quat_error_magnitude(current[..., 3:], initial[..., 3:])).max())
    unchanged = distance <= params.passive_move_thresh_m and angle <= params.passive_turn_thresh_deg
    if not unchanged:
        print(
            f"[clutter] {label}: passive drift {distance:.6f} m, {angle:.3f} deg; "
            f"limits {params.passive_move_thresh_m:.6f} m, {params.passive_turn_thresh_deg:.3f} deg"
        )
    return unchanged
