# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter generation from a constructed scene, using Arena relation placement."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_error_magnitude

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_base_rotation_per_asset, write_layout_to_sim
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.isaac_cap.clutter.geometry import (
    dynamic_rigid_object_keys,
    region_above_support,
    resting_extents,
    spawned_geometry_is_fixed,
    spawned_rigid_body_has_gravity,
)
from isaaclab_arena_environments.isaac_cap.clutter.validation import (
    ClutterSettleParams,
    SettleTracker,
    check_resting_poses,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


@dataclass(frozen=True)
class ClutterGroup:
    """Rigid objects sharing a fixed clutter support, identified by scene keys."""

    support: str
    """Scene key of the static or kinematic support."""

    objects: tuple[str, ...]
    """Scene keys of the objects to drop together."""


def groups_from_assets(assets: list[PlaceableAsset]) -> list[ClutterGroup]:
    """Read clutter supports and members from their ClutterOn relations."""
    members: dict[str, list[str]] = {}
    for asset in assets:
        relation = get_relation(asset, ClutterOn)
        if relation is not None:
            members.setdefault(relation.parent.get_scene_key(), []).append(asset.get_scene_key())
    assert members, "Environment must declare at least one ClutterOn relation"
    return [ClutterGroup(support, tuple(objects)) for support, objects in members.items()]


def settle_clutter(
    env: ManagerBasedEnv,
    assets: list[PlaceableAsset],
    *,
    seed: int = 42,
    attempts: int = 5,
    params: ClutterSettleParams | None = None,
    placer_params: ObjectPlacerParams | None = None,
) -> list[dict[str, Pose]]:
    """Generate one resting layout per environment and restore the caller's scene state.

    Args:
        env: Constructed scene at the desired initial poses and articulation configuration.
        assets: Scene assets carrying ClutterOn members, IsAnchor supports, and fixed neighbors.
        seed: Seed for independent release samples in each environment and attempt.
        attempts: Maximum trials for each environment before failing.
        params: Physics time budget, quiet thresholds and containment tolerances.
        placer_params: Solver, candidate-count and validation settings for release poses, before settling.

    Returns:
        Environment-local poses for every dynamic rigid object, indexed by environment.
        Articulation configurations must remain within the passive drift tolerances.
    """
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects

    env = env.unwrapped
    params = replace(params) if params is not None else ClutterSettleParams()
    placer_params = _release_placer_params(placer_params)
    requested_checks = (placer_params.enabled_checks or set()) | (placer_params.required_checks or set())
    requested_checks |= {PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION}
    reachability_targets = [asset.get_scene_key() for asset in assets if asset.requires_reachability]
    assert not reachability_targets, f"Offline settling cannot validate final-pose reachability: {reachability_targets}"
    groups = groups_from_assets(assets)
    placement_assets = [asset for asset in assets if asset.get_relations()]
    assert all(
        asset.is_anchor or get_relation(asset, ClutterOn) is not None for asset in placement_assets
    ), "Offline settling requires non-clutter placement to be resolved to fixed anchors first"
    collision_objects = get_passive_collision_objects(assets)
    uncovered = [
        asset.get_scene_key()
        for asset in assets
        if asset.get_scene_key() in env.scene.rigid_objects
        and asset not in placement_assets
        and asset not in collision_objects
    ]
    assert not uncovered, f"Passive rigid objects need fixed poses and collision geometry: {uncovered}"
    anchors = {asset for asset in placement_assets if asset.is_anchor}
    assert attempts > 0, "attempts must be positive"
    members = [key for group in groups for key in group.objects]
    assert len(set(members)) == len(members), "An object must belong to exactly one group"
    assert all(group.support not in members for group in groups), "Supports cannot be clutter members"
    gravity = env.cfg.sim.gravity
    assert gravity[0] == 0 and gravity[1] == 0 and gravity[2] < 0, "Offline pouring requires downward world-Z gravity"
    for group in groups:
        assert spawned_geometry_is_fixed(
            env.scene, group.support
        ), f"Support {group.support!r} must be static or kinematic"
        for key in group.objects:
            assert key in env.scene.rigid_objects, f"Clutter object {key!r} must be a rigid object"
            assert not spawned_geometry_is_fixed(env.scene, key), f"Clutter object {key!r} must be dynamic"
            assert spawned_rigid_body_has_gravity(env.scene, key), f"Clutter object {key!r} must have gravity enabled"

    dt = env.sim.get_physics_dt()
    poll_steps, max_steps = _step_budget(dt, params)
    assert (
        0.5 * abs(gravity[2]) * (poll_steps * dt) ** 2 > params.move_thresh_m
    ), "Poll interval is too short to distinguish free fall from rest"
    world = env.arena_world
    capture_keys = dynamic_rigid_object_keys(env.scene)
    geometry_keys = sorted(set(env.scene.rigid_objects) | {group.support for group in groups})
    passive_keys = sorted(set(geometry_keys) - set(members))
    boxes = {key: world.get_aabb_in_local_frame(key) for key in geometry_keys}
    initial_poses = {key: world.get_pose_e(key).clone() for key in geometry_keys}
    original_state = env.scene.get_state()
    original_targets = {
        key: (
            asset.data.joint_pos_target.torch.clone(),
            asset.data.joint_vel_target.torch.clone(),
            asset.data.joint_effort_target.torch.clone(),
        )
        for key, asset in env.scene.articulations.items()
    }
    initial_links = {key: asset.data.body_link_pose_w.torch.clone() for key, asset in env.scene.articulations.items()}

    def restore_scene() -> None:
        env.scene.reset_to(original_state)
        for key, (position, velocity, effort) in original_targets.items():
            asset = env.scene.articulations[key]
            asset.set_joint_position_target_index(target=position)
            asset.set_joint_velocity_target_index(target=velocity)
            asset.set_joint_effort_target_index(target=effort)
        env.scene.write_data_to_sim()
        env.sim.forward()

    candidate_count = env.num_envs * placer_params.max_placement_attempts
    accepted: dict[int, dict[str, Pose]] = {}
    failures: dict[int, list[str]] = {i: [] for i in range(env.num_envs)}
    try:
        for attempt in range(attempts):
            restore_scene()
            pending = [i for i in range(env.num_envs) if i not in accepted]
            placer = ObjectPlacer(replace(placer_params, placement_seed=seed + attempt * candidate_count))
            releases = placer.place_ranked_per_env(
                placement_assets, num_envs=env.num_envs, results_per_env=1, collision_objects=collision_objects
            )
            released = []
            for env_id in pending:
                release = releases[env_id][0]
                validation = release.validation_results
                missing_checks = requested_checks - validation.validation_results.keys()
                assert not missing_checks, f"Offline release validators did not run: {sorted(missing_checks)}"
                if not release.success:
                    checks = validation.get_failed_validation_check_names
                    failures[env_id].append(f"attempt {attempt + 1}: release placement failed: {checks}")
                    print(f"[clutter] env {env_id}, {failures[env_id][-1]}")
                    continue
                _release_objects(env, env_id, release, anchors, placement_assets)
                released.append(env_id)
            if not released:
                continue
            trackers = {i: SettleTracker(params) for i in released}
            for _ in range(max_steps // poll_steps):
                # Refresh actuator feedback while retaining the caller's control targets.
                for _ in range(poll_steps):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt)
                states = torch.stack([world.get_pose_e(key) for key in capture_keys], dim=1)
                for i in released:
                    if not trackers[i].diverged:
                        trackers[i].update(states[i, :, :3], states[i, :, 3:])
                if all(trackers[i].settled or trackers[i].diverged for i in released):
                    break
            for env_id in released:
                reasons = []
                rest_failure = trackers[env_id].failure_reason(capture_keys)
                if rest_failure:
                    reasons.append(rest_failure)
                for key in passive_keys:
                    drift = _pose_drift_reason(initial_poses[key][env_id], world.get_pose_e(key)[env_id], params)
                    if drift:
                        reasons.append(f"{key}: {drift}")
                for key, initial in initial_links.items():
                    current = env.scene.articulations[key].data.body_link_pose_w.torch[env_id]
                    drift = _pose_drift_reason(initial[env_id], current, params)
                    if drift:
                        reasons.append(f"{key} links: {drift}")
                if not reasons:
                    layout = {key: _pose(world.get_pose_e(key)[env_id]) for key in capture_keys}
                    for group in groups:
                        support = _pose(initial_poses[group.support][env_id])
                        region = region_above_support(
                            support.position_xyz,
                            _box_for_env(boxes[group.support], env_id),
                            support_rotation_xyzw=support.rotation_xyzw,
                        )
                        positions = torch.tensor([layout[key].position_xyz for key in group.objects])
                        extents = [
                            resting_extents(_box_for_env(boxes[key], env_id), layout[key].rotation_xyzw)
                            for key in group.objects
                        ]
                        verdict = check_resting_poses(positions, region, params, extents)
                        if not verdict.ok:
                            reasons.append(f"support {group.support}: {verdict.describe(list(group.objects))}")
                    if not reasons:
                        accepted[env_id] = layout
                if reasons:
                    failures[env_id].append(f"attempt {attempt + 1}: {'; '.join(reasons)}")
                    print(f"[clutter] env {env_id}, {failures[env_id][-1]}")
            if len(accepted) == env.num_envs:
                return [accepted[i] for i in range(env.num_envs)]
        rejected = {i: failures[i] for i in range(env.num_envs) if i not in accepted}
        raise AssertionError(f"No settled layout after {attempts} attempt(s): {rejected}")
    finally:
        restore_scene()


def _release_placer_params(params: ObjectPlacerParams | None) -> ObjectPlacerParams:
    """Release settings with mandatory geometry checks."""
    params = params if params is not None else ObjectPlacerParams()
    requested_checks = (params.enabled_checks or set()) | (params.required_checks or set())
    unsupported_checks = requested_checks & {PlacementCheck.IK_REACHABLE, PlacementCheck.PHYSICS_SETTLED}
    assert (
        not unsupported_checks
    ), f"Offline release validation cannot certify settled-pose checks: {sorted(unsupported_checks)}"
    if params.enabled_checks is not None and params.required_checks is not None:
        assert params.required_checks <= params.enabled_checks, "Required release checks must be enabled"
    geometry_checks = {PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION}
    return replace(
        params,
        enabled_checks=None if params.enabled_checks is None else params.enabled_checks | geometry_checks,
        required_checks=None if params.required_checks is None else params.required_checks | geometry_checks,
    )


def _release_objects(
    env: ManagerBasedEnv,
    env_id: int,
    layout: PlacementResult,
    anchors: set[PlaceableAsset],
    assets: list[PlaceableAsset],
) -> None:
    """Write a complete ObjectPlacer release layout with zero root velocities."""
    write_layout_to_sim(env, env_id, layout, anchors, get_base_rotation_per_asset(assets))
    env.scene.write_data_to_sim()
    env.sim.forward()


def _box_for_env(box: AxisAlignedBoundingBox, env_id: int) -> AxisAlignedBoundingBox:
    """Return one environment's local geometry bounds on the CPU."""
    return AxisAlignedBoundingBox(box.min_point[env_id : env_id + 1].cpu(), box.max_point[env_id : env_id + 1].cpu())


def _pose(value: torch.Tensor) -> Pose:
    """Convert a finite xyz/xyzw tensor of shape (7,) to an Arena pose."""
    assert torch.isfinite(value).all(), "Cannot cache a non-finite pose"
    return Pose(tuple(value[:3].tolist()), tuple(value[3:7].tolist()))


def _pose_drift_reason(initial: torch.Tensor, current: torch.Tensor, params: ClutterSettleParams) -> str | None:
    """Report excessive passive drift for poses shaped (..., 7), ordered xyz/xyzw."""
    if not torch.isfinite(current).all():
        return "non-finite passive pose"
    distance = float((current[..., :3] - initial[..., :3]).norm(dim=-1).max())
    angle = float(torch.rad2deg(quat_error_magnitude(current[..., 3:], initial[..., 3:])).max())
    if distance > params.passive_move_thresh_m or angle > params.passive_turn_thresh_deg:
        return (
            f"passive drift {distance:.6f} m, {angle:.3f} deg; limits "
            f"{params.passive_move_thresh_m:.6f} m, {params.passive_turn_thresh_deg:.3f} deg"
        )
    return None


def _step_budget(dt: float, params: ClutterSettleParams) -> tuple[int, int]:
    """Return poll and trial step counts that allow the required quiet windows."""
    poll_steps = math.ceil(params.poll_interval_s / dt)
    max_steps = math.floor(params.timeout_s / dt)
    required_polls = params.required_quiet_windows + 1
    assert max_steps // poll_steps >= required_polls, (
        f"timeout_s allows {max_steps // poll_steps} polls at physics dt={dt:g}; need {required_polls}. "
        "Increase timeout_s or reduce poll_interval_s."
    )
    return poll_steps, max_steps
