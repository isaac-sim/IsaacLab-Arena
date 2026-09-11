# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter generation from a constructed scene, independent of relation solving."""

from __future__ import annotations

import hashlib
import math
import torch
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_error_magnitude

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.isaac_cap.clutter.drop_poses import (
    ClutterDropParams,
    DropOrder,
    MemberDropParams,
    OccupiedFootprint,
    compute_drop_poses,
)
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


@dataclass(frozen=True)
class ClutterGroup:
    """Offline release parameters for ordinary rigid objects identified by scene keys."""

    support: str
    """Scene key of the static or kinematic support."""

    objects: tuple[str, ...]
    """Scene keys of the objects to drop, in release order."""

    spread: float = 1.0
    """Fraction of the support footprint used for release positions."""

    gap_m: float = 0.03
    """Vertical gap above overlapping release footprints."""

    clearance_m: float = 0.01
    """Release clearance above the support surface."""

    random_yaw: bool = True
    """Sample world-Z yaw while retaining the scene's authored base rotations."""

    drop_order: DropOrder = DropOrder.AS_LISTED
    """Order in which release poses are assigned."""

    def __post_init__(self) -> None:
        assert self.objects and len(set(self.objects)) == len(self.objects), "Group objects must be nonempty and unique"
        assert self.support not in self.objects, "A support cannot be a clutter member"
        assert 0 < self.spread <= 1, "spread must be in (0, 1]"
        assert math.isfinite(self.gap_m) and self.gap_m >= 0, "gap_m must be finite and non-negative"
        assert math.isfinite(self.clearance_m) and self.clearance_m >= 0, "clearance_m must be finite and non-negative"


def settle_clutter(
    env: ManagerBasedEnv,
    groups: list[ClutterGroup],
    *,
    seed: int = 42,
    attempts: int = 5,
    params: ClutterSettleParams | None = None,
) -> list[dict[str, Pose]]:
    """Generate one resting layout per environment and restore the caller's scene state.

    Args:
        env: Constructed scene at the desired initial poses and articulation configuration.
        groups: Objects and supports identified by their Isaac Lab scene keys.
        seed: Seed for independent release samples in each environment and attempt.
        attempts: Maximum trials for each environment before failing.
        params: Physics time budget, quiet thresholds and containment tolerances.

    Returns:
        Environment-local poses for every dynamic rigid object, indexed by environment.
        Articulation configurations must remain within the passive drift tolerances.
    """
    env = env.unwrapped
    params = replace(params) if params is not None else ClutterSettleParams()
    assert groups, "At least one clutter group is required"
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

    accepted: dict[int, dict[str, Pose]] = {}
    failures: dict[int, list[str]] = {i: [] for i in range(env.num_envs)}
    try:
        for attempt in range(attempts):
            restore_scene()
            pending = [i for i in range(env.num_envs) if i not in accepted]
            for env_id in pending:
                local_boxes = {key: _box_for_env(box, env_id) for key, box in boxes.items()}
                poses = {key: _pose(value[env_id]) for key, value in initial_poses.items()}
                payload = f"{seed}:{env_id}:{attempt}".encode()
                generator = torch.Generator().manual_seed(
                    int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)
                )
                _release_objects(env, env_id, groups, poses, local_boxes, generator)
            trackers = {i: SettleTracker(params) for i in pending}
            for _ in range(max_steps // poll_steps):
                # Refresh actuator feedback while retaining the caller's control targets.
                for _ in range(poll_steps):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt)
                states = torch.stack([world.get_pose_e(key) for key in capture_keys], dim=1)
                for i in pending:
                    if not trackers[i].diverged:
                        trackers[i].update(states[i, :, :3], states[i, :, 3:])
                if all(trackers[i].settled or trackers[i].diverged for i in pending):
                    break
            for env_id in pending:
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


def _release_objects(
    env: ManagerBasedEnv,
    env_id: int,
    groups: list[ClutterGroup],
    poses: dict[str, Pose],
    boxes: dict[str, AxisAlignedBoundingBox],
    generator: torch.Generator,
) -> None:
    """Write noninterpenetrating release poses above each group's support."""
    unplaced = {key for group in groups for key in group.objects}
    env_ids = torch.tensor([env_id], device=env.device)
    for group in groups:
        support = poses[group.support]
        region = region_above_support(
            support.position_xyz, boxes[group.support], group.spread, support_rotation_xyzw=support.rotation_xyzw
        )
        occupied = []
        for key, pose in poses.items():
            if key in unplaced or key == group.support:
                continue
            bbox = boxes[key].rotated_by_quat(pose.rotation_xyzw).translated(pose.position_xyz)
            lower, upper = bbox.min_point[0], bbox.max_point[0]
            if float(upper[2]) <= region.floor_z:
                continue
            occupied.append(
                OccupiedFootprint(
                    center=(float(lower[0] + upper[0]) / 2, float(lower[1] + upper[1]) / 2),
                    half_extents=(float(upper[0] - lower[0]) / 2, float(upper[1] - lower[1]) / 2),
                    top_z=float(upper[2]),
                )
            )
        drops = compute_drop_poses(
            [boxes[key] for key in group.objects],
            region,
            ClutterDropParams(drop_order=group.drop_order),
            generator,
            occupied=occupied,
            base_rotations_xyzw=[poses[key].rotation_xyzw for key in group.objects],
            member_params=[
                MemberDropParams(clearance_m=group.clearance_m, gap_m=group.gap_m, random_yaw=group.random_yaw)
            ]
            * len(group.objects),
        )
        for key, drop in zip(group.objects, drops):
            pose = Pose(drop.position_xyz, drop.rotation_xyzw)
            poses[key] = pose
            T_E_O = pose.to_tensor(device=env.device).unsqueeze(0)
            T_W_O = T_E_O.clone()
            T_W_O[:, :3] += env.scene.env_origins[env_id]
            env.scene[key].write_root_pose_to_sim(T_W_O, env_ids=env_ids)
            env.scene[key].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=env_ids)
            unplaced.remove(key)
    env.scene.write_data_to_sim()
    env.sim.forward()


def _box_for_env(box: AxisAlignedBoundingBox, env_id: int) -> AxisAlignedBoundingBox:
    """Return one environment's local geometry bounds on the CPU."""
    row = env_id if box.num_envs > 1 else 0
    return AxisAlignedBoundingBox(box.min_point[row : row + 1].cpu(), box.max_point[row : row + 1].cpu())


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
