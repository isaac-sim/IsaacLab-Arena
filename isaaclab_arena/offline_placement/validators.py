# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configurable checks of measured poses after offline physics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

from isaaclab_arena.relations.placement_validation import PlacementValidator, PlacementValidatorReport

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.offline_placement.scene_snapshot import SceneSnapshot
    from isaaclab_arena.offline_placement.settle import ClutterGroup
    from isaaclab_arena.offline_placement.validation import SettleTracker
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose


@dataclass
class PostPhysicsState:
    """Measured layout and reference state for one of N simulation environments."""

    env: ManagerBasedEnv
    """Simulation environment containing the measured bodies."""
    env_id: int
    """Absolute environment index."""
    poses: dict[str, Pose]
    """Final environment-local root poses, keyed by scene name."""
    snapshot: SceneSnapshot
    """Initial scene poses and articulation links for N environments."""
    passive_keys: list[str]
    """Bodies that must retain their initial poses."""
    groups: list[ClutterGroup]
    """Clutter members and their fixed supports."""
    boxes: dict[str, AxisAlignedBoundingBox]
    """Local bounds with min/max tensors shaped (N, 3), keyed by scene name."""
    rest_tracker: SettleTracker | None
    """Sampled motion history, or None when the rest check is disabled."""


@dataclass
class PostPhysicsPlacementValidator(PlacementValidator[PostPhysicsState, PlacementValidatorReport]):
    """A dataclass-configured check of a measured layout after physics."""

    stage: ClassVar[str] = "post_physics"
    enabled: bool = True
    """Whether this check must pass before accepting a layout."""

    def configuration(self) -> dict:
        """Return this validator's import path and effective settings."""
        return {"_target_": f"{type(self).__module__}.{type(self).__qualname__}", **asdict(self)}

    def report(self, reason: str = "") -> PlacementValidatorReport:
        """Describe this check's settings and pass, failure or disabled outcome."""
        return PlacementValidatorReport(
            self.check,
            self.stage,
            self.configuration(),
            not bool(reason) if self.enabled else None,
            reason if self.enabled else "disabled by configuration",
        )


@dataclass
class RestValidator(PostPhysicsPlacementValidator):
    """Require consecutive quiet pose samples before accepting a layout."""

    check: ClassVar[str] = "rest"
    move_thresh_m: float = 0.002
    """Maximum translation between samples, in metres."""
    turn_thresh_deg: float = 2.0
    """Maximum rotation between samples, in degrees."""
    required_quiet_windows: int = 2
    """Number of consecutive quiet intervals required after a baseline sample."""

    def __post_init__(self) -> None:
        assert self.required_quiet_windows >= 1, "required_quiet_windows must be positive"
        assert math.isfinite(self.move_thresh_m) and self.move_thresh_m >= 0, "move_thresh_m must be non-negative"
        assert math.isfinite(self.turn_thresh_deg) and self.turn_thresh_deg >= 0, "turn_thresh_deg must be non-negative"

    def validate(self, data: PostPhysicsState) -> PlacementValidatorReport:
        assert data.rest_tracker is not None, "Rest validation requires sampled pose history"
        return self.report(data.rest_tracker.failure_reason(list(data.poses)) or "")


@dataclass
class SupportContainmentValidator(PostPhysicsPlacementValidator):
    """Require clutter to remain inside its full support footprint and above its surface."""

    check: ClassVar[str] = "support_containment"
    containment_margin_m: float = 0.0
    """Permitted overhang beyond the support, in metres."""
    fall_through_tolerance_m: float = 0.01
    """Permitted penetration below the support, in metres."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.containment_margin_m) and self.containment_margin_m >= 0, "Invalid containment margin"
        assert (
            math.isfinite(self.fall_through_tolerance_m) and self.fall_through_tolerance_m >= 0
        ), "Invalid penetration tolerance"

    def validate(self, data: PostPhysicsState) -> PlacementValidatorReport:
        import torch

        from isaaclab_arena.offline_placement.geometry import region_above_support
        from isaaclab_arena.offline_placement.validation import check_resting_poses
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.pose import Pose

        reasons = []
        for group in data.groups:
            value = data.snapshot.poses[group.support][data.env_id].tolist()
            support = Pose(tuple(value[:3]), tuple(value[3:]))
            region = region_above_support(
                support.position_xyz, _box_for_env(data.boxes[group.support], data.env_id), support.rotation_xyzw
            )
            bounds = []
            for key in group.objects:
                pose = data.poses[key]
                bounds.append(
                    _box_for_env(data.boxes[key], data.env_id)
                    .rotated_by_quat(pose.rotation_xyzw)
                    .translated(pose.position_xyz)
                )
            verdict = check_resting_poses(
                AxisAlignedBoundingBox(
                    torch.cat([b.min_point for b in bounds]), torch.cat([b.max_point for b in bounds])
                ),
                region,
                containment_margin_m=self.containment_margin_m,
                fall_through_tolerance_m=self.fall_through_tolerance_m,
            )
            if not verdict.ok:
                reasons.append(f"support {group.support}: {verdict.describe(list(group.objects))}")
        return self.report("; ".join(reasons))


@dataclass
class PassiveDriftValidator(PostPhysicsPlacementValidator):
    """Keep passive bodies and robot links close to their initial poses."""

    check: ClassVar[str] = "passive_drift"
    passive_move_thresh_m: float = 0.002
    """Maximum displacement from the initial pose, in metres."""
    passive_turn_thresh_deg: float = 2.0
    """Maximum rotation from the initial pose, in degrees."""

    def __post_init__(self) -> None:
        assert (
            math.isfinite(self.passive_move_thresh_m) and self.passive_move_thresh_m >= 0
        ), "Invalid passive translation limit"
        assert (
            math.isfinite(self.passive_turn_thresh_deg) and self.passive_turn_thresh_deg >= 0
        ), "Invalid passive rotation limit"

    def validate(self, data: PostPhysicsState) -> PlacementValidatorReport:
        reasons = []
        for key in data.passive_keys:
            initial = data.snapshot.poses[key][data.env_id]
            current = data.env.arena_world.get_pose_e(key)[data.env_id]
            reason = _pose_drift_reason(initial, current, self)
            if reason:
                reasons.append(f"{key}: {reason}")
        for key, poses in data.snapshot.links.items():
            current = data.env.scene.articulations[key].data.body_link_pose_w.torch[data.env_id]
            reason = _pose_drift_reason(poses[data.env_id], current, self)
            if reason:
                reasons.append(f"{key} links: {reason}")
        return self.report("; ".join(reasons))


def default_post_physics_validators() -> dict[str, dict]:
    """Default rest, containment and passive-drift checks with their effective settings."""
    return {
        validator.check: validator.configuration()
        for validator in (RestValidator(), SupportContainmentValidator(), PassiveDriftValidator())
    }


def build_post_physics_validators(configurations: dict[str, dict]) -> list[PostPhysicsPlacementValidator]:
    """Construct configured checks, failing explicitly if a requested implementation cannot load."""
    from hydra.utils import instantiate

    validators = []
    for name, configuration in configurations.items():
        validator = instantiate(configuration)
        assert isinstance(validator, PostPhysicsPlacementValidator), f"'{name}' must be a PostPhysicsPlacementValidator"
        assert validator.check == name, f"Check name '{name}' does not match implementation '{validator.check}'"
        status = "enabled; required to pass" if validator.enabled else "skipped; disabled by configuration"
        print(f"[post_physics] {name}: {status}; settings={asdict(validator)}")
        validators.append(validator)
    assert any(validator.enabled for validator in validators), "Enable at least one post-physics validator"
    return validators


def _box_for_env(box: AxisAlignedBoundingBox, env_id: int) -> AxisAlignedBoundingBox:
    """Return one environment's local geometry bounds on the CPU."""
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    return AxisAlignedBoundingBox(box.min_point[env_id : env_id + 1].cpu(), box.max_point[env_id : env_id + 1].cpu())


def _pose_drift_reason(initial: torch.Tensor, current: torch.Tensor, params: PassiveDriftValidator) -> str | None:
    """Report excessive passive drift for poses shaped (..., 7), ordered xyz/xyzw."""
    import torch

    from isaaclab.utils.math import quat_error_magnitude

    if not (torch.isfinite(initial).all() and torch.isfinite(current).all()):
        return "non-finite passive pose"
    distance = float((current[..., :3] - initial[..., :3]).norm(dim=-1).max())
    angle = float(torch.rad2deg(quat_error_magnitude(current[..., 3:], initial[..., 3:])).max())
    if distance > params.passive_move_thresh_m or angle > params.passive_turn_thresh_deg:
        return (
            f"passive drift {distance:.6f} m, {angle:.3f} deg; limits "
            f"{params.passive_move_thresh_m:.6f} m, {params.passive_turn_thresh_deg:.3f} deg"
        )
    return None
