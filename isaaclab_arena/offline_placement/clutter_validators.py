# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter-specific checks using the shared post-physics validator interface."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, ClassVar

from isaaclab_arena.offline_placement.clutter_geometry import assert_flat_support_surface, fixed_poses_match
from isaaclab_arena.offline_placement.post_physics_validation import (
    PostPhysicsPlacementValidator,
    default_post_physics_validators,
)
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.offline_placement.settled_batch import SettledBatch
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.validation.types import PlacementValidatorReport


@dataclass
class SupportContainmentValidator(PostPhysicsPlacementValidator):
    """Require clutter to stay within its support footprint and above a minimum resting height."""

    check: ClassVar[str] = "support_containment"
    containment_margin_m: float = 0.0
    """Permitted overhang beyond the support, in metres."""
    fall_through_tolerance_m: float = 0.01
    """Permitted penetration below the minimum resting height, in metres."""
    minimum_resting_heights_m: dict[str, float] = field(default_factory=dict)
    """Minimum object-bottom Z by support scene key, in scaled support-local metres.

    Unspecified supports use their verified flat top. Explicit heights permit
    container interiors below the rim; they do not certify mesh containment.
    """

    def __post_init__(self) -> None:
        assert math.isfinite(self.containment_margin_m) and self.containment_margin_m >= 0, "Invalid containment margin"
        assert (
            math.isfinite(self.fall_through_tolerance_m) and self.fall_through_tolerance_m >= 0
        ), "Invalid penetration tolerance"
        for key, height in self.minimum_resting_heights_m.items():
            assert isinstance(key, str) and key, "Minimum resting heights require support scene keys"
            assert (
                isinstance(height, Real) and not isinstance(height, bool) and math.isfinite(height)
            ), f"Support {key!r}: minimum resting height must be a finite number"

    def validate_scene(self, env: ManagerBasedEnv, assets: Sequence[PlaceableAsset]) -> None:
        """Verify support keys and heights, retaining flat-top checks for unspecified supports."""
        support_keys = set()
        for asset in assets:
            relation = get_relation(asset, ClutterOn)
            if relation is not None:
                support_keys.add(relation.parent.get_scene_key())
        unknown = self.minimum_resting_heights_m.keys() - support_keys
        assert not unknown, f"Minimum resting heights reference unknown ClutterOn supports: {sorted(unknown)}"
        for key in sorted(support_keys):
            height = self.minimum_resting_heights_m.get(key)
            if height is None:
                assert_flat_support_surface(env.scene, key)
                continue
            bounds = env.arena_world.get_aabb_in_local_frame(key)
            assert math.isfinite(height) and bool(
                ((bounds.min_point[:, 2] <= height) & (height <= bounds.max_point[:, 2])).all()
            ), f"Support {key!r}: minimum resting height must be finite and within its local Z bounds"

    def get_geometry_keys(self, assets: Sequence[PlaceableAsset]) -> set[str]:
        keys = set()
        for asset in assets:
            relation = get_relation(asset, ClutterOn)
            if relation is not None:
                keys.update((asset.get_scene_key(), relation.parent.get_scene_key()))
        return keys

    def validate(self, data: SettledBatch) -> list[PlacementValidatorReport]:
        reports = []
        for env_id in data.env_ids:
            layout = data.source_layouts[env_id]
            reasons = []
            for check in (PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION):
                if layout.validation_results.validation_results.get(check) is not True:
                    reasons.append(f"release must pass {check}")
            for asset in layout.positions:
                relation = get_relation(asset, ClutterOn)
                if relation is None:
                    continue
                child_key, support_key = asset.get_scene_key(), relation.parent.get_scene_key()
                child, support = data.geometry[child_key], data.geometry[support_key]
                expected = relation.parent.get_initial_pose().to_tensor(support.initial_poses.device)
                if not all(
                    fixed_poses_match(expected, poses[env_id]) for poses in (support.initial_poses, support.final_poses)
                ):
                    reasons.append(f"support {support_key!r} differs from its configured pose")
                    continue
                support_pose = support.final_poses[env_id]
                quarters = quaternion_to_90_deg_z_quarters(tuple(support_pose[3:].tolist()))
                support_bounds_e = support.bounds[env_id].rotated_90_around_z(quarters).translated(support_pose[:3])
                minimum_height_e = self.minimum_resting_heights_m.get(support_key)
                if minimum_height_e is not None:
                    # Keep configured heights in the same tensor precision as preflight and captured bounds.
                    minimum_height_e = support_pose[2] + minimum_height_e
                if not torch.isfinite(child.final_poses[env_id]).all():
                    reasons.append(f"{child_key}: non-finite pose")
                    continue
                pose = child.final_poses[env_id].tolist()
                bounds = (
                    AxisAlignedBoundingBox(
                        child.bounds.min_point[env_id : env_id + 1], child.bounds.max_point[env_id : env_id + 1]
                    )
                    .rotated_by_quat(tuple(pose[3:]))
                    .translated(child.final_poses[env_id, :3])
                )
                verdict = check_resting_poses(
                    bounds,
                    support_bounds_e,
                    self.containment_margin_m,
                    self.fall_through_tolerance_m,
                    minimum_resting_height_m=minimum_height_e,
                )
                if not verdict.ok:
                    reasons.append(f"support {support_key}: {verdict.describe([child_key])}")
            reports.append(self.report(not reasons, "; ".join(reasons)))
        return reports


def default_clutter_validators() -> dict[str, dict]:
    """Shared velocity/link checks, non-clutter root limits and support containment."""
    validators = default_post_physics_validators()
    containment = SupportContainmentValidator()
    validators[containment.check] = containment.configuration()
    return validators


@dataclass
class ClutterContainmentResult:
    """Indices of clutter members that failed containment checks."""

    diverged: list[int] = field(default_factory=list)
    """Indices of non-finite poses."""

    fell_through: list[int] = field(default_factory=list)
    """Indices below the minimum resting height."""

    fell_off: list[int] = field(default_factory=list)
    """Indices outside the support footprint."""

    @property
    def ok(self) -> bool:
        """Whether every member satisfies the containment checks."""
        return not (self.diverged or self.fell_through or self.fell_off)

    def describe(self, names: list[str]) -> str:
        """Return a human-readable summary naming the offending members."""
        parts = []
        for label, indices in (
            ("diverged", self.diverged),
            ("fell through", self.fell_through),
            ("fell off", self.fell_off),
        ):
            if indices:
                offenders = ", ".join(names[index] for index in indices)
                parts.append(f"{label}: {offenders}")
        return "; ".join(parts) if parts else "all members within support"


def check_resting_poses(
    bounds: AxisAlignedBoundingBox,
    support_bounds: AxisAlignedBoundingBox,
    containment_margin_m: float,
    fall_through_tolerance_m: float,
    *,
    minimum_resting_height_m: float | torch.Tensor | None = None,
) -> ClutterContainmentResult:
    """Return containment failures for N members.

    Args:
        bounds: Rotated object bounds in the environment frame, min/max shape (N, 3).
        support_bounds: Full support bounds in the environment frame, min/max shape (1, 3).
        containment_margin_m: Permitted overhang beyond the support footprint.
        fall_through_tolerance_m: Permitted penetration below the minimum resting height.
        minimum_resting_height_m: Minimum object-bottom Z in the environment frame, as a
            scalar in metres. None uses the support bounds' top surface.
    """
    verdict = ClutterContainmentResult()
    margin = containment_margin_m
    support_lower, support_upper = support_bounds.min_point[0], support_bounds.max_point[0]
    height = support_upper[2] if minimum_resting_height_m is None else minimum_resting_height_m
    floor = height - fall_through_tolerance_m
    for index, (lower, upper) in enumerate(zip(bounds.min_point, bounds.max_point, strict=True)):
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            verdict.diverged.append(index)
            continue
        if lower[2] < floor:
            verdict.fell_through.append(index)
        if not (
            lower[0] >= support_lower[0] - margin
            and upper[0] <= support_upper[0] + margin
            and lower[1] >= support_lower[1] - margin
            and upper[1] <= support_upper[1] + margin
        ):
            verdict.fell_off.append(index)
    return verdict
