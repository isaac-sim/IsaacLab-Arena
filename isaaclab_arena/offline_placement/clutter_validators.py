# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter-specific checks using the shared post-physics validator interface."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from isaaclab_arena.offline_placement.clutter_geometry import check_resting_poses, get_placement_region
from isaaclab_arena.offline_placement.post_physics_validation import (
    PostPhysicsPlacementValidator,
    default_post_physics_validators,
)
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.physics_settle import get_pose_drift

if TYPE_CHECKING:
    from isaaclab_arena.offline_placement.settled_batch import SettledBatch
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.validation.types import PlacementValidatorReport


@dataclass
class SupportContainmentValidator(PostPhysicsPlacementValidator):
    """Require clutter to remain within its fixed support footprint and above its surface."""

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
            for check in ("no_overlap", "clutter_on_relation"):
                if layout.validation_results.validation_results.get(check) is not True:
                    reasons.append(f"release must pass {check}")
            for asset in layout.positions:
                relation = get_relation(asset, ClutterOn)
                if relation is None:
                    continue
                child_key, support_key = asset.get_scene_key(), relation.parent.get_scene_key()
                child, support = data.geometry[child_key], data.geometry[support_key]
                expected = relation.parent.get_initial_pose().to_tensor(support.initial_poses.device)
                initial_drift = get_pose_drift(expected, support.initial_poses[env_id])
                final_drift = get_pose_drift(expected, support.final_poses[env_id])
                # Fixed geometry may differ only by float32 transform roundoff.
                if any(drift is None or drift[0] > 1e-5 or drift[1] > 1e-3 for drift in (initial_drift, final_drift)):
                    reasons.append(f"support {support_key!r} differs from its configured pose")
                    continue
                support_pose = support.final_poses[env_id].tolist()
                support_bounds = AxisAlignedBoundingBox(
                    support.bounds.min_point[env_id : env_id + 1], support.bounds.max_point[env_id : env_id + 1]
                )
                region = get_placement_region(tuple(support_pose[:3]), support_bounds, tuple(support_pose[3:]))
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
                verdict = check_resting_poses(bounds, region, self.containment_margin_m, self.fall_through_tolerance_m)
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
