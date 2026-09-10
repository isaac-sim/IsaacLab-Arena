# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_validation import PlacementCheck

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults


@dataclass
class PlacementResult:
    """Asset poses and validation results for one environment. E is its local frame."""

    validation_results: PlacementValidationResults
    """Validation checklist for the placement."""

    positions: dict[PlaceableAsset, tuple[float, float, float]]
    """Object positions in E; each value has shape (3,), in metres."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""

    orientations: dict[PlaceableAsset, float] = field(default_factory=dict)
    """Sparse absolute yaw angles in E; each value is a scalar in radians."""

    rotations: dict[PlaceableAsset, tuple[float, float, float, float]] = field(default_factory=dict)
    """Sparse object-to-E quaternions, each shape (4,) in (x, y, z, w) order.

    These take precedence over orientations.
    """

    @property
    def success(self) -> bool:
        """Whether all required validation checks passed."""
        return self.validation_results.do_all_required_validation_checks_pass()

    @property
    def is_prepared(self) -> bool:
        """Whether captured poses passed settling, containment and final validation."""
        checks = self.validation_results.validation_results
        return self.success and all(
            checks.get(check) is True
            for check in (
                PlacementCheck.CAPTURED_OBJECTS_SETTLED,
                PlacementCheck.CLUTTER_CONTAINED,
                PlacementCheck.FINAL_POSES_VALIDATED,
            )
        )
