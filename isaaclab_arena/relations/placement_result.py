# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults


@dataclass
class PlacementResult:
    """Solved asset layout for one environment."""

    validation_results: PlacementValidationResults
    """Validation checklist for the placement."""

    positions: dict[PlaceableAsset, tuple[float, float, float]]
    """Final ``(x, y, z)`` positions in metres in the environment-local frame."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""

    orientations: dict[PlaceableAsset, float] = field(default_factory=dict)
    """Sparse map of world yaw angles ``theta_z`` in radians; omitted assets retain marker orientation."""

    @property
    def success(self) -> bool:
        """Whether all required validation checks passed."""
        return self.validation_results.do_all_required_validation_checks_pass()
