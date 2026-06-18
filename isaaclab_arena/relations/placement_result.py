# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults


@dataclass
class PlacementResult:
    """Solved object layout for one environment."""

    validation_results: PlacementValidationResults
    """Validation checklist for the placement."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""

    orientations: dict[ObjectBase, float] = field(default_factory=dict)
    """Per-object yaw (radians) about the world up (Z) axis, composed on top of each object's
    base rotation. Keyed by object, like positions. Empty when unrotated."""

    @property
    def success(self) -> bool:
        """True when all required validation checks pass.

        place() returns a best-loss fallback even on failure; check this to tell validated from fallback.
        """
        return self.validation_results.do_all_required_validation_checks_pass()
