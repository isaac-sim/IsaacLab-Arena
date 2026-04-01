# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass
class PlacementResult:
    """Result of an ObjectPlacer.place() call."""

    success: bool
    """Whether placement passed validation checks."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""


@dataclass
class MultiEnvPlacementResult:
    """Result of an ObjectPlacer.place() call for multiple environments."""

    results: list[PlacementResult]
    """One PlacementResult per environment (same length as num_envs)."""

    @property
    def success(self) -> bool:
        """True if every environment's placement succeeded."""
        return all(r.success for r in self.results)

    @property
    def attempts(self) -> int:
        """Number of attempts (same for all envs in the batched run)."""
        return self.results[0].attempts if self.results else 0
