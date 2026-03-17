# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


@dataclass
class PlacementResult:
    """Result of an ObjectPlacer.place() call (num_envs=1). Same as main: no event_cfg; placer applies to objects."""

    success: bool
    """Whether placement passed validation checks."""

    positions: dict[Object, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""


@dataclass
class MultiEnvPlacementResult:
    """Result of multi-env placement: one PlacementResult per environment.

    Returned by ObjectPlacer.place(..., num_envs>1). Use .results[env_id] for that
    env's layout. .event_cfg is merged into env events so layouts are applied at reset.
    """

    results: list[PlacementResult]
    """One PlacementResult per environment (same length as num_envs)."""

    event_cfg: Any
    """Placement event config to merge into env events so layouts are applied at reset."""

    @property
    def success(self) -> bool:
        """True if every environment's placement succeeded."""
        return all(r.success for r in self.results)

    @property
    def attempts(self) -> int:
        """Number of attempts (same for all envs in the batched run)."""
        return self.results[0].attempts if self.results else 0
