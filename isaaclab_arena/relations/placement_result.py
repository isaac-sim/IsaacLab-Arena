# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


@dataclass
class PlacementResult:
    """Result of an ObjectPlacer.place() call."""

    success: bool
    """Whether placement passed validation checks."""

    positions: dict[Object, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""
