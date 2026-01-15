# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from isaaclab_arena.assets.dummy_object import DummyObject


@dataclass
class PlacementResult:
    """Result of an ObjectPlacer.place() call."""

    success: bool
    """Whether placement succeeded (loss < threshold within max_attempts)."""

    positions: dict[DummyObject, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""
