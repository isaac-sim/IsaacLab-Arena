# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

FRANKA_PANDA_MAX_REACH_M = 0.855
"""Franka Panda / Research 3 datasheet maximum reach, in metres."""

FRANKA_PANDA_SHOULDER_HEIGHT_M = 0.333
"""Height of the Franka shoulder (panda_link2 origin) above panda_link0."""


@dataclass(frozen=True)
class ArmReach:
    """Sphere bounding the points one arm can reach, in the robot root frame."""

    center_xyz_R: tuple[float, float, float]
    """Sphere centre in the robot root frame R, in metres, typically the shoulder."""

    radius_m: float
    """Sphere radius in metres, typically the manufacturer's published maximum reach."""

    name: str = "arm"
    """Arm identifier, unique within an embodiment."""

    def __post_init__(self) -> None:
        assert len(self.center_xyz_R) == 3, f"center_xyz_R must have 3 values, got {self.center_xyz_R!r}"
        assert all(math.isfinite(value) for value in self.center_xyz_R), "center_xyz_R must be finite"
        assert math.isfinite(self.radius_m) and self.radius_m > 0.0, "radius_m must be finite and positive"


FRANKA_PANDA_ARM_REACH = ArmReach(
    center_xyz_R=(0.0, 0.0, FRANKA_PANDA_SHOULDER_HEIGHT_M),
    radius_m=FRANKA_PANDA_MAX_REACH_M,
)
"""Franka reach sphere for embodiments whose root frame is panda_link0."""
