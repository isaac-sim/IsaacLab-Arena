# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmReach:
    """Reach of one arm, declared by an embodiment and centred on a joint of its USD articulation."""

    center_joint_name: str
    """Name of the USD joint whose origin is the reach centre, typically the shoulder."""

    reach_m: float
    """Reach in metres from the centre joint, typically the manufacturer's published maximum reach."""

    name: str = "arm"
    """Arm identifier, unique within an embodiment."""

    def __post_init__(self) -> None:
        assert self.center_joint_name, "center_joint_name must be non-empty"
        assert math.isfinite(self.reach_m) and self.reach_m > 0.0, "reach_m must be finite and positive"


@dataclass(frozen=True)
class ReachSphere:
    """Sphere bounding the points one arm can reach, in the robot root frame R."""

    center_xyz_R: tuple[float, float, float]
    """Sphere centre in the robot root frame R, in metres."""

    reach_m: float
    """Sphere radius in metres."""

    name: str = "arm"
    """Arm identifier, unique within an embodiment."""

    def __post_init__(self) -> None:
        assert len(self.center_xyz_R) == 3, f"center_xyz_R must have 3 values, got {self.center_xyz_R!r}"
        assert all(math.isfinite(value) for value in self.center_xyz_R), "center_xyz_R must be finite"
        assert math.isfinite(self.reach_m) and self.reach_m > 0.0, "reach_m must be finite and positive"
