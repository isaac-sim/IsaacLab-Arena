# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sphere:
    """Sphere with a centre and radius expressed in the same coordinate frame."""

    center_xyz: tuple[float, float, float]
    """Centre coordinates in metres; the containing variable identifies the frame."""

    radius_m: float
    """Radius in metres."""

    def __post_init__(self) -> None:
        assert len(self.center_xyz) == 3, f"center_xyz must have 3 values, got {self.center_xyz!r}"
        assert all(math.isfinite(value) for value in self.center_xyz), "center_xyz must be finite"
        assert math.isfinite(self.radius_m) and self.radius_m > 0.0, "radius_m must be finite and positive"
