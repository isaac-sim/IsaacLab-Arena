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
    """Reach in metres from the centre joint."""

    def __post_init__(self) -> None:
        assert self.center_joint_name, "center_joint_name must be non-empty"
        assert math.isfinite(self.reach_m) and self.reach_m > 0.0, "reach_m must be finite and positive"
