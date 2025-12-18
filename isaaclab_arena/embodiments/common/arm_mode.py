# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ArmMode(str, Enum):
    """
    The arm mode for the embodiment.

    Attributes:
        SINGLE_ARM: Single arm mode (the robot has only one arm).
        DUAL_ARM: Dual arm mode (bimanual robot, task is performed with both arms in the demonstration).
        LEFT: Left arm mode (bimanual robot, task is performed with the left arm in the demonstration, right arm is idle).
        RIGHT: Right arm mode (bimanual robot, task is performed with the right arm in the demonstration, left arm is idle).
    """

    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    LEFT = "left"
    RIGHT = "right"

    def get_other_arm(self) -> str:
        assert self in [ArmMode.LEFT, ArmMode.RIGHT], f"Arm mode {self} is not a bimanual arm mode"
        return ArmMode.RIGHT if self == ArmMode.LEFT else ArmMode.LEFT
