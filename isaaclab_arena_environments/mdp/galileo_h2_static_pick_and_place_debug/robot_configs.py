# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot-specific configuration constants for the H2 static apple debug task."""

from __future__ import annotations

# Mild open-arm posture using shoulder joints only. This is just for H2 debug
# visualization in the shelf scene; the final H2 controller should own its reset pose.
H2_STATIC_DEBUG_OPEN_ARM_JOINT_POS: dict[str, float] = {
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.35,
    "right_shoulder_yaw_joint": -0.35,
}
