# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot-specific configuration constants for the G1 static apple task."""

from __future__ import annotations

# Mild open-arm posture using shoulder joints only. Shoulder yaw keeps the forearms in
# the liked orientation; shoulder roll moves the arms away from the torso.
G1_STATIC_OPEN_ARM_JOINT_POS: dict[str, float] = {
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.5,
    "right_shoulder_yaw_joint": -0.5,
}

G1_STATIC_FINGER_FRICTION_MATERIAL_PATH = "/World/Materials/g1_static_pick_place_high_friction_fingers"
G1_STATIC_FINGER_STATIC_FRICTION = 6.0
G1_STATIC_FINGER_DYNAMIC_FRICTION = 5.0
G1_STATIC_FINGER_PRIM_NAME_MARKERS: tuple[str, ...] = (
    "hand",
    "thumb",
    "index",
    "middle",
)
