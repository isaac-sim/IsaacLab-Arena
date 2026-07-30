# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gripper joint helpers for Gear Assembly reset-time grasp initialization."""

from __future__ import annotations

import torch


def set_finger_joint_pos_grav(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
) -> None:
    """Set Flexiv Grav gripper finger joints."""
    assert len(finger_joints) >= 6, f"Grav gripper requires at least 6 finger joints, got {len(finger_joints)}"
    for idx in reset_ind_joint_pos:
        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position
        joint_pos[idx, finger_joints[2]] = finger_joint_position
        joint_pos[idx, finger_joints[3]] = finger_joint_position
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position


def set_finger_joint_pos_robotiq_2f140(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
) -> None:
    """Set Robotiq 2F-140 finger joints."""
    assert len(finger_joints) >= 8, f"2F-140 gripper requires at least 8 finger joints, got {len(finger_joints)}"
    for idx in reset_ind_joint_pos:
        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position
        joint_pos[idx, finger_joints[2]] = 0.0
        joint_pos[idx, finger_joints[3]] = 0.0
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position
        joint_pos[idx, finger_joints[6]] = finger_joint_position
        joint_pos[idx, finger_joints[7]] = finger_joint_position


def set_finger_joint_pos_robotiq_2f85(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
) -> None:
    """Set Robotiq 2F-85 finger joints."""
    assert len(finger_joints) >= 6, f"2F-85 gripper requires at least 6 finger joints, got {len(finger_joints)}"
    for idx in reset_ind_joint_pos:
        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position
        joint_pos[idx, finger_joints[2]] = -finger_joint_position
        joint_pos[idx, finger_joints[3]] = finger_joint_position
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position
