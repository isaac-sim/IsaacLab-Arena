# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka-specific grasp parameters for NIST gear insertion."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab_arena.tasks.nist_gear_insertion.events import GraspCfg


def franka_gripper_joint_setter(
    joint_pos: torch.Tensor,
    row_indices: Sequence[int],
    finger_joint_indices: Sequence[int],
    width: float,
) -> None:
    """Set Franka Panda finger joints from a total gripper opening width."""
    for jid in finger_joint_indices:
        joint_pos[row_indices, jid] = width / 2.0


def get_franka_nist_gear_insertion_grasp_config() -> GraspCfg:
    """Return reset grasp parameters for the Franka NIST gear insertion policy.

    The reset first opens the fingers to 3 cm while inverse kinematics moves
    ``panda_hand`` to the gear, then closes the fingers to 0 m to hold the gear.
    In Isaac Lab 3.0 xyzw convention, ``[1, 0, 0, 0]`` is an intentional
    180-degree rotation about X. The ``[0.02, 0.0, -0.128]`` translation is the
    held-gear-root to ``panda_hand`` offset that centers the gear between the
    fingertips.
    """
    return GraspCfg(
        hand_grasp_width=0.03,
        hand_close_width=0.0,
        gripper_joint_setter_func=franka_gripper_joint_setter,
        end_effector_body_name="panda_hand",
        finger_joint_names="panda_finger_joint[1-2]",
        grasp_rot_offset=[1.0, 0.0, 0.0, 0.0],
        grasp_offset=[0.02, 0.0, -0.128],
        arm_joint_names="panda_joint[1-7]",
    )
