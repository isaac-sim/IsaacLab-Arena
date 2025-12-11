# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence
from enum import Enum

import isaaclab.utils.math as PoseUtils


def get_rigid_and_articulated_object_poses(state: dict, env_ids: Sequence[int] | None = None):
    """
    Gets the pose of each object(rigid and articulated) in the current scene.
    Args:
        state: The state of the scene.
    Returns:
        A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
    """

    def pose_from(obj_state) -> "torch.Tensor":
        rp = obj_state["root_pose"][env_ids]  # (..., 7): [x,y,z, qx,qy,qz,qw]
        pos, quat = rp[..., :3], rp[..., 3:7]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))  # (..., 4, 4)

    groups = ["rigid_object", "articulation"]

    object_pose_matrix = {
        name: pose_from(obj_state) for group in groups for name, obj_state in state.get(group, {}).items()
    }

    return object_pose_matrix


class MimicArmMode(str, Enum):
    """
    The arm mode for the mimic environment configuration.

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
