# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the Franka embodiment."""

import torch

import warp as wp
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the two Franka finger joint positions, with the second sign-flipped to match the first.

    Args:
        env: Environment to read from.
        robot_cfg: Scene entity of the robot articulation.

    Returns:
        Finger positions with shape `(num_envs, 2)`.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = wp.to_torch(robot.data.joint_pos)[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * wp.to_torch(robot.data.joint_pos)[:, -2].clone().unsqueeze(1)
    return torch.cat((finger_joint_1, finger_joint_2), dim=1)
