# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    joint_pos = robot.data.joint_pos[0, joint_indices]
    return joint_pos


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (torch.pi / 4)

    return joint_pos
