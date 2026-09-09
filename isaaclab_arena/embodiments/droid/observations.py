# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_DROID_NEWTON_GRIPPER_CLOSE_RAD = 0.461
"""Finger_joint close target for Newton's explicit 6-DOF gripper actuation (no USD mimic)."""


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    return wp.to_torch(robot.data.joint_pos)[:, joint_indices]


def _normalized_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, close_position: float) -> torch.Tensor:
    """Return the finger joint position normalized by its closed position."""
    robot = env.scene[asset_cfg.name]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name == "finger_joint"]
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, joint_indices]
    return joint_pos / close_position


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns gripper position as 0 for open and 1 for closed."""
    return _normalized_gripper_pos(env, asset_cfg, torch.pi / 4)


def newton_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns Newton DROID gripper position as 0 for open and 1 for closed."""
    return _normalized_gripper_pos(env, asset_cfg, _DROID_NEWTON_GRIPPER_CLOSE_RAD)


def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the end effector position (x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("base_link")  # Robotiq gripper base link
    return wp.to_torch(robot.data.body_pos_w)[:, body_idx, :]


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the end effector orientation as quaternion (w, x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("base_link")  # Robotiq gripper base link
    return wp.to_torch(robot.data.body_quat_w)[:, body_idx, :]
