# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID-shaped observations with fail-closed FR3 name selection."""

from __future__ import annotations

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .config import ARM_JOINT_NAMES, END_EFFECTOR_BODY_NAME, GRIPPER_CLOSED_ANGLE, GRIPPER_JOINT_NAME


def _robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _index(names: list[str], expected: str, kind: str) -> int:
    try:
        return names.index(expected)
    except ValueError as error:
        raise ValueError(f"expected {kind} {expected!r} was not found") from error


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return FR3 arm positions in the declared action order."""

    robot = _robot(env, asset_cfg)
    joint_indices = [_index(robot.data.joint_names, name, "joint") for name in ARM_JOINT_NAMES]
    return wp.to_torch(robot.data.joint_pos)[:, joint_indices]


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the driver angle normalized to DROID's zero-open, one-closed convention."""

    robot = _robot(env, asset_cfg)
    joint_index = _index(robot.data.joint_names, GRIPPER_JOINT_NAME, "joint")
    position = wp.to_torch(robot.data.joint_pos)[:, joint_index : joint_index + 1]
    return torch.clamp(position / GRIPPER_CLOSED_ANGLE, min=0.0, max=1.0)


def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the world-frame position of the Robotiq base."""

    robot = _robot(env, asset_cfg)
    body_index = _index(robot.data.body_names, END_EFFECTOR_BODY_NAME, "body")
    return wp.to_torch(robot.data.body_pos_w)[:, body_index, :]


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the world-frame quaternion of the Robotiq base (w, x, y, z)."""

    robot = _robot(env, asset_cfg)
    body_index = _index(robot.data.body_names, END_EFFECTOR_BODY_NAME, "body")
    return wp.to_torch(robot.data.body_quat_w)[:, body_index, :]
