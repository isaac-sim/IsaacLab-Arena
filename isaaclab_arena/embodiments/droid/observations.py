# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms


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


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns gripper position as 0 for open and 1 for closed."""
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, joint_indices]
    # rescale to 0–1
    return joint_pos / (torch.pi / 4)


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


def droid_eef_pose_base(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return DROID's panda_link8 pose in panda_link0 coordinates, with a wxyz quaternion."""
    # DROID's Polymetis config uses panda_link8, not the Robotiq base_link or fingertips.
    # The fixed panda_joint8 translates 0.107 m along panda_link7's Z axis.
    # Reconstruct it from link7 because fixed joints may be merged by the USD importer.
    robot = env.scene[asset_cfg.name]
    base_idx = robot.data.body_names.index("panda_link0")
    wrist_idx = robot.data.body_names.index("panda_link7")
    positions = wp.to_torch(robot.data.body_pos_w)
    quaternions = wp.to_torch(robot.data.body_quat_w)
    offset = torch.zeros_like(positions[:, wrist_idx])
    offset[:, 2] = 0.107
    position_w, quaternion_w = combine_frame_transforms(positions[:, wrist_idx], quaternions[:, wrist_idx], offset)
    position_b, quaternion_b = subtract_frame_transforms(
        positions[:, base_idx], quaternions[:, base_idx], position_w, quaternion_w
    )
    # Isaac Lab stores xyzw; the GR00T adapter accepts XYZ/wxyz poses.
    return torch.cat((position_b, quaternion_b[:, [3, 0, 1, 2]]), dim=-1)
