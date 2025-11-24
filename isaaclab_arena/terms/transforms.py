# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as PoseUtils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def transform_pose_from_world_to_target_frame(
    env: ManagerBasedEnv,
    target_link_name: str,
    target_frame_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the pose of the target link in the specified target frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    assert (
        target_link_name in asset.data.body_names
    ), f"Target link {target_link_name} not found in asset {asset_cfg.name}"
    assert (
        target_frame_name in asset.data.body_names
    ), f"Target frame {target_frame_name} not found in asset {asset_cfg.name}"

    target_link_pose_w = asset.data.body_link_state_w[:, asset.data.body_names.index(target_link_name), :]
    target_frame_pose_w = asset.data.body_link_state_w[:, asset.data.body_names.index(target_frame_name), :]

    # Convert to pose matrix
    target_link_position_w = target_link_pose_w[:, :3]
    target_link_rot_mat_w = PoseUtils.matrix_from_quat(target_link_pose_w[:, 3:7])
    target_link_pose_mat_w = PoseUtils.make_pose(target_link_position_w, target_link_rot_mat_w)

    target_frame_position_w = target_frame_pose_w[:, :3]
    target_frame_rot_mat_w = PoseUtils.matrix_from_quat(target_frame_pose_w[:, 3:7])
    target_frame_pose_mat_w = PoseUtils.make_pose(target_frame_position_w, target_frame_rot_mat_w)

    # Get target frame inverse transform to convert from world to target frame
    target_frame_pose_inv = PoseUtils.pose_inv(target_frame_pose_mat_w)

    # Transform target link poses from world frame to target frame
    target_link_pose_target_frame = torch.matmul(target_frame_pose_inv, target_link_pose_mat_w)

    return target_link_pose_target_frame


def get_target_link_position_in_target_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "left_wrist_yaw_link",
    target_frame_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the position of the target link in the target frame."""
    target_link_pose_target_frame = transform_pose_from_world_to_target_frame(
        env, target_link_name, target_frame_name, asset_cfg
    )
    target_link_position_target_frame, left_target_link_rot_target_frame = PoseUtils.unmake_pose(
        target_link_pose_target_frame
    )
    return target_link_position_target_frame


def get_target_link_quaternion_in_target_frame(
    env: ManagerBasedEnv,
    target_link_name: str = "left_wrist_yaw_link",
    target_frame_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the quaternion of the target link in the target frame."""
    target_link_pose_target_frame = transform_pose_from_world_to_target_frame(
        env, target_link_name, target_frame_name, asset_cfg
    )
    target_link_position_target_frame, left_target_link_rot_target_frame = PoseUtils.unmake_pose(
        target_link_pose_target_frame
    )
    target_link_quat_target_frame = PoseUtils.quat_from_matrix(left_target_link_rot_target_frame)
    return target_link_quat_target_frame


def get_navigate_cmd(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Get the navigate command."""
    return env.action_manager.get_term("g1_action").navigate_cmd.clone()


def get_asset_position(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot position."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def get_asset_quaternion(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot quaternion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w
