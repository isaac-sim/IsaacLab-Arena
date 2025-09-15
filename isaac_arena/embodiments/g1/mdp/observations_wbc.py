# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as PoseUtils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv,ManagerBasedRLEnv
    from isaaclab.assets import Articulation


def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint accelerations of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their accelerations returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]

def eef_pose_pelvis_frame(env: ManagerBasedEnv, eef_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    body_state_w = asset.data.body_state_w

    eef_pose_w = asset.data.body_link_state_w[:, asset.data.body_names.index(eef_name), :]
    pelvis_pose_w = asset.data.body_link_state_w[:, asset.data.body_names.index("pelvis"), :]

    # Convert to pose matrix
    eef_position_w = eef_pose_w[:, :3]
    eef_rot_mat_w = PoseUtils.matrix_from_quat(eef_pose_w[:, 3:7])
    eef_pose_mat_w = PoseUtils.make_pose(eef_position_w, eef_rot_mat_w)

    pelvis_position_w = pelvis_pose_w[:, :3]
    pelvis_rot_mat_w = PoseUtils.matrix_from_quat(pelvis_pose_w[:, 3:7])
    pelvis_pose_mat_w = PoseUtils.make_pose(pelvis_position_w, pelvis_rot_mat_w)

    # Get pelvis inverse transform to convert from world to pelvis frame
    pelvis_pose_inv = PoseUtils.pose_inv(pelvis_pose_mat_w)

    # Transform wrist poses from world frame to pelvis frame
    eef_pose_pelvis_frame = torch.matmul(pelvis_pose_inv, eef_pose_mat_w)

    return eef_pose_pelvis_frame


def get_eef_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_name: str = "left_wrist_yaw_link",
    mode: str = "pos",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    eef_pose_w = asset.data.body_link_state_w[:, asset.data.body_names.index(eef_name), :]
    pelvis_pose_w = asset.data.body_link_state_w[:, asset.data.body_names.index("pelvis"), :]

    # Get eef pose matrix in world frame
    eef_position_w = eef_pose_w[:, :3] - env.scene.env_origins
    eef_rot_mat_w = PoseUtils.matrix_from_quat(eef_pose_w[:, 3:7])
    eef_pose_mat_w = PoseUtils.make_pose(eef_position_w, eef_rot_mat_w)

    pelvis_position_w = pelvis_pose_w[:, :3] - env.scene.env_origins
    pelvis_rot_mat_w = PoseUtils.matrix_from_quat(pelvis_pose_w[:, 3:7])
    pelvis_pose_mat_w = PoseUtils.make_pose(pelvis_position_w, pelvis_rot_mat_w)

    # Get pelvis inverse transform to convert from world to pelvis frame
    pelvis_pose_inv = PoseUtils.pose_inv(pelvis_pose_mat_w)

    # Transform wrist poses from world frame to pelvis frame
    eef_pose_pelvis_frame = torch.matmul(pelvis_pose_inv, eef_pose_mat_w)

    # Convert from pose matrix to position and quaternion
    eef_pos_pelvis_frame, left_eef_rot_pelvis_frame = PoseUtils.unmake_pose(eef_pose_pelvis_frame)
    eef_quat_pelvis_frame = PoseUtils.quat_from_matrix(left_eef_rot_pelvis_frame)

    if mode == "pos":
        return eef_pos_pelvis_frame
    elif mode == "quat":
        return eef_quat_pelvis_frame
    else:
        raise ValueError(f"Invalid mode: {mode}")

def get_navigate_cmd(
    env: ManagerBasedRLEnv,
):
    return env.action_manager.get_term("g1_action").navigate_cmd.clone()

def get_robot_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w

def get_robot_quat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w