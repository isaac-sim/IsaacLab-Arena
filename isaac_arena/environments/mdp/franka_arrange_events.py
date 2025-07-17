# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def permute_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    pose_selection_list: list[tuple] = [],
):
    """Permute poses between objects.

    Args:
        env_ids: List of environment ids to permute poses for.
        asset_cfgs: List of asset configurations to permute poses for.
        pose_selection_list: List of poses to permute between objects.
                             Each pose is a tuple of (x, y, z, roll, pitch, yaw).
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        assert len(asset_cfgs) == len(
            pose_selection_list
        ), "Number of asset cfgs and pose selection list must be the same"
        random.shuffle(pose_selection_list)

        num_poses = len(pose_selection_list)
        for i in range(num_poses):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor(pose_selection_list[i], device=env.device)
            position = pose_tensor[0:3] + env.scene.env_origins[cur_env, 0:3]
            orientation = math_utils.quat_from_euler_xyz(
                pose_tensor[3], pose_tensor[4], pose_tensor[5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([position, orientation], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )


def permute_object_poses_relative_to_parent(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    parent_asset_cfg: SceneEntityCfg,
    asset_cfgs: list[SceneEntityCfg],
    relative_object_poses: list[tuple],
):
    """Permute poses between objects relative to a parent object's position.

    Args:
        env: The environment instance.
        env_ids: List of environment ids to permute poses for.
        parent_asset_cfg: Asset configuration for the parent object.
        asset_cfgs: List of asset configurations to permute poses for.
        relative_object_poses: List of poses relative to parent object (orientation is absolute).
                             Each pose is a tuple of (x, y, z, roll, pitch, yaw).
    """
    assert len(asset_cfgs) <= len(
        relative_object_poses
    ), "Number of asset cfgs must be less than or equal to the number of relative object poses"

    if env_ids is None:
        return
    parent = env.scene[parent_asset_cfg.name]

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        random.shuffle(relative_object_poses)
        for object_idx in range(len(asset_cfgs)):
            asset = env.scene[asset_cfgs[object_idx].name]

            # Write pose to simulation
            relative_pose_tensor = torch.tensor(
                relative_object_poses[object_idx], device=env.device
            )
            positions = parent.data.root_pos_w[cur_env, :] + relative_pose_tensor[0:3]
            # Orientation is absolute for all objects.
            orientations = math_utils.quat_from_euler_xyz(
                relative_pose_tensor[3], relative_pose_tensor[4], relative_pose_tensor[5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )