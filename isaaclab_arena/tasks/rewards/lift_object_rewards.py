# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_arena.scene.object_geometry import object_geometry
from isaaclab_arena.scene.object_state import object_state


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object_pos_w = object_state(env, object_cfg.name).position_w()
    return torch.where(object_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    object_pos_w = object_state(env, object_cfg.name).position_w()
    distance = torch.norm(des_pos_w - object_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_nearest_point_distance_to_frame(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the object's nearest representative point from a frame target."""
    frame = env.scene[frame_cfg.name]
    frame_pos_w = wp.to_torch(frame.data.target_pos_w)[..., 0, :]
    nearest_point_w = object_geometry(env, object_cfg.name).nearest_point_w(frame_pos_w)
    distance = torch.linalg.vector_norm(nearest_point_w - frame_pos_w, dim=-1)
    return 1.0 - torch.tanh(distance / std)
