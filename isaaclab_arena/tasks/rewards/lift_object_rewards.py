# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for lifting an object to a goal pose."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def object_is_lifted(
    env: IsaacLabArenaManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height.

    Args:
            env: The environment.
            minimal_height: World height in meters above which the object counts as lifted.
            object_cfg: Scene entity of the object.

        Returns:
            Reward with shape (num_envs,): 1 where the object is lifted, otherwise 0.
    """
    object_height_w = env.arena_world.get_position_w(object_cfg.name)[:, 2]
    return torch.where(object_height_w > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: IsaacLabArenaManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.

    Args:
            env: The environment.
            std: Length scale in meters of the tanh kernel.
            minimal_height: World height in meters above which the object must be for a non-zero reward.
            command_name: Name of the command term whose goal position is expressed in the robot base frame.
            robot_cfg: Scene entity of the robot, whose base frame the command is expressed in.
            object_cfg: Scene entity of the object.

        Returns:
            Reward with shape (num_envs,), which is 1 at zero goal distance and decays towards 0 with distance.
    """
    arena_world = env.arena_world
    T_W_B = arena_world.get_pose_w(robot_cfg.name)
    object_position_w = arena_world.get_position_w(object_cfg.name)

    command = env.command_manager.get_command(command_name)
    desired_position_b = command[:, :3]
    desired_position_w, _ = combine_frame_transforms(
        T_W_B[:, :3],
        T_W_B[:, 3:],
        desired_position_b,
    )
    distance_to_goal = torch.norm(desired_position_w - object_position_w, dim=1)
    object_is_above_minimal_height = object_position_w[:, 2] > minimal_height
    return object_is_above_minimal_height * (1 - torch.tanh(distance_to_goal / std))
