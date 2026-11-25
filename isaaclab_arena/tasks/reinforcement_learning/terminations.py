# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def lift_object_rl_success(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rl_training: bool = False,
    command_name: str = "object_pose",
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Supports multiple modes:
    - RL training: Always returns False (no early termination)
    - RL evaluation: Uses goal from command manager

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        robot_cfg: The robot configuration (needed to transform the goal to world frame).
        rl_training: If True, always returns False (disables success termination for RL training).
        command_name: The name of the command that is used to control the object.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    arena_world = env.arena_world
    T_W_B = arena_world.get_pose_w(robot_cfg.name)
    object_position_w = arena_world.get_position_w(object_cfg.name)

    command = env.command_manager.get_command(command_name)
    desired_position_b = command[:, :3]

    # Transform goal from robot-base frame to world frame
    desired_position_w, _ = combine_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], desired_position_b)

    distance = torch.linalg.norm(desired_position_w - object_position_w, dim=1)
    return distance < position_tolerance
