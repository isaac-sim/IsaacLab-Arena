# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import torch
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_END_IDX,
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_END_IDX,
    TORSO_ORIENTATION_RPY_CMD_START_IDX,
)


class ActionComponentMode(Enum):
    """Enum for different action component modes."""

    LEFT_EEF_POS = "left_eef_pos"
    LEFT_EEF_QUAT = "left_eef_quat"
    RIGHT_EEF_POS = "right_eef_pos"
    RIGHT_EEF_QUAT = "right_eef_quat"
    NAVIGATE_CMD = "navigate_cmd"
    BASE_HEIGHT_CMD = "base_height_cmd"
    TORSO_ORIENTATION_RPY_CMD = "torso_orientation_rpy_cmd"


def get_navigate_cmd(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Get the P-controller navigate command."""
    return env.action_manager.get_term("g1_action").navigate_cmd.clone()


def extract_action_components(
    env: ManagerBasedEnv,
    mode: ActionComponentMode,
) -> torch.Tensor:
    """Extract the individual components of the G1 WBC PINK action."""
    current_action = env.action_manager.action.clone()

    if mode == ActionComponentMode.LEFT_EEF_POS:
        left_wrist_pos = current_action[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX]
        return left_wrist_pos
    elif mode == ActionComponentMode.LEFT_EEF_QUAT:
        left_wrist_quat = current_action[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX]
        return left_wrist_quat
    elif mode == ActionComponentMode.RIGHT_EEF_POS:
        right_wrist_pos = current_action[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX]
        return right_wrist_pos
    elif mode == ActionComponentMode.RIGHT_EEF_QUAT:
        right_wrist_quat = current_action[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX]
        return right_wrist_quat
    elif mode == ActionComponentMode.NAVIGATE_CMD:
        navigate_cmd = current_action[:, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX]
        return navigate_cmd
    elif mode == ActionComponentMode.BASE_HEIGHT_CMD:
        base_height_cmd = current_action[:, BASE_HEIGHT_CMD_START_IDX:BASE_HEIGHT_CMD_END_IDX]
        return base_height_cmd
    elif mode == ActionComponentMode.TORSO_ORIENTATION_RPY_CMD:
        torso_orientation_rpy_cmd = current_action[
            :, TORSO_ORIENTATION_RPY_CMD_START_IDX:TORSO_ORIENTATION_RPY_CMD_END_IDX
        ]
        return torso_orientation_rpy_cmd
    else:
        raise ValueError(f"Invalid action component mode: {mode}")


def is_navigating(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    return torch.tensor([copy.deepcopy(env.action_manager.get_term("g1_action").is_navigating)])


def navigation_goal_reached(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    return torch.tensor([copy.deepcopy(env.action_manager.get_term("g1_action").navigation_goal_reached)])
