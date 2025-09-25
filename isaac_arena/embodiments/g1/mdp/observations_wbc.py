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

import copy
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def get_navigate_cmd(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Get the P-controller navigate command."""
    return env.action_manager.get_term("g1_action").navigate_cmd.clone()


def extract_action_components(
    env: ManagerBasedEnv,
    mode: str,
):
    """Extract the individual components of the G1 WBC PINK action."""
    # get the current action
    current_action = env.action_manager.action.clone()

    if mode == "left_eef_pos":
        left_wrist_pos = current_action[:, 2:5]
        return left_wrist_pos
    elif mode == "left_eef_quat":
        left_wrist_quat = current_action[:, 5:9]
        return left_wrist_quat
    elif mode == "right_eef_pos":
        right_wrist_pos = current_action[:, 9:12]
        return right_wrist_pos
    elif mode == "right_eef_quat":
        right_wrist_quat = current_action[:, 12:16]
        return right_wrist_quat
    elif mode == "navigate_cmd":
        navigate_cmd = current_action[:, 16:19]
        return navigate_cmd
    elif mode == "base_height_cmd":
        base_height_cmd = current_action[:, 19:20]
        return base_height_cmd
    elif mode == "torso_orientation_rpy_cmd":
        torso_orientation_rpy_cmd = current_action[:, 20:23]
        return torso_orientation_rpy_cmd


def is_navigating(
    env: ManagerBasedEnv,
):
    return torch.tensor([copy.deepcopy(env.action_manager.get_term("g1_action").is_navigating)])


def navigation_goal_reached(
    env: ManagerBasedEnv,
):
    return torch.tensor([copy.deepcopy(env.action_manager.get_term("g1_action").navigation_goal_reached)])
