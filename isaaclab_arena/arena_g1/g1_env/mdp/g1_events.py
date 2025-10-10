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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_decoupled_wbc_joint_policy(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # Reset lower body RL-based policy
    policy = env.action_manager.get_term("g1_action").get_wbc_policy
    policy.lower_body_policy.reset(env_ids)


def reset_decoupled_wbc_pink_policy(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # Reset upper body IK solver
    env.action_manager.get_term("g1_action").upperbody_controller.body_ik_solver.initialize()
    env.action_manager.get_term("g1_action").upperbody_controller.in_warmup = True

    # Reset lower body RL-based policy
    policy = env.action_manager.get_term("g1_action").get_wbc_policy
    policy.lower_body_policy.reset(env_ids)

    # Reset P-controller
    env.action_manager.get_term("g1_action")._is_navigating = False
    env.action_manager.get_term("g1_action")._navigation_goal_reached = False
    env.action_manager.get_term("g1_action")._num_navigation_subgoals_reached = -1
