# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
