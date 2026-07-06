# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class ZeroActionPolicyCfg(PolicyCfg):
    """Configure a policy that always returns zero actions."""


@register_policy
class ZeroActionPolicy(PolicyBase[ZeroActionPolicyCfg]):

    name = "zero_action"

    def __init__(self, config: ZeroActionPolicyCfg):
        """
        Initialize ZeroActionPolicy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Always returns a zero action.
        """
        return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))
