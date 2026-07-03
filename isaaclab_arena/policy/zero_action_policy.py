# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
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
    # enable from_dict() from policy_base.PolicyBase
    config_class = ZeroActionPolicyCfg

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

    # TODO(cvolk, 2026-07-03): Move this legacy argparse adapter into the policy CLI frontend.
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add zero action policy specific arguments to the parser.

        This policy has no configuration parameters, so no arguments are added.

        Args:
            parser: The argument parser to add arguments to

        Returns:
            The updated argument parser (unchanged)
        """
        # No additional command line arguments for zero action policy
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ZeroActionPolicy":
        """
        Create a ZeroActionPolicy instance from parsed CLI arguments.

        Path: CLI args → ConfigDataclass → init cls

        Args:
            args: Parsed command line arguments

        Returns:
            ZeroActionPolicy instance
        """
        return ZeroActionPolicy(ZeroActionPolicyCfg())
