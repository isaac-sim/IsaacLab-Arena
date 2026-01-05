# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.policy_base import PolicyBase


class ZeroActionPolicy(PolicyBase):
    def __init__(self):
        super().__init__()

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Always returns a zero action.
        """
        return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add zero action policy specific arguments to the parser."""
        # No additional command line arguments for zero action policy
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ZeroActionPolicy":
        """Create a zero action policy from the arguments."""
        return ZeroActionPolicy()
