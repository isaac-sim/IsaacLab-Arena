# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict
from pathlib import Path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.scripts.reinforcement_learning.utils import get_agent_cfg

import cli_args  # isort: skip


@register_policy
class RslRlActionPolicy(PolicyBase):
    """
    Send the actions from a RSL-RL policy to the environment.
    """

    name = "rsl_rl"

    def __init__(self, args_cli: argparse.Namespace, checkpoint_path: str):
        super().__init__()
        self.rsl_rl_policy = None
        self.args_cli = args_cli
        self.checkpoint_path = checkpoint_path

    def set_policy(self, env: gym.Env) -> None:
        """Set the RSL-RL policy."""

        agent_cfg = get_agent_cfg(self.args_cli)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(self.args_cli.checkpoint_path)

        self.rsl_rl_policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Get the action from the RSL-RL policy."""
        if self.rsl_rl_policy is None:
            self.set_policy(env)

        return self.rsl_rl_policy(observation)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add replay action policy specific arguments to the parser."""
        rsl_rl_group = parser.add_argument_group("RSL-RL Action Policy", "Arguments for RSL-RL action policy")
        rsl_rl_group.add_argument(
            "--checkpoint_path",
            type=str,
            help="Path to the checkpoint file containing the RSL-RL policy (required with --policy_type rsl_rl)",
        )
        rsl_rl_group.add_argument(
            "--agent_cfg_path",
            type=Path,
            default=Path("isaaclab_arena/policy/rl_policy/generic_policy.json"),
            help="Path to the RL agent configuration file.",
        )
        cli_args.add_rsl_rl_args(rsl_rl_group)
        cli_args.add_rsl_rl_policy_args(rsl_rl_group)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "RslRlActionPolicy":
        """Create a replay action policy from the arguments."""
        return RslRlActionPolicy(
            args_cli=args,
            checkpoint_path=args.checkpoint_path,
        )
