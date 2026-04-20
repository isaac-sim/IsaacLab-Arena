# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gymnasium as gym
import math
import torch
import yaml
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class RlGamesActionPolicyConfig:
    """Configuration for RL-Games action policy.

    Supports both dict-based (from JSON eval runner) and CLI-based configuration.
    """

    checkpoint_path: str
    """Path to the RL-Games .pth checkpoint file."""

    agent_cfg_path: Path = None
    """Path to the RL-Games agent YAML configuration file.

    When using the CLI (``policy_runner.py``), this is set via ``--agent_cfg_path``.
    """

    device: str = "cuda:0"
    """Device to run the policy on."""

    deterministic: bool = True
    """Use mean actions (no exploration noise) during evaluation."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> RlGamesActionPolicyConfig:
        return cls(
            checkpoint_path=args.checkpoint_path,
            agent_cfg_path=args.agent_cfg_path,
            device=args.device if hasattr(args, "device") else "cuda:0",
            deterministic=getattr(args, "deterministic", True),
        )


@register_policy
class RlGamesActionPolicy(PolicyBase):
    """Policy that uses a trained RL-Games model for inference.

    Wraps the RL-Games player for use with the Arena policy runner and eval runner.
    Handles observation processing (concatenation, clipping) and LSTM state management.

    """

    name = "rl_games"
    config_class = RlGamesActionPolicyConfig

    def __init__(self, config: RlGamesActionPolicyConfig, args_cli: argparse.Namespace | None = None):
        super().__init__(config)
        self.config: RlGamesActionPolicyConfig = config
        self._player: BasePlayer | None = None
        self._wrapper: RlGamesVecEnvWrapper | None = None
        self._rnn_initialized = False
        self.args_cli = args_cli

    def _load_policy(self, env: gym.Env) -> None:
        """Set up the RL-Games infrastructure and load the checkpoint.

        Creates the RlGamesVecEnvWrapper (for observation processing only),
        registers the env with the RL-Games runner, creates the player,
        and restores the checkpoint weights.
        """
        with open(self.config.agent_cfg_path) as f:
            agent_cfg = yaml.safe_load(f)

        device = self.config.device
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device

        resume_path = retrieve_file_path(self.config.checkpoint_path)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        self._wrapper = RlGamesVecEnvWrapper(
            env,
            device,
            clip_obs,
            clip_actions,
            obs_groups,
            concate_obs,
        )

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: self._wrapper},
        )

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        runner = Runner()
        runner.load(agent_cfg)
        self._player = runner.create_player()
        self._player.restore(resume_path)
        self._player.reset()

        print(f"[INFO] Loaded RL-Games checkpoint from: {resume_path}")

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if self._player is None:
            self._load_policy(env)

        assert self._player is not None
        assert self._wrapper is not None

        obs_copy = dict(observation)
        obs_rlg = self._wrapper._process_obs(obs_copy)
        obs_tensor = obs_rlg["obs"]

        if not self._rnn_initialized:
            _ = self._player.get_batch_size(obs_tensor, 1)
            if self._player.is_rnn:
                self._player.init_rnn()
            self._rnn_initialized = True

        obs_tensor = self._player.obs_to_torch(obs_tensor)

        with torch.inference_mode():
            action = self._player.get_action(obs_tensor, is_deterministic=self.config.deterministic)

        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._player is None or not self._player.is_rnn:
            return
        if not hasattr(self._player, "states") or self._player.states is None:
            return
        if env_ids is None:
            for s in self._player.states:
                s.zero_()
        else:
            for s in self._player.states:
                s[:, env_ids, :] = 0.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> RlGamesActionPolicy:
        config = RlGamesActionPolicyConfig(**config_dict)
        return cls(config, args_cli=None)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("RL-Games Action Policy", "Arguments for RL-Games action policy")
        group.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help="Path to the .pth checkpoint file containing the RL-Games policy",
        )
        group.add_argument(
            "--agent_cfg_path",
            type=Path,
            required=True,
            help="Path to the RL-Games agent YAML configuration file.",
        )
        group.add_argument(
            "--deterministic",
            action="store_true",
            default=True,
            help="Use mean actions without exploration noise (default: True).",
        )
        group.add_argument(
            "--stochastic",
            dest="deterministic",
            action="store_false",
            help="Use stochastic actions with exploration noise.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> RlGamesActionPolicy:
        config = RlGamesActionPolicyConfig.from_cli_args(args)
        return RlGamesActionPolicy(config, args_cli=args)
