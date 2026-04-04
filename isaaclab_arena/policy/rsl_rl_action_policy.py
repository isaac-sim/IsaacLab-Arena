# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import os
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase


class _RslRlInferenceEnvWrapper(RslRlVecEnvWrapper):
    """``RslRlVecEnvWrapper`` that skips ``env.reset()`` during ``__init__``.

    The base wrapper calls ``env.reset()`` because the RSL-RL training
    runner never calls reset itself.  For inference inside
    ``policy_runner.py`` the rollout loop already resets the env, so the
    extra reset records a phantom episode in the ``RecorderManager`` and
    inflates ``num_episodes`` by one.
    """

    def __init__(self, env: gym.Env, clip_actions: float | None = None):
        _original_reset = env.reset
        # return cached obs to env.reset() method to avoid triggering a record method in RecorderManager.
        env.reset = lambda *a, **kw: (env.unwrapped.obs_buf, {})
        try:
            # reset becomes a no-op
            super().__init__(env, clip_actions=clip_actions)
        finally:
            # restore the original reset method
            env.reset = _original_reset


@dataclass
class RslRlActionPolicyConfig:
    """Configuration dataclass for RSL-RL action policy."""

    checkpoint_path: str
    """Path to the RSL-RL checkpoint file.

    The agent config is loaded automatically from ``params/agent.yaml`` in the
    same directory, which is saved by IsaacLab's ``train.py`` alongside the checkpoint.
    """

    device: str = "cuda:0"
    """Device to run the policy on."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "RslRlActionPolicyConfig":
        return cls(
            checkpoint_path=args.checkpoint_path,
            device=args.device if hasattr(args, "device") else "cuda:0",
        )


@register_policy
class RslRlActionPolicy(PolicyBase):
    """Policy that uses a trained RSL-RL model for inference.

    Loads the checkpoint and agent config (``params/agent.yaml``) produced by
    IsaacLab's ``train.py``. No separate JSON config file is required.

    Example configuration for eval runner:

    .. code-block:: json

        {
          "jobs": [
            {
              "name": "eval_lift_cube",
              "policy_type": "rsl_rl",
              "policy_config_dict": {
                "checkpoint_path": "logs/rsl_rl/lift_object/2026-01-28_17-26-10/model_1000.pt",
                "device": "cuda:0"
              },
              "arena_env_args": ["lift_object", "--embodiment", "franka_ik"]
            }
          ]
        }
    """

    name = "rsl_rl"
    config_class = RslRlActionPolicyConfig

    def __init__(self, config: RslRlActionPolicyConfig):
        super().__init__(config)
        self.config: RslRlActionPolicyConfig = config
        self._policy = None
        self._runner = None

    def _load_policy(self, env: gym.Env) -> None:
        """Load the RSL-RL policy from checkpoint and its accompanying agent.yaml."""
        checkpoint_path = retrieve_file_path(self.config.checkpoint_path)
        agent_yaml_path = os.path.join(os.path.dirname(checkpoint_path), "params", "agent.yaml")

        if not os.path.exists(agent_yaml_path):
            raise FileNotFoundError(
                f"No agent config found at {agent_yaml_path}. "
                "Ensure the checkpoint was produced by IsaacLab's train.py."
            )

        agent_cfg_dict = load_yaml(agent_yaml_path)
        agent_cfg_dict["device"] = self.config.device

        clip_actions = agent_cfg_dict.get("clip_actions")
        class_name = agent_cfg_dict.get("class_name", "OnPolicyRunner")

        wrapped_env = _RslRlInferenceEnvWrapper(env, clip_actions=clip_actions)

        if class_name == "OnPolicyRunner":
            self._runner = OnPolicyRunner(wrapped_env, agent_cfg_dict, log_dir=None, device=self.config.device)
        elif class_name == "DistillationRunner":
            self._runner = DistillationRunner(wrapped_env, agent_cfg_dict, log_dir=None, device=self.config.device)
        else:
            raise ValueError(f"Unsupported runner class: {class_name}")

        print(f"[INFO] Loading RSL-RL checkpoint from: {checkpoint_path}")
        self._runner.load(checkpoint_path)
        self._policy = self._runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if self._policy is None:
            self._load_policy(env)

        assert self._policy is not None, "Policy should be loaded after _load_policy()"

        with torch.inference_mode():
            return self._policy(observation)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RslRlActionPolicy":
        config = RslRlActionPolicyConfig(**config_dict)
        return cls(config)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add RSL-RL action policy specific arguments to the parser."""
        rsl_rl_group = parser.add_argument_group("RSL-RL Action Policy", "Arguments for RSL-RL action policy")
        rsl_rl_group.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help=(
                "Path to the checkpoint file. Agent config is loaded automatically from params/agent.yaml in the same"
                " directory."
            ),
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "RslRlActionPolicy":
        config = RslRlActionPolicyConfig.from_cli_args(args)
        return RslRlActionPolicy(config)
