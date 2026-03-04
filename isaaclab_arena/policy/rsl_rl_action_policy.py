# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.rl_policy.base_rsl_rl_policy import get_agent_cfg

# from isaaclab_arena.scripts.reinforcement_learning import cli_args


@dataclass
class RslRlActionPolicyConfig:
    """
    Configuration dataclass for RSL-RL action policy.

    This dataclass serves as the single source of truth for policy configuration,
    supporting both dict-based (from JSON) and CLI-based configuration paths.
    """

    checkpoint_path: str
    """Path to the RSL-RL checkpoint file."""

    agent_cfg_path: Path = Path("isaaclab_arena/policy/rl_policy/generic_policy.json")
    """Path to the RL agent configuration file."""

    device: str = "cuda:0"
    """Device to run the policy on."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "RslRlActionPolicyConfig":
        """
        Create configuration from parsed CLI arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            RslRlActionPolicyConfig instance
        """
        return cls(
            checkpoint_path=args.checkpoint_path,
            agent_cfg_path=args.agent_cfg_path,
            device=args.device if hasattr(args, "device") else "cuda:0",
        )


@register_policy
class RslRlActionPolicy(PolicyBase):
    """
    Policy that uses a trained RSL-RL model for inference.

    This policy loads a checkpoint from RSL-RL training and uses it to generate
    actions. It expects the environment to already be wrapped with RslRlVecEnvWrapper
    if called from evaluation scripts.

    Example JSON configuration for eval runner:

    .. code-block:: json

        {
          "jobs": [
            {
              "name": "eval_lift_cube",
              "policy_type": "rsl_rl",
              "policy_config_dict": {
                "checkpoint_path": "logs/rsl_rl/lift_object/model_1000.pt",
                "agent_cfg_path": "isaaclab_arena/policy/rl_policy/generic_policy.json",
                "device": "cuda:0",
              },
              "arena_env_args": ["lift_object", "--embodiment", "franka"]
            }
          ]
        }
    """

    name = "rsl_rl"
    config_class = RslRlActionPolicyConfig

    def __init__(self, config: RslRlActionPolicyConfig, args_cli: argparse.Namespace | None = None):
        """
        Initialize RSL-RL action policy from a configuration dataclass.

        Args:
            config: RslRlActionPolicyConfig configuration dataclass
            args_cli: Optional CLI arguments namespace. If provided, uses get_agent_cfg().
                     If None, loads agent config directly from JSON file.
        """
        super().__init__(config)
        self.config: RslRlActionPolicyConfig = config
        self._policy = None
        self._runner = None
        self._env_is_wrapped = False
        self.args_cli = args_cli

    def _load_policy(self, env: gym.Env) -> None:
        """
        Load the RSL-RL policy from checkpoint.

        Args:
            env: The gym environment (should already be wrapped with RslRlVecEnvWrapper)
        """
        import json

        # Load agent configuration
        # Prefer using get_agent_cfg() if args_cli is available (more robust)
        # Otherwise, load directly from JSON (for from_dict() path)
        if self.args_cli is not None:
            agent_cfg = get_agent_cfg(self.args_cli)
        else:
            # Fallback: Load agent configuration directly from JSON file
            with open(self.config.agent_cfg_path) as f:
                agent_cfg_dict = json.load(f)

            # Import the config class and create agent config
            from isaaclab_arena.policy.rl_policy.base_rsl_rl_policy import RLPolicyCfg

            policy_cfg = agent_cfg_dict["policy_cfg"]
            algorithm_cfg = agent_cfg_dict["algorithm_cfg"]
            obs_groups = agent_cfg_dict.get("obs_groups", {})

            # Use defaults for training-specific parameters (not needed for inference)
            num_steps_per_env = agent_cfg_dict.get("num_steps_per_env", 24)
            max_iterations = agent_cfg_dict.get("max_iterations", 1500)
            save_interval = agent_cfg_dict.get("save_interval", 100)
            experiment_name = agent_cfg_dict.get("experiment_name", "rsl_rl")

            agent_cfg = RLPolicyCfg.update_cfg(
                policy_cfg, algorithm_cfg, obs_groups, num_steps_per_env, max_iterations, save_interval, experiment_name
            )

        # Override device from config
        agent_cfg.device = self.config.device

        # Check if environment is already wrapped
        if isinstance(env, RslRlVecEnvWrapper):
            wrapped_env = env
            self._env_is_wrapped = True
        else:
            # Wrap if needed (for standalone policy runner usage)
            wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            self._env_is_wrapped = False

        # Create the appropriate runner
        if agent_cfg.class_name == "OnPolicyRunner":
            self._runner = OnPolicyRunner(
                wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device  # type: ignore[attr-defined]
            )
        elif agent_cfg.class_name == "DistillationRunner":
            self._runner = DistillationRunner(
                wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device  # type: ignore[attr-defined]
            )
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        # Load the checkpoint
        checkpoint_path = retrieve_file_path(self.config.checkpoint_path)
        print(f"[INFO] Loading RSL-RL checkpoint from: {checkpoint_path}")
        self._runner.load(checkpoint_path)

        # Get the inference policy
        self._policy = self._runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Get the action from the RSL-RL policy.

        Args:
            env: The gym environment
            observation: Current observation from the environment

        Returns:
            Action tensor from the policy
        """
        # Load policy on first call
        if self._policy is None:
            self._load_policy(env)

        # Type checker doesn't know _policy is not None after _load_policy
        assert self._policy is not None, "Policy should be loaded after _load_policy()"

        with torch.inference_mode():
            return self._policy(observation)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the policy state for specific environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        # RSL-RL policies are typically stateless for evaluation
        # Override if your policy has recurrent components
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RslRlActionPolicy":
        """
        Create a policy instance from a configuration dictionary.

        This override ensures args_cli is None when loading from JSON config.

        Args:
            config_dict: Dictionary containing the configuration fields

        Returns:
            RslRlActionPolicy instance
        """
        config = RslRlActionPolicyConfig(**config_dict)
        return cls(config, args_cli=None)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add RSL-RL action policy specific arguments to the parser."""
        rsl_rl_group = parser.add_argument_group("RSL-RL Action Policy", "Arguments for RSL-RL action policy")
        rsl_rl_group.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help="Path to the checkpoint file containing the RSL-RL policy",
        )
        rsl_rl_group.add_argument(
            "--agent_cfg_path",
            type=Path,
            default=Path("isaaclab_arena/policy/rl_policy/generic_policy.json"),
            help="Path to the RL agent configuration file.",
        )
        # append RSL-RL cli arguments
        cli_args.add_rsl_rl_args(parser)
        cli_args.add_rsl_rl_policy_args(parser)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "RslRlActionPolicy":
        """
        Create a RSL-RL action policy instance from parsed CLI arguments.

        Path: CLI args → ConfigDataclass → init cls

        Args:
            args: Parsed command line arguments

        Returns:
            RslRlActionPolicy instance
        """
        config = RslRlActionPolicyConfig.from_cli_args(args)
        return RslRlActionPolicy(config, args_cli=args)
