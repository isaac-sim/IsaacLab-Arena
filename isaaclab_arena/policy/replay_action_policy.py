# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import torch
from dataclasses import dataclass, field
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab.utils.datasets import HDF5DatasetFileHandler

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class ReplayActionPolicyArgs:
    """
    Configuration dataclass for ReplayActionPolicy.

    This dataclass serves as the single source of truth for policy configuration,
    supporting both dict-based (from JSON) and CLI-based configuration paths.

    Field metadata is used to auto-generate argparse arguments, ensuring consistency
    between the dataclass definition and CLI argument parsing.
    """

    replay_file_path: str = field(
        metadata={
            "help": "Path to the HDF5 file containing the episode",
            "required": True,
        }
    )

    device: str = field(
        default="cuda",
        metadata={
            "help": "Device to use for loading the dataset",
        },
    )

    episode_name: str | None = field(
        default=None,
        metadata={
            "help": "Name of the episode to replay. If not provided, the first episode will be replayed",
        },
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ReplayActionPolicyArgs":
        """
        Create configuration from parsed CLI arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            ReplayActionPolicyArgs instance
        """
        return cls(
            replay_file_path=args.replay_file_path,
            device=getattr(args, "device", "cuda"),
            episode_name=args.episode_name,
        )


@register_policy
class ReplayActionPolicy(PolicyBase):
    """
    Replay the actions from an named episode stored in a HDF5 file.
    If no episode name is provided, the first episode will be replayed.
    """

    name = "replay"

    def __init__(self, config: ReplayActionPolicyArgs):
        """
        Initialize ReplayActionPolicy from a configuration dataclass.

        Args:
            config: ReplayActionPolicyArgs configuration dataclass
        """
        super().__init__()
        self.config = config
        self.episode_name = config.episode_name
        self.dataset_file_handler = HDF5DatasetFileHandler()
        self.dataset_file_handler.open(config.replay_file_path)
        self.available_episode_names = list(self.dataset_file_handler.get_episode_names())

        # Take the first episode if no episode name is provided
        if self.episode_name is None:
            self.episode_name = self.available_episode_names[0]
        else:
            assert self.episode_name in self.available_episode_names, (
                f"Episode {self.episode_name} not found in {config.replay_file_path}."
                f"Available episodes: {self.available_episode_names}"
            )

        self.episode_data = self.dataset_file_handler.load_episode(self.episode_name, device=config.device)
        self.current_action_index = 0

    def __len__(self) -> int:
        """Return the number of actions in the episode."""
        return len(self.episode_data.data["actions"])

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor | None:
        """Get the action of the next current index from the dataset."""
        action = self.get_action_from_index(self.current_action_index)
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Add batch dimension
        if action is not None:
            self.current_action_index += 1
        return action

    def get_action_from_index(self, action_index: int) -> torch.Tensor | None:
        """Get the action of the specified index from the dataset."""
        if "actions" not in self.episode_data.data:
            return None
        if action_index >= len(self.episode_data.data["actions"]):
            return None
        return self.episode_data.data["actions"][action_index]

    def get_available_episode_names(self):
        return self.available_episode_names

    def get_initial_state(self) -> torch.Tensor:
        return self.episode_data.get_initial_state()

    def has_length(self) -> bool:
        """Check if the policy is based on a recording (i.e. is a dataset-driven policy)."""
        return True

    def length(self) -> int:
        """Get the length of the policy (for dataset-driven policies)."""
        return len(self)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add replay action policy specific arguments to the parser."""
        replay_group = parser.add_argument_group("Replay Action Policy", "Arguments for replay action policy")
        replay_group.add_argument(
            "--replay_file_path",
            type=str,
            required=True,
            help="Path to the HDF5 file containing the episode",
        )
        replay_group.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device to use for loading the dataset (default: cuda)",
        )
        replay_group.add_argument(
            "--episode_name",
            type=str,
            default=None,
            help="Name of the episode to replay. If not provided, the first episode will be replayed",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ReplayActionPolicy":
        """
        Create a ReplayActionPolicy instance from parsed CLI arguments.

        Path: CLI args → ConfigDataclass → init cls

        Args:
            args: Parsed command line arguments

        Returns:
            ReplayActionPolicy instance
        """
        config = ReplayActionPolicyArgs.from_cli_args(args)
        return ReplayActionPolicy(config)
