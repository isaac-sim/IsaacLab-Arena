# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces.dict import Dict as GymSpacesDict
from typing import Any


class PolicyBase(ABC):
    """
    Base class for policies.

    Subclasses should define a `config_class` class variable pointing to their configuration dataclass
    to enable configuration from dictionaries via the from_dict() method.
    """

    # Optional: Subclasses can define this to enable from_dict()
    config_class: type | None = None

    def __init__(self, config: Any):
        """
        Base class for policies.
        """
        self.config = config

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PolicyBase":
        """
        Create a policy instance from a configuration dictionary.

        This method instantiates the policy's config_class from the dict and then
        creates the policy from that config.

        Path: dict → ConfigDataclass → Policy instance

        Args:
            config_dict: Dictionary containing the configuration fields

        Returns:
            Policy instance
        """
        if cls.config_class is None:
            raise NotImplementedError(f"{cls.__name__} must define 'config_class' to use from_dict()")

        # Create config from dict
        config = cls.config_class(**config_dict)  # type: ignore[misc]

        # Create policy from config
        return cls(config)  # type: ignore[call-arg]

    @abstractmethod
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        raise NotImplementedError("Function not implemented yet.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the policy.
        """
        pass

    def set_task_description(self, task_description: str | None) -> str:
        """Set the task description of the task being evaluated."""
        self.task_description = task_description
        return self.task_description

    def has_length(self) -> bool:
        """Check if the policy is based on a recording (i.e. is a dataset-driven policy)."""
        return False

    def length(self) -> int | None:
        """Get the length of the policy (for dataset-driven policies)."""
        pass

    @staticmethod
    @abstractmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add policy-specific arguments to the parser."""
        raise NotImplementedError("Function not implemented yet.")

    @staticmethod
    @abstractmethod
    def from_args(args: argparse.Namespace) -> "PolicyBase":
        """Create a policy from the arguments."""
        raise NotImplementedError("Function not implemented yet.")
