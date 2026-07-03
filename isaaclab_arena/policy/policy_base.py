# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict
from typing import Any, Generic, Self, TypeVar


@dataclass
class PolicyCfg:
    """Mark a typed Arena policy configuration."""


PolicyCfgT = TypeVar("PolicyCfgT", bound=PolicyCfg)


class PolicyBase(ABC, Generic[PolicyCfgT]):
    """Define runtime behavior shared by Arena policies."""

    config_class: type[PolicyCfg] | None = None
    """Concrete config used by the legacy dictionary-based evaluation path."""

    def __init__(self, config: PolicyCfgT):
        self.config = config

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Create a policy through the legacy dictionary configuration path.

        Args:
            config_dict: Values used to instantiate ``config_class``.

        Returns:
            A policy initialized with its typed config.
        """
        if cls.config_class is None:
            raise NotImplementedError(f"{cls.__name__} must define 'config_class' to use from_dict()")

        config = cls.config_class(**config_dict)  # type: ignore[misc]
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

    def close(self) -> None:
        """Release resources held by the policy."""
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

    @property
    def is_remote(self) -> bool:
        """Check if policy is run remotely."""
        return False
