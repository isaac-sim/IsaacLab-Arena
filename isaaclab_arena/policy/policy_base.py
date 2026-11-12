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
from typing import Generic, TypeVar


@dataclass
class PolicyCfg:
    """Mark a typed Arena policy configuration."""


PolicyCfgT = TypeVar("PolicyCfgT", bound=PolicyCfg)


class PolicyBase(ABC, Generic[PolicyCfgT]):
    """Define runtime behavior shared by Arena policies."""

    def __init__(self, config: PolicyCfgT):
        self.config = config

    @classmethod
    def default_image_size(cls, config: PolicyCfgT) -> tuple[int, int] | None:
        """Return the (height, width) of the camera images this policy is designed for, or None if it has none.

        Policies accept images of any size; this is the size that their image preprocessing targets.
        """
        return None  # noqa: R501 - subclasses with image inputs return a size

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action for every environment given the environment and observation.

        The environment ignores the actions of environments whose episode ended without a replacement,
        so the policy may skip computing them.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        return self._get_action(env, observation, self._get_active_env_mask(env))

    @abstractmethod
    def _get_action(self, env: gym.Env, observation: GymSpacesDict, active_env_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute an action for every environment given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.
            active_env_mask: Boolean (num_envs,) CPU mask of environments whose actions are used. The other
                rows are ignored, so the policy may fill them with any finite values.

        Returns:
            torch.Tensor: The action to take.
        """
        raise NotImplementedError("Function not implemented yet.")

    def _get_active_env_mask(self, env: gym.Env) -> torch.Tensor:
        """Return a boolean (num_envs,) CPU mask of environments whose actions are used on this step."""
        finished_episode_mask = getattr(env.unwrapped, "finished_episode_mask", None)
        if finished_episode_mask is None:
            return torch.ones(env.unwrapped.num_envs, dtype=torch.bool)
        return ~finished_episode_mask.cpu()

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
