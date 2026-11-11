# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from gymnasium.spaces.dict import Dict as GymSpacesDict
from typing import Generic, TypeVar


@dataclass
class PolicyCfg:
    """Mark a typed Arena policy configuration."""

    hold_action_while_inactive: bool = field(default=True, kw_only=True)
    """Hold the last action, without computing a new one, in environments whose episode ended without a replacement."""


PolicyCfgT = TypeVar("PolicyCfgT", bound=PolicyCfg)


class PolicyBase(ABC, Generic[PolicyCfgT]):
    """Define runtime behavior shared by Arena policies."""

    def __init__(self, config: PolicyCfgT):
        self.config = config
        self._last_actions: torch.Tensor | None = None
        """Last action returned for each environment, or None before the first step."""
        self._has_last_action: torch.Tensor | None = None
        """Whether each environment has a last action to hold, on the CPU."""

    @classmethod
    def default_image_size(cls, config: PolicyCfgT) -> tuple[int, int] | None:
        """Return the (height, width) of the camera images this policy is designed for, or None if it has none.

        Policies accept images of any size; this is the size that their image preprocessing targets.
        """
        return None  # noqa: R501 - subclasses with image inputs return a size

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action for every environment given the environment and observation.

        With hold_action_while_inactive, an environment whose episode ended without a replacement
        holds its last action instead. Holding keeps the robot still only for absolute action targets.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        inactive_env_mask = self._get_inactive_env_mask(env)
        actions = self._get_action(env, observation, ~inactive_env_mask)
        if actions is None:
            return actions
        if self._last_actions is not None:
            hold_mask = inactive_env_mask.to(actions.device).reshape(-1, *[1] * (actions.dim() - 1))
            actions = torch.where(hold_mask, self._get_hold_action(env, observation), actions)
        self._last_actions = actions.clone()
        self._has_last_action = torch.ones_like(inactive_env_mask)
        return actions

    @abstractmethod
    def _get_action(self, env: gym.Env, observation: GymSpacesDict, active_env_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute an action for every environment given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.
            active_env_mask: Boolean (num_envs,) CPU mask of environments that need a new action. Rows of the
                other environments are replaced by their hold action, so the policy may skip computing them.

        Returns:
            torch.Tensor: The action to take.
        """
        raise NotImplementedError("Function not implemented yet.")

    def _get_hold_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Return the actions that inactive environments hold, defaulting to each one's last action."""
        return self._last_actions

    def _get_inactive_env_mask(self, env: gym.Env) -> torch.Tensor:
        """Return a boolean (num_envs,) CPU mask of environments that hold their last action on this step."""
        num_envs = env.unwrapped.num_envs
        active_episode_mask = getattr(env.unwrapped, "active_episode_mask", None)
        if not self.config.hold_action_while_inactive or active_episode_mask is None or self._has_last_action is None:
            return torch.zeros(num_envs, dtype=torch.bool)
        return ~active_episode_mask.cpu() & self._has_last_action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the policy, discarding the last actions of selected environments, or all environments.

        Subclasses that override this call it too.
        """
        if env_ids is None:
            self._last_actions = None
            self._has_last_action = None
        elif self._has_last_action is not None:
            self._has_last_action[env_ids.reshape(-1).cpu()] = False

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
