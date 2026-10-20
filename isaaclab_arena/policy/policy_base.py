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
    def camera_observation_interval_steps(cls, config: PolicyCfgT) -> int:
        """Return how often the policy needs fresh camera observations, in control steps."""
        return 1

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


def resolve_camera_update_period_steps(
    policy_type: type[PolicyBase],
    policy_cfg: PolicyCfg,
    configured_period_steps: int | None,
    record_camera_video: bool,
) -> int:
    """Resolve an explicit, video, or policy-selected camera cadence."""
    if configured_period_steps is not None:
        return configured_period_steps
    if record_camera_video:
        return 1
    period_steps = policy_type.camera_observation_interval_steps(policy_cfg)
    assert (
        isinstance(period_steps, int) and not isinstance(period_steps, bool) and period_steps > 0
    ), f"{policy_type.__name__}.camera_observation_interval_steps() must return a positive integer"
    return period_steps
