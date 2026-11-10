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

    inference_latency_ms: float = field(default=0.0, kw_only=True)
    """Time the deployment needs between capturing an observation and applying the first action
    computed from it, in milliseconds.

    Modeled in simulation time as ``ceil(inference_latency_ms / control_step_dt)`` control steps: an action
    computed from the observation of step ``t`` is applied at step ``t + d``, and the robot holds its pose
    until the first action of an episode is available. The result does not depend on how fast the
    simulation or the policy actually run. 0 keeps inference instantaneous, which is the behavior
    without this setting.
    """


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

    @property
    def handles_inference_latency(self) -> bool:
        """Whether ``get_action`` already applies ``config.inference_latency_ms`` itself.

        Chunked policies schedule their requests around the latency. For every other policy the
        rollout loop delays the returned actions by the configured number of control steps.
        """
        return False
