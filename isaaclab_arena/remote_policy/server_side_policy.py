# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .remote_policy_config import ClientPolicyConfig


class ServerSidePolicy(ABC):
    """Server-side policy interface."""

    def __init__(self) -> None:
        self._task_description: str | None = None

    @abstractmethod
    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute and return the next action based on current observation with validation.

        This is the main public interface. It validates the observation, calls
        the internal _get_action(), and validates the resulting action.

        Args:
            observation: Dictionary containing the current state/observation
            options: Optional configuration dict for action computation

        Returns:
            Tuple of (action, info):
                - action: Dictionary containing the validated action
                - info: Dictionary containing additional metadata

        Raises:
            AssertionError/ValueError: If observation or action validation fails
        """

    @abstractmethod
    def reset(
        self,
        env_ids: list[int] | None = None,
        reset_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the policy to its initial state.
    
        Args:
            env_ids: Optional list of environment indices to reset. If None, reset all.
            reset_options: Optional dictionary containing reset options.
    
        Returns:
            Dictionary containing the info after resetting the policy.
        """

    @abstractmethod
    def get_init_info(self) -> ClientPolicyConfig:
        """Return static initialization info for the client."""


    def set_task_description(self, task_description: str | None) -> str:
        """Default implementation that stores task description."""
