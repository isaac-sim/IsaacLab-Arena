# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces.dict import Dict as GymSpacesDict


class PolicyBase(ABC):
    def __init__(self):
        """
        Base class for policies.
        """

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

    def set_language_instruction(self, language_instruction: str | None) -> str:
        """Set the language instruction of the task being evaluated."""
        self.language_instruction = language_instruction
        return self.language_instruction
