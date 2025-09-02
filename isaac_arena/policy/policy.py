# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gymnasium as gym
import numpy as np
import torch
from abc import ABC, abstractmethod


class PolicyBase(ABC):
    def __init__(self, env: gym.Env):
        """
        Base class for policies.

        Args:
            env: The environment whose action space this policy will use.
        """
        self.action_space = env.action_space
        self.device = torch.device(env.unwrapped.device)

    @abstractmethod
    def get_action(self, env: gym.Env, observation: dict[str, dict[str, np.ndarray]]) -> torch.Tensor:
        """
        Compute an action given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        pass


class ZeroActionPolicy(PolicyBase):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def get_action(self, env: gym.Env, observation: dict[str, dict[str, np.ndarray]]) -> torch.Tensor:
        """
        Always returns a zero action.
        """
        return torch.zeros(self.action_space.shape, device=self.device)


class ReplayPolicy(PolicyBase):
    """
    Replay the actions from a saved trajectory.
    """

    def __init__(self, replay_file_path: str):
        raise NotImplementedError("ReplayPolicy is not implemented")

    def get_action(self, env: gym.Env, observation: dict[str, dict[str, np.ndarray]]) -> torch.Tensor:
        pass
