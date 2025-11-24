# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab.utils.datasets import HDF5DatasetFileHandler

from isaaclab_arena.policy.policy_base import PolicyBase


class ReplayActionPolicy(PolicyBase):
    """
    Replay the actions from an named episode stored in a HDF5 file.
    If no episode name is provided, the first episode will be replayed.
    """

    def __init__(self, replay_file_path: str, device: str = "cuda", episode_name: str | None = None):
        super().__init__()
        self.episode_name = episode_name
        self.dataset_file_handler = HDF5DatasetFileHandler()
        self.dataset_file_handler.open(replay_file_path)
        self.available_episode_names = list(self.dataset_file_handler.get_episode_names())

        # Take the first episode if no episode name is provided
        if self.episode_name is None:
            self.episode_name = self.available_episode_names[0]
        else:
            assert self.episode_name in self.available_episode_names, (
                f"Episode {self.episode_name} not found in {replay_file_path}."
                f"Available episodes: {self.available_episode_names}"
            )

        self.episode_data = self.dataset_file_handler.load_episode(self.episode_name, device=device)
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
