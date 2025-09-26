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
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces.dict import Dict as GymSpacesDict

from gr00t.data.dataset import LeRobotSingleDataset

class ReplayLerobotActionPolicy(ABC):
    def __init__(self, policy_config: dict):
        """
        Base class for replay action policies from Lerobot dataset.
        """
        self.policy_config = policy_config
        self.policy = self._load_policy()
        self.policy_iter = iter(self.policy)
        self._load_policy_joints_config()
        self._load_sim_joints_config()

    def _load_policy_joints_config(self):
        """Load the policy joint config from the data config."""
        self.gr00t_joints_config = load_robot_joints_config(self.policy_config.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """Load the simulation joint config from the data config."""
        self.g1_state_joints_config = load_robot_joints_config(self.policy_config.state_joints_config_path)
        self.g1_action_joints_config = load_robot_joints_config(self.policy_config.action_joints_config_path)

    def _load_policy(self):
        """Load the policy from the model path."""
        assert os.path.exists(self.args.dataset_path), f"Dataset path {self.args.dataset_path} does not exist"

        # Use the same data preprocessor as the loaded fine-tuned ckpts
        self.data_config = DATA_CONFIG_MAP[self.args.data_config]

        modality_config = self.data_config.modality_config()

        return LeRobotSingleDataset(
            dataset_path=self.args.dataset_path,
            modality_configs=modality_config,
            video_backend=self.args.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=self.args.embodiment_tag,
        )

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
        pass
