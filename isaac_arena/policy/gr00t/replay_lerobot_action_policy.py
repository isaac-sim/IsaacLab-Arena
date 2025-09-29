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

import os
import gymnasium as gym
import torch
import numpy as np
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaac_arena.policy.data_utils.io_utils import load_robot_joints_config_from_yaml
from isaac_arena.policy.policy_base import PolicyBase
from isaac_arena.policy.data_utils.joints_conversion import remap_policy_joints_to_sim_joints

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP

class ReplayLerobotActionPolicy(PolicyBase):
    def __init__(self, policy_config: dict):
        """
        Base class for replay action policies from Lerobot dataset.
        """
        self.policy_config = policy_config
        self.policy = self._load_policy()
        self.policy_iter = iter(self.policy)
        # determine rollout how many action prediction per observation
        self.num_feedback_actions = self.policy_config.num_feedback_actions
        self.current_action_index = 0
        self.current_action_chunk = None

        self._load_policy_joints_config()
        self._load_sim_joints_config()

    def _load_policy_joints_config(self):
        """Load the policy joint config from the data config."""
        self.gr00t_joints_config = load_robot_joints_config_from_yaml(self.policy_config.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """Load the simulation joint config from the data config."""
        self.g1_state_joints_config = load_robot_joints_config_from_yaml(self.policy_config.state_joints_config_path)
        self.g1_action_joints_config = load_robot_joints_config_from_yaml(self.policy_config.action_joints_config_path)

    def _load_policy(self):
        """Load the policy from the model path."""
        assert os.path.exists(self.policy_config.dataset_path), f"Dataset path {self.policy_config.dataset_path} does not exist"

        # Use the same data preprocessor as the loaded fine-tuned ckpts
        self.data_config = DATA_CONFIG_MAP[self.policy_config.data_config]

        modality_config = self.data_config.modality_config()

        return LeRobotSingleDataset(
            dataset_path=self.policy_config.dataset_path,
            modality_configs=modality_config,
            video_backend=self.policy_config.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=self.policy_config.embodiment_tag,
        )
    def get_trajectory_length(self):
        return len(self.policy.trajectory_lengths)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        # get new predictions and return the first action from the chunk
        if self.current_action_chunk is None and self.current_action_index == 0:
            self.current_action_chunk = self.get_action_chunk()
            assert self.current_action_chunk.shape[1] >= self.num_feedback_actions
        assert self.current_action_chunk is not None
        assert self.current_action_index < self.num_feedback_actions
        action = self.current_action_chunk[:, self.current_action_index]
        assert action.ndim == 2, f"Action is {action.shape} but should be (B, action_dim)"

        self.current_action_index += 1
        # reset to empty action chunk
        if self.current_action_index == self.num_feedback_actions:
            self.current_action_chunk = None
            self.current_action_index = 0
        return action

    def get_action_chunk(self) -> torch.Tensor:
        # Run policy prediction on the given observations. Produce a new action goal for the robot.
        # Args:
        #     current_state: robot proprioceptive state observation
        #     ego_camera: camera sensor observation
        #     language_instruction: language instruction for the task
        # Returns:
        #     A dictionary containing the inferred action for robot joints.

        data_point = next(self.policy_iter)
        actions = {
            "action.left_arm": np.tile(np.array(data_point["action.left_arm"]), (self.args.num_envs, 1, 1)),
            "action.right_arm": np.tile(np.array(data_point["action.right_arm"]), (self.args.num_envs, 1, 1)),
            "action.left_hand": np.tile(np.array(data_point["action.left_hand"]), (self.args.num_envs, 1, 1)),
            "action.right_hand": np.tile(np.array(data_point["action.right_hand"]), (self.args.num_envs, 1, 1)),
            "action.base_height_command": np.tile(np.array(data_point["action.base_height_command"]), (self.args.num_envs, 1, 1)),
            "action.navigate_command": np.tile(np.array(data_point["action.navigate_command"]), (self.args.num_envs, 1, 1)),
            "action.torso_orientation_rpy_command": np.tile(np.array(data_point["action.torso_orientation_rpy_command"]), (self.args.num_envs, 1, 1)),
        }

        robot_action_sim = remap_policy_joints_to_sim_joints(
            actions, self.gr00t_joints_config, self.g1_action_joints_config, self.args.simulation_device
        )
        # concat along axis = 1
        action_tensor = torch.cat([
            robot_action_sim.get_joints_pos(),
            actions["action.base_height_command"],
            actions["action.navigate_command"],
            actions["action.torso_orientation_rpy_command"],
        ], axis=2)
        assert action_tensor.shape[1] >= self.num_feedback_actions
        return action_tensor

    def reset(self):
        """Resets the policy's internal state."""
        # As GR00T is a single-shot policy, we don't need to reset its internal state
        self.policy_iter = iter(self.policy)
        self.current_action_chunk = None
        self.current_action_index = 0