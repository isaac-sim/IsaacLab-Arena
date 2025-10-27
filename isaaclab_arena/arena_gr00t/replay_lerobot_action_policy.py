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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config

from isaaclab_arena.arena_gr00t.policy_config import LerobotReplayActionPolicyConfig, TaskMode
from isaaclab_arena.policy.data_utils.io_utils import create_config_from_yaml, load_robot_joints_config_from_yaml
from isaaclab_arena.policy.data_utils.joints_conversion import remap_policy_joints_to_sim_joints
from isaaclab_arena.policy.policy_base import PolicyBase


class ReplayLerobotActionPolicy(PolicyBase):
    def __init__(
        self, policy_config_yaml_path: Path, num_envs: int = 1, device: str = "cuda", trajectory_index: int = 0
    ):
        """
        Base class for replay action policies from Lerobot dataset.
        """

        self.policy_config = create_config_from_yaml(policy_config_yaml_path, LerobotReplayActionPolicyConfig)
        self.policy = self.load_policy(self.policy_config)
        # Start from the trajectory_index trajectory in the dataset
        self.trajectory_index = trajectory_index
        self.policy_iter = self.create_trajectory_iterator(trajectory_index)
        # determine rollout how many action prediction per observation
        self.num_feedback_actions = self.policy_config.num_feedback_actions
        self.current_action_index = 0
        self.current_action_chunk = None
        self.num_envs = num_envs
        self.device = device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        self.policy_joints_config = self.load_policy_joints_config(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = self.load_sim_joints_config(self.policy_config.action_joints_config_path)

    def load_policy_joints_config(self, policy_config_path: Path) -> dict[str, Any]:
        """Load the policy joint config from the data config."""
        return load_robot_joints_config_from_yaml(policy_config_path)

    def load_sim_joints_config(self, action_config_path: Path) -> dict[str, Any]:
        """Load the simulation joint config from the data config."""
        return load_robot_joints_config_from_yaml(action_config_path)

    def load_policy(self, policy_config: LerobotReplayActionPolicyConfig) -> LeRobotSingleDataset:
        """Load the dataset, whose iterator will be used as the policy."""
        assert Path(policy_config.dataset_path).exists(), f"Dataset path {policy_config.dataset_path} does not exist"

        # Use the same data preprocessor specified in the  data config map
        if policy_config.data_config in DATA_CONFIG_MAP:
            self.data_config = DATA_CONFIG_MAP[policy_config.data_config]
        elif policy_config.data_config == "unitree_g1_sim_wbc":
            self.data_config = load_data_config("isaaclab_arena.arena_gr00t.data_config:UnitreeG1SimWBCDataConfig")
        else:
            raise ValueError(f"Invalid data config: {policy_config.data_config}")

        modality_config = self.data_config.modality_config()

        return LeRobotSingleDataset(
            dataset_path=policy_config.dataset_path,
            modality_configs=modality_config,
            video_backend=policy_config.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=self.policy_config.embodiment_tag,
        )

    def get_trajectory_length(self, trajectory_index: int) -> int:
        """Get the number of frames in one trajectory in the dataset."""
        assert self.policy.trajectory_lengths is not None
        assert trajectory_index < len(self.policy.trajectory_lengths)
        return self.policy.trajectory_lengths[trajectory_index]

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return action from the dataset."""
        # get new predictions and return the first action from the chunk
        if self.current_action_chunk is None and self.current_action_index == 0:
            self.current_action_chunk = self.get_action_chunk()
            assert self.current_action_chunk.shape[1] >= self.num_feedback_actions

        assert self.current_action_chunk is not None
        assert self.current_action_index < self.num_feedback_actions

        action = self.current_action_chunk[:, self.current_action_index]
        assert action.shape == env.action_space.shape, f"{action.shape=} != {env.action_space.shape=}"

        self.current_action_index += 1
        # reset to empty action chunk
        if self.current_action_index == self.num_feedback_actions:
            self.current_action_chunk = None
            self.current_action_index = 0
        return action

    def get_action_chunk(self) -> torch.Tensor:
        """Get action_horizon number of actions, as an action chunk, from the dataset"""
        step_index = next(self.policy_iter)
        data_point = self.policy[step_index]
        # Support MultiEnv running
        actions = {
            "action.left_arm": np.tile(np.array(data_point["action.left_arm"]), (self.num_envs, 1, 1)),
            "action.right_arm": np.tile(np.array(data_point["action.right_arm"]), (self.num_envs, 1, 1)),
            "action.left_hand": np.tile(np.array(data_point["action.left_hand"]), (self.num_envs, 1, 1)),
            "action.right_hand": np.tile(np.array(data_point["action.right_hand"]), (self.num_envs, 1, 1)),
        }

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            # additional data for WBC interface
            actions["action.base_height_command"] = np.tile(
                np.array(data_point["action.base_height_command"]), (self.num_envs, 1, 1)
            )
            actions["action.navigate_command"] = np.tile(
                np.array(data_point["action.navigate_command"]), (self.num_envs, 1, 1)
            )
            # NOTE(xinjieyao, 2025-09-29): we don't use torso_orientation_rpy_command in the policy due
            # to output dim=32 constraints in the pretrained checkpoint, so we set it to 0
            actions["action.torso_orientation_rpy_command"] = 0 * actions["action.navigate_command"]
        # NOTE(xinjieyao, 2025-09-29): assume gr1 tabletop manipulation does not use waist, arms_only

        robot_action_sim = remap_policy_joints_to_sim_joints(
            actions, self.policy_joints_config, self.robot_action_joints_config, self.device
        )

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.from_numpy(actions["action.navigate_command"]).to(self.device),
                    torch.from_numpy(actions["action.base_height_command"]).to(self.device),
                    torch.from_numpy(actions["action.torso_orientation_rpy_command"]).to(self.device),
                ],
                axis=2,
            )
        elif self.task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
            action_tensor = robot_action_sim.get_joints_pos()

        assert action_tensor.shape[1] >= self.num_feedback_actions
        return action_tensor

    def reset(self):
        """Resets the policy's internal state."""
        # As GR00T is a single-shot policy, we don't need to reset its internal state
        # Only reset the action chunking mechanism
        self.policy_iter = self.create_trajectory_iterator(self.trajectory_index)
        self.current_action_chunk = None
        self.current_action_index = 0

    def create_trajectory_iterator(self, trajectory_index: int = 0) -> Iterator[int]:
        """Create an iterator starting from a specific trajectory index."""
        num_trajectories = len(self.policy.trajectory_lengths)
        if trajectory_index >= num_trajectories:
            raise ValueError(f"Trajectory index {trajectory_index} exceeds available trajectories {num_trajectories}")

        # absolute starting step index
        start_step = sum(self.policy.trajectory_lengths[:trajectory_index])
        # iterator from that step to the end of the dataset
        return iter(range(start_step, len(self.policy)))

    def set_trajectory_index(self, trajectory_index: int):
        """Set the policy to start from a specific trajectory index."""
        num_trajectories = len(self.policy.trajectory_lengths)
        if trajectory_index >= num_trajectories:
            raise ValueError(f"Trajectory index {trajectory_index} exceeds available trajectories {num_trajectories}")

        self.trajectory_index = trajectory_index
        # iterator to start from specified trajectory
        self.policy_iter = iter(range(trajectory_index, num_trajectories))
        self.current_action_chunk = None
        self.current_action_index = 0

    def get_trajectory_index(self) -> int:
        return self.trajectory_index
