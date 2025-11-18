# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional

from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config
from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.data_utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.data_utils.io_utils import create_config_from_yaml, load_robot_joints_config_from_yaml
from isaaclab_arena_gr00t.data_utils.joints_conversion import (
    remap_policy_joints_to_sim_joints,
    remap_sim_joints_to_policy_joints,
)
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition
from isaaclab_arena_gr00t.policy_config import Gr00tClosedloopPolicyConfig, TaskMode


class Gr00tClosedloopPolicy(PolicyBase):
    def __init__(self, policy_config_yaml_path: Path, num_envs: int = 1, device: str = "cuda"):
        """
        Base class for closedloop inference from obs using GR00T N1.5 policy
        """
        self.policy_config = create_config_from_yaml(policy_config_yaml_path, Gr00tClosedloopPolicyConfig)
        self.policy = self.load_policy()

        # determine rollout how many action prediction per observation
        self.num_feedback_actions = self.policy_config.num_feedback_actions
        self.num_envs = num_envs
        self.device = device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        self.policy_joints_config = self.load_policy_joints_config(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = self.load_sim_action_joints_config(
            self.policy_config.action_joints_config_path
        )
        self.robot_state_joints_config = self.load_sim_state_joints_config(self.policy_config.state_joints_config_path)
        # initialize to NaN to indicate that the action chunk is not yet computed
        self.current_action_chunk = torch.ones((num_envs, self.num_feedback_actions, len(self.robot_action_joints_config)), dtype=torch.float32, device=device) * float('nan')
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def load_policy_joints_config(self, policy_config_path: Path) -> dict[str, Any]:
        """Load the GR00T policy joint config from the data config."""
        return load_robot_joints_config_from_yaml(policy_config_path)

    def load_sim_state_joints_config(self, state_config_path: Path) -> dict[str, Any]:
        """Load the simulation state joint config from the data config."""
        return load_robot_joints_config_from_yaml(state_config_path)

    def load_sim_action_joints_config(self, action_config_path: Path) -> dict[str, Any]:
        """Load the simulation action joint config from the data config."""
        return load_robot_joints_config_from_yaml(action_config_path)

    def load_policy(self) -> Gr00tPolicy:
        """Load the dataset, whose iterator will be used as the policy."""
        assert Path(
            self.policy_config.model_path
        ).exists(), f"Dataset path {self.policy_config.dataset_path} does not exist"

        # Use the same data preprocessor specified in the  data config map
        if self.policy_config.data_config in DATA_CONFIG_MAP:
            self.data_config = DATA_CONFIG_MAP[self.policy_config.data_config]
        elif self.policy_config.data_config == "unitree_g1_sim_wbc":
            self.data_config = load_data_config("isaaclab_arena_gr00t.data_config:UnitreeG1SimWBCDataConfig")
        else:
            raise ValueError(f"Invalid data config: {self.policy_config.data_config}")

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        return Gr00tPolicy(
            model_path=self.policy_config.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.policy_config.embodiment_tag,
            denoising_steps=self.policy_config.denoising_steps,
            device=self.policy_config.policy_device,
        )

    def get_observations(self, observation: dict[str, Any], camera_name: str = "robot_head_cam_rgb") -> dict[str, Any]:
        rgb = observation["camera_obs"][camera_name]
        # gr00t uses numpy arrays
        rgb = rgb.cpu().numpy()
        # Apply preprocessing to rgb if size is not the same as the target size
        if rgb.shape[1:3] != self.policy_config.target_image_size[:2]:
            rgb = resize_frames_with_padding(
                rgb, target_image_size=self.policy_config.target_image_size, bgr_conversion=False, pad_img=True
            )
        # GR00T uses np arrays, needs to copy torch tensor from gpu to cpu before conversion
        joint_pos_sim = observation["policy"]["robot_joint_pos"].cpu()
        joint_pos_state_sim = JointsAbsPosition(joint_pos_sim, self.robot_state_joints_config)
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        joint_pos_state_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, self.policy_joints_config)

        # Pack inputs to dictionary and run the inference
        policy_observations = {
            "annotation.human.task_description": [self.policy_config.language_instruction] * self.num_envs,
            "video.ego_view": rgb.reshape(
                self.num_envs,
                1,
                self.policy_config.target_image_size[0],
                self.policy_config.target_image_size[1],
                self.policy_config.target_image_size[2],
            ),
            "state.left_arm": joint_pos_state_policy["left_arm"].reshape(self.num_envs, 1, -1),
            "state.right_arm": joint_pos_state_policy["right_arm"].reshape(self.num_envs, 1, -1),
            "state.left_hand": joint_pos_state_policy["left_hand"].reshape(self.num_envs, 1, -1),
            "state.right_hand": joint_pos_state_policy["right_hand"].reshape(self.num_envs, 1, -1),
        }
        # NOTE(xinjieyao, 2025-10-07): waist is not used in GR1 tabletop manipulation
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            policy_observations["state.waist"] = joint_pos_state_policy["waist"].reshape(self.num_envs, 1, -1)
        return policy_observations

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return action from the dataset."""
        # get new predictions and return the first action from the chunk
        # get action chunk if not yet computed or the action chunk is not valid (all elements are NaN)
        action_chunk_nan_mask = torch.isnan(self.current_action_chunk).any(dim=1).any(dim=1)
        assert action_chunk_nan_mask.shape == (self.num_envs,), f"action_chunk_nan_mask.shape: {action_chunk_nan_mask.shape} != ({self.num_envs},)"
        if action_chunk_nan_mask.any():
            #get thos env_ids that the action chunk is not valid
            invalid_env_ids = torch.where(action_chunk_nan_mask)[0]
            # get the action chunk for the invalid env_ids
            returned_action_chunk = self.get_action_chunk(observation, self.policy_config.pov_cam_name_sim)
            print(f"returned_action_chunk.shape: {returned_action_chunk.shape}")
            print(f"invalid_env_ids: {invalid_env_ids}")
            print(f"self.current_action_chunk.shape: {self.current_action_chunk.shape}")
            print(f"self.current_action_chunk[invalid_env_ids].shape: {self.current_action_chunk[invalid_env_ids].shape}")
            print(f"returned_action_chunk[invalid_env_ids].shape: {returned_action_chunk[invalid_env_ids].shape}")
            self.current_action_chunk[invalid_env_ids] = returned_action_chunk[invalid_env_ids]
            # reset the action index for the invalid env_ids
            self.current_action_index[invalid_env_ids] = 0

        # assert for all env_ids that two conditions are met:
        # 1. the action chunk is not NaN
        # 2. the action index is less than the number of feedback actions
        assert not torch.isnan(self.current_action_chunk).any(dim=1).any(dim=1).any(), "Action chunk is NaN"
        assert self.current_action_index.max() < self.num_feedback_actions, "Action index is greater than the number of feedback actions"
        # for i-th row in action_chunk, use the value iof i-th element from current_action_index to index the action chunk
        action = self.current_action_chunk[torch.arange(self.num_envs), self.current_action_index]
        assert action.shape == env.action_space.shape, f"{action.shape=} != {env.action_space.shape=}"

        self.current_action_index += 1
        # reset to empty action chunk
        # for those rows in current_action_chunk that equal to num_feedback_actions, reset to empty action chunk
        reset_env_ids = self.current_action_index == self.num_feedback_actions
        self.current_action_chunk[reset_env_ids] = 1 * float('nan')
        self.current_action_index[reset_env_ids] = 0
        return action

    def get_action_chunk(self, observation: dict[str, Any], camera_name: str = "robot_head_cam_rgb") -> torch.Tensor:
        policy_observations = self.get_observations(observation, camera_name)
        robot_action_policy = self.policy.get_action(policy_observations)
        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.policy_joints_config, self.robot_action_joints_config, self.device
        )

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            # NOTE(xinjieyao, 2025-09-29): GR00T output dim=32, does not fit the entire action space,
            # including torso_orientation_rpy_command. Manually set it to 0.
            torso_orientation_rpy_command = np.zeros(robot_action_policy["action.navigate_command"].shape)
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.from_numpy(robot_action_policy["action.navigate_command"]).to(self.device),
                    torch.from_numpy(robot_action_policy["action.base_height_command"]).to(self.device),
                    torch.from_numpy(torso_orientation_rpy_command).to(self.device),
                ],
                axis=2,
            )
        elif self.task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
            action_tensor = robot_action_sim.get_joints_pos()
        print(f"action_tensor.shape: {action_tensor.shape}")
        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.num_feedback_actions
        return action_tensor

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Resets the policy's internal state."""
        # As GR00T is a single-shot policy, we don't need to reset its internal state
        # Only reset the action chunking mechanism
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] *= float('nan')
        self.current_action_index[env_ids] = 0
