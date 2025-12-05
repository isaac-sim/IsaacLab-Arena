# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from pathlib import Path
from typing import Any

from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config
from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
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
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.num_envs = num_envs
        self.device = device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        self.policy_joints_config = self.load_policy_joints_config(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = self.load_sim_action_joints_config(
            self.policy_config.action_joints_config_path
        )
        self.robot_state_joints_config = self.load_sim_state_joints_config(self.policy_config.state_joints_config_path)

        self.action_dim = len(self.robot_action_joints_config)
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            # GR00T outputs are used for WBC inputs dim. So adding WBC commands to the action dim.
            # WBC commands: navigate_command, base_height_command, torso_orientation_rpy_command
            self.action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD

        self.current_action_chunk = torch.zeros(
            (num_envs, self.policy_config.action_horizon, self.action_dim),
            dtype=torch.float,
            device=device,
        )
        # Use a bool list toindicate that the action chunk is not yet computed for each env
        # True means the action chunk is not yet computed, False means the action chunk is valid
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)

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
        """Get the the immediate next action from the current action chunk.
        If the action chunk is not yet computed, compute a new action chunk first before returning the action.

        Returns:
            action: The immediate next action to execute per env.step() call. Shape: (num_envs, action_dim)
        """
        # get action chunk if not yet computed
        if any(self.env_requires_new_action_chunk):
            # compute a new action chunk for the envs that require a new action chunk
            returned_action_chunk = self.get_action_chunk(observation, self.policy_config.pov_cam_name_sim)
            self.current_action_chunk[self.env_requires_new_action_chunk] = returned_action_chunk[
                self.env_requires_new_action_chunk
            ]
            # reset the action index for those env_ids
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            # reset the env_requires_new_action_chunk for those env_ids
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        # assert for all env_ids that the action index is valid

        assert self.current_action_index.min() >= 0, "At least one env's action index is less than 0"
        assert (
            self.current_action_index.max() < self.action_chunk_length
        ), "At least one env's action index is greater than the action chunk length"
        # for i-th row in action_chunk, use the value of i-th element in current_action_index to select the action from the action chunk
        action = self.current_action_chunk[torch.arange(self.num_envs), self.current_action_index]
        assert action.shape == (
            self.num_envs,
            self.action_dim,
        ), f"{action.shape=} != ({self.num_envs}, {self.action_dim})"

        self.current_action_index += 1

        # for those rows in current_action_chunk that equal to action_chunk_length, reset to o
        reset_env_ids = self.current_action_index == self.action_chunk_length
        self.current_action_chunk[reset_env_ids] = 0.0
        # indicate that the action chunk is not yet computed for those env_ids
        self.env_requires_new_action_chunk[reset_env_ids] = True
        # set the action index for those env_ids to -1 to indicate that the action chunk is reset
        self.current_action_index[reset_env_ids] = -1
        return action

    def get_action_chunk(self, observation: dict[str, Any], camera_name: str = "robot_head_cam_rgb") -> torch.Tensor:
        """Get a sequence of multiple future low-level actions that the policy predicts and outputs
        in a single forward pass, given the current observation and language instruction.

        Returns:
            action_chunk: a sequence of multiple future low-level actions.
            Shape: (num_envs, action_chunk_length, self.action_dim)
        """
        policy_observations = self.get_observations(observation, camera_name)
        robot_action_policy = self.policy.get_action(policy_observations)
        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.policy_joints_config, self.robot_action_joints_config, self.device
        )

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            # NOTE(xinjieyao, 2025-09-29): GR00T output dim=32, does not fit the entire action space,
            # including torso_orientation_rpy_command. Manually set it to 0.
            torso_orientation_rpy_command = torch.zeros(
                robot_action_policy["action.navigate_command"].shape, dtype=torch.float, device=self.device
            )
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.tensor(robot_action_policy["action.navigate_command"], dtype=torch.float, device=self.device),
                    torch.tensor(
                        robot_action_policy["action.base_height_command"], dtype=torch.float, device=self.device
                    ),
                    torso_orientation_rpy_command,
                ],
                axis=2,
            )
        elif self.task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
            action_tensor = robot_action_sim.get_joints_pos()

        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Resets the action chunking mechanism. As GR00T policy predicts a sequence of future
        low-level actions in a single forward pass, we don't need to reset its internal state.
        It zeros the action chunk, sets the action index to -1, and sets the
        boolean indicator env_requires_new_action_chunk to True for the required env_ids.

        Args:
            env_ids: the env_ids to reset. If None, reset all envs.
        """
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True
