# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_inputs_np,
    compute_action_dim,
    load_gr00t_joint_configs,
    load_gr00t_policy_from_config,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml


@dataclass
class Gr00tClosedloopPolicyArgs(Gr00tBasePolicyArgs):
    """
    Configuration dataclass for Gr00tClosedloopPolicy.
    """

    num_envs: int = field(
        default=1,
        metadata={
            "help": "Number of environments to simulate",
        },
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Gr00tClosedloopPolicyArgs":
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
        )


class Gr00tClosedloopPolicy(PolicyBase):

    name = "gr00t_closedloop"
    # enable from_dict() from PolicyBase
    config_class = Gr00tClosedloopPolicyArgs

    def __init__(self, config: Gr00tClosedloopPolicyArgs):
        """
        Initialize Gr00tClosedloopPolicy from a configuration dataclass.

        Args:
            config: Gr00tClosedloopPolicyArgs configuration dataclass
        """
        super().__init__(config)
        self.policy_config = create_config_from_yaml(config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig)

        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configurations
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length

        self.policy: Gr00tPolicy = load_gr00t_policy_from_config(self.policy_config)

        # Chunking state (local-only logic)
        self.current_action_chunk = torch.zeros(
            (self.num_envs, self.policy_config.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # True means the action chunk is not yet computed, False means the action chunk is valid.
        self.env_requires_new_action_chunk = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-env index into the current action chunk
        self.current_action_index = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Task description of the task being evaluated. It will be set externally.
        self.task_description: str | None = None

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Gr00tClosedloopPolicy":
        """
        Create a Gr00tClosedloopPolicy instance from parsed CLI arguments.

        Path: CLI args → ConfigDataclass → init cls

        Args:
            args: Parsed command line arguments

        Returns:
            Gr00tClosedloopPolicy instance
        """
        config = Gr00tClosedloopPolicyArgs.from_cli_args(args)
        return Gr00tClosedloopPolicy(config)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add gr00t closedloop policy specific arguments to the parser."""
        gr00t_closedloop_group = parser.add_argument_group(
            "Gr00t Closedloop Policy", "Arguments for gr00t closedloop policy"
        )
        gr00t_closedloop_group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the Gr00t closedloop policy config YAML file",
        )
        gr00t_closedloop_group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device to use for the policy-related operations (default: cuda)",
        )
        return parser

    def set_task_description(self, task_description: str | None) -> str:
        """Set the language instruction of the task being evaluated."""
        if task_description is None:
            task_description = self.policy_config.language_instruction
        self.task_description = task_description
        return self.task_description

    def get_observations(self, observation: dict[str, Any], camera_name: str = "robot_head_cam_rgb") -> dict[str, Any]:
        """Adapter: torch env observation -> numpy GR00T policy inputs.

        The core GR00T logic in gr00t_core.py works on numpy. This method:
          - extracts torch tensors from the environment observation,
          - moves them to CPU and converts to numpy,
          - uses the shared numpy-based preprocessing,
          - returns a numpy dict suitable for Gr00tPolicy.
        """
        assert self.task_description is not None, "Task description is not set"

        # Extract torch tensors from observation
        rgb_t: torch.Tensor = observation["camera_obs"][camera_name]
        joint_pos_sim_t: torch.Tensor = observation["policy"]["robot_joint_pos"]

        # Convert to numpy for core logic
        rgb_np: np.ndarray = rgb_t.detach().cpu().numpy()
        joint_pos_sim_np: np.ndarray = joint_pos_sim_t.detach().cpu().numpy()

        # Use shared numpy-based preprocessing
        policy_obs_np = build_gr00t_policy_inputs_np(
            rgb_np=rgb_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
        )
        return policy_obs_np

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
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
        )

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
