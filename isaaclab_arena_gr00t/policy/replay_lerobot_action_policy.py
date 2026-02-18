# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gr00t.policy.replay_policy import ReplayPolicy

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.lerobot_replay_action_policy_config import (
    LerobotReplayActionPolicyConfig,
    TaskMode,
)
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    load_gr00t_modality_config_from_file,
    load_robot_joints_config_from_yaml,
)
from isaaclab_arena_gr00t.utils.joints_conversion import remap_policy_joints_to_sim_joints


@dataclass
class ReplayLerobotActionPolicyArgs:
    """
    Configuration dataclass for ReplayLerobotActionPolicy.

    This dataclass serves as the single source of truth for policy configuration,
    supporting both dict-based (from JSON) and CLI-based configuration paths.

    Field metadata is used to auto-generate argparse arguments, ensuring consistency
    between the dataclass definition and CLI argument parsing.
    """

    policy_config_yaml_path: str = field(
        metadata={
            "help": "Path to the Lerobot action policy config YAML file",
            "required": True,
            "arg_name": "config_yaml_path",  # Override argparse name
        }
    )

    device: str = field(
        default="cuda",
        metadata={
            "help": "Device to use for the policy-related operations",
        },
    )

    num_envs: int = field(
        default=1,
        metadata={
            "help": "Number of environments to simulate",
        },
    )

    trajectory_index: int = field(
        default=0,
        metadata={
            "help": "Index of the trajectory to run the policy for",
        },
    )

    max_steps: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of steps to run the policy for",
        },
    )

    # from_dict() is not needed - can use ReplayLerobotActionPolicyArgs(**dict) directly
    # or use ReplayLerobotActionPolicy.from_dict() which is inherited from PolicyBase

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ReplayLerobotActionPolicyArgs":
        """
        Create configuration from parsed CLI arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            ReplayLerobotActionPolicyArgs instance
        """
        return cls(
            policy_config_yaml_path=args.config_yaml_path,
            device=getattr(args, "device", "cuda"),
            num_envs=args.num_envs,
            trajectory_index=args.trajectory_index,
            max_steps=args.max_steps,
        )


class ReplayLerobotActionPolicy(PolicyBase):

    name = "replay_lerobot"
    # enable from_dict() from policy_base.PolicyBase
    config_class = ReplayLerobotActionPolicyArgs

    def __init__(self, config: ReplayLerobotActionPolicyArgs):
        """
        Initialize ReplayLerobotActionPolicy from a configuration dataclass.

        Args:
            config: ReplayLerobotActionPolicyArgs configuration dataclass
        """
        super().__init__(config)
        self.policy_config = create_config_from_yaml(config.policy_config_yaml_path, LerobotReplayActionPolicyConfig)
        self.policy = self.load_policy(self.policy_config)
        # Start from the trajectory_index trajectory in the dataset
        self.trajectory_index = config.trajectory_index
        assert (
            self.policy.num_episodes > config.trajectory_index
        ), f"Trajectory index {config.trajectory_index} exceeds available trajectories {self.policy.num_episodes}"
        self.policy.reset(options={"episode_index": config.trajectory_index})

        # determine rollout how many action prediction per observation
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.current_action_index = 0
        self.current_action_chunk = None
        self.num_envs = config.num_envs
        self.device = config.device
        self.max_steps = config.max_steps
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        self.policy_joints_config = self.load_policy_joints_config(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = self.load_sim_joints_config(self.policy_config.action_joints_config_path)

    def load_policy_joints_config(self, policy_config_path: Path) -> dict[str, Any]:
        """Load the policy joint config from the data config."""
        return load_robot_joints_config_from_yaml(policy_config_path)

    def load_sim_joints_config(self, action_config_path: Path) -> dict[str, Any]:
        """Load the simulation joint config from the data config."""
        return load_robot_joints_config_from_yaml(action_config_path)

    def load_policy(self, policy_config: LerobotReplayActionPolicyConfig) -> ReplayPolicy:
        """Load the dataset, whose iterator will be used as the policy."""
        assert Path(policy_config.dataset_path).exists(), f"Dataset path {policy_config.dataset_path} does not exist"

        modality_configs = load_gr00t_modality_config_from_file(
            modality_config_path=policy_config.modality_config_path, embodiment_tag=policy_config.embodiment_tag
        )

        return ReplayPolicy(
            dataset_path=policy_config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=policy_config.action_horizon,
            video_backend=policy_config.video_backend,
            # by pass obs and action validation
            strict=False,
        )

    def get_trajectory_length(self, trajectory_index: int) -> int:
        """Get the number of frames in one trajectory in the dataset."""
        assert self.policy.episode_length is not None
        assert trajectory_index < self.policy.episode_length
        return self.policy.episode_length

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return action from the dataset."""
        # get new predictions and return the first action from the chunk
        if self.current_action_chunk is None and self.current_action_index == 0:
            self.current_action_chunk = self.get_action_chunk()
            assert self.current_action_chunk.shape[1] >= self.action_chunk_length

        assert self.current_action_chunk is not None
        assert self.current_action_index < self.action_chunk_length

        action = self.current_action_chunk[:, self.current_action_index]
        assert action.shape == env.action_space.shape, f"{action.shape=} != {env.action_space.shape=}"

        self.current_action_index += 1
        # reset to empty action chunk
        if self.current_action_index == self.action_chunk_length:
            self.current_action_chunk = None
            self.current_action_index = 0
        return action

    def get_action_chunk(self) -> torch.Tensor:
        """Get action_horizon number of actions, as an action chunk, from the dataset"""

        data_point, info = self.policy.get_action(observation=None, options={"batch_size": self.num_envs})
        # Support MultiEnv running
        actions = {
            "left_arm": np.tile(np.array(data_point["left_arm"]), (self.num_envs, 1, 1)),
            "right_arm": np.tile(np.array(data_point["right_arm"]), (self.num_envs, 1, 1)),
            "left_hand": np.tile(np.array(data_point["left_hand"]), (self.num_envs, 1, 1)),
            "right_hand": np.tile(np.array(data_point["right_hand"]), (self.num_envs, 1, 1)),
        }

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            # additional data for WBC interface
            actions["base_height_command"] = np.tile(np.array(data_point["base_height_command"]), (self.num_envs, 1, 1))
            actions["navigate_command"] = np.tile(np.array(data_point["navigate_command"]), (self.num_envs, 1, 1))
            # NOTE(xinjieyao, 2025-09-29): we don't use torso_orientation_rpy_command in the policy due
            # to output dim=32 constraints in the pretrained checkpoint, so we set it to 0
            actions["torso_orientation_rpy_command"] = 0 * actions["navigate_command"]
        # NOTE(xinjieyao, 2025-09-29): assume gr1 tabletop manipulation does not use waist, arms_only

        robot_action_sim = remap_policy_joints_to_sim_joints(
            actions,
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.from_numpy(actions["navigate_command"]).to(self.device),
                    torch.from_numpy(actions["base_height_command"]).to(self.device),
                    torch.from_numpy(actions["torso_orientation_rpy_command"]).to(self.device),
                ],
                axis=2,
            )
        elif self.task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
            action_tensor = robot_action_sim.get_joints_pos()
        else:
            raise ValueError(f"Unsupported task mode: {self.task_mode}")

        assert action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def reset(self, env_ids: torch.Tensor | None = None, trajectory_index: int = 0):
        """Resets the policy's internal state."""
        # As GR00T is a single-shot policy, we don't need to reset its internal state
        # Only reset the action chunking mechanism
        self.policy.reset(options={"episode_index": trajectory_index})
        self.current_action_chunk = None
        self.current_action_index = 0

    def get_trajectory_index(self) -> int:
        return self.trajectory_index

    def has_length(self) -> bool:
        """Check if the policy is based on a recording (i.e. is a dataset-driven policy)."""
        return True

    def length(self) -> int:
        """Get the length of the policy (for dataset-driven policies)."""
        if self.max_steps is not None:
            return self.max_steps
        else:
            return self.get_trajectory_length(self.get_trajectory_index())

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add replay Lerobot action policy specific arguments to the parser."""
        replay_lerobot_group = parser.add_argument_group(
            "Replay Lerobot Action Policy", "Arguments for replay Lerobot dataset action policy"
        )
        replay_lerobot_group.add_argument(
            "--config_yaml_path",
            type=str,
            required=True,
            help="Path to the Lerobot action policy config YAML file",
        )
        replay_lerobot_group.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="Maximum number of steps to run the policy for.",
        )
        replay_lerobot_group.add_argument(
            "--trajectory_index",
            type=int,
            default=0,
            help="Index of the trajectory to run the policy for (default: 0)",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ReplayLerobotActionPolicy":
        """
        Create a ReplayLerobotActionPolicy instance from parsed CLI arguments.

        Path: CLI args → ConfigDataclass → init cls

        Args:
            args: Parsed command line arguments

        Returns:
            ReplayLerobotActionPolicy instance
        """
        config = ReplayLerobotActionPolicyArgs.from_cli_args(args)
        return ReplayLerobotActionPolicy(config)
