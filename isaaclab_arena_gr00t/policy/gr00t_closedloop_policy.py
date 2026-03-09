# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Prepend GR00T deps when loaded without re-exec (e.g. eval_runner, tests before conftest re-exec).
_GROOT_DEPS_DIR = os.environ.get("GROOT_DEPS_DIR")
if _GROOT_DEPS_DIR and _GROOT_DEPS_DIR not in sys.path:
    sys.path.insert(0, _GROOT_DEPS_DIR)

import gymnasium as gym
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from isaaclab_arena.policy.action_chunking import ActionChunkingState
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena_gr00t.utils.eagle_config_compat import apply_eagle_config_compat
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
    load_gr00t_policy_from_config,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file


@dataclass
class Gr00tClosedloopPolicyArgs(Gr00tBasePolicyArgs):
    """
    Configuration dataclass for Gr00tClosedloopPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyArgs,
    and adds num_envs for local simulation.
    """

    num_envs: int = field(
        default=1,
        metadata={
            "help": "Number of environments to simulate",
        },
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tClosedloopPolicyArgs:
        """Create configuration from parsed CLI arguments."""
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
        )


class Gr00tClosedloopPolicy(PolicyBase):

    name = "gr00t_closedloop"
    config_class = Gr00tClosedloopPolicyArgs

    def __init__(self, config: Gr00tClosedloopPolicyArgs):
        """Initialize Gr00tClosedloopPolicy from a configuration dataclass."""
        super().__init__(config)

        # Config / policy
        self.policy_config: Gr00tClosedloopPolicyConfig = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig
        )
        self.policy: Gr00tPolicy = load_gr00t_policy_from_config(self.policy_config)

        # Basic attributes
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        world_size = get_world_size()
        if world_size > 1 and "cuda" in config.policy_device:
            local_rank = get_local_rank()
            self.device = f"cuda:{local_rank}"

        # Joint configs
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        # Modality config
        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length

        # Shared chunking state (unified with remote ActionChunkingClientSidePolicy)
        self._chunking_state = ActionChunkingState(
            num_envs=self.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.policy_config.action_horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=torch.float,
        )

        # task description of task being evaluated. It will be set by the task being evaluated.
        self.task_description: str | None = None

    # ---------------------- CLI helpers (local) -------------------

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tClosedloopPolicy:
        """Create a Gr00tClosedloopPolicy instance from parsed CLI arguments."""
        config = Gr00tClosedloopPolicyArgs.from_cli_args(args)
        return Gr00tClosedloopPolicy(config)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add gr00t closedloop policy specific arguments to the parser."""
        group = parser.add_argument_group("Gr00t Closedloop Policy", "Arguments for gr00t closedloop policy")
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the Gr00t closedloop policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device to use for the policy-related operations (default: cuda)",
        )
        return parser

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

        apply_eagle_config_compat()

        return Gr00tPolicy(
            model_path=self.policy_config.model_path,
            embodiment_tag=EmbodimentTag[self.policy_config.embodiment_tag],
            device=self.device,
            strict=True,
        )

    def set_task_description(self, task_description: str | None) -> str:
        """Set the language instruction of the task being evaluated."""
        if task_description is None:
            task_description = self.policy_config.language_instruction
        self.task_description = task_description
        return self.task_description

    def get_observations(
        self, observation: dict[str, Any], camera_names: list[str] = ["robot_head_cam_rgb"]
    ) -> dict[str, Any]:
        """Adapter: torch env observation -> numpy GR00T policy inputs.

        Uses ``extract_obs_numpy_from_torch`` as the single explicit
        torch-to-numpy conversion boundary, then delegates to the shared
        core preprocessing.
        """
        assert self.task_description is not None, "Task description is not set"

        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_torch(nested_obs=observation, camera_names=camera_names)

        return build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Get the immediate next action from the current action chunk.

        If the action chunk is not yet computed, compute a new action chunk first
        before returning the action.
        """

        def fetch_chunk() -> torch.Tensor:
            return self.get_action_chunk(observation, self.policy_config.pov_cam_name_sim)

        return self._chunking_state.get_action(fetch_chunk)

    def get_action_chunk(
        self, observation: dict[str, Any], camera_names: list[str] | str = "robot_head_cam_rgb"
    ) -> torch.Tensor:
        """Get a sequence of multiple future low-level actions.

        Returns:
            action_chunk: Shape (num_envs, action_chunk_length, action_dim).
        """
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        policy_observations = self.get_observations(observation, camera_names)
        robot_action_policy, _ = self.policy.get_action(policy_observations)

        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the action chunking mechanism."""
        if env_ids is None:
            env_ids = slice(None)
        # placeholder for future reset options from GR00T repo
        self.policy.reset()
        self._chunking_state.reset(env_ids)
