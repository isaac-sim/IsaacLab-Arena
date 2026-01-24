# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any

from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena.remote_policy.action_protocol import ChunkingActionProtocol
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
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
class Gr00tRemotePolicyArgs(Gr00tBasePolicyArgs):
    """Configuration for Gr00tRemoteServerSidePolicy.

    Reuses policy_config_yaml_path and policy_device from the base.
    """

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tRemotePolicyArgs:
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
        )


class Gr00tRemoteServerSidePolicy(ServerSidePolicy):
    """Server-side wrapper around Gr00tPolicy."""

    config_class = Gr00tRemotePolicyArgs

    def __init__(self, config: Gr00tRemotePolicyArgs) -> None:
        super().__init__(config)

        print(f"[Gr00tRemoteServerSidePolicy] loading config from: {config.policy_config_yaml_path}")
        self.policy_config = create_config_from_yaml(config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig)
        print(
            "[Gr00tRemoteServerSidePolicy] config:\n"
            f"  model_path        = {self.policy_config.model_path}\n"
            f"  embodiment_tag    = {self.policy_config.embodiment_tag}\n"
            f"  task_mode_name    = {self.policy_config.task_mode_name}\n"
            f"  data_config       = {self.policy_config.data_config}\n"
            f"  action_horizon    = {self.policy_config.action_horizon}\n"
            f"  action_chunk_len  = {self.policy_config.action_chunk_length}\n"
            f"  pov_cam_name_sim  = {self.policy_config.pov_cam_name_sim}\n"
            f"  policy_device     = {self.policy_config.policy_device}\n"
        )

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

        self.required_observation_keys: list[str] = [
            f"camera_obs.{self.policy_config.pov_cam_name_sim}",
            "policy.robot_joint_pos",
        ]

        # Underlying GR00T policy
        self.policy: Gr00tPolicy = load_gr00t_policy_from_config(self.policy_config)
        print("[Gr00tRemoteServerSidePolicy] Gr00tPolicy loaded successfully")

        # Task description will be set via set_task_description RPC
        self._task_description: str | None = None

    # ---------------------- CLI helpers (server-side) -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add server-side GR00T remote policy arguments."""
        group = parser.add_argument_group(
            "Gr00t Remote Server Policy",
            "Arguments for GR00T-based server-side remote policy.",
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the GR00T closedloop policy config YAML file.",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device to use for server-side GR00T inference (default: cuda).",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tRemoteServerSidePolicy:
        """Create a Gr00tRemoteServerSidePolicy from CLI arguments."""
        config = Gr00tRemotePolicyArgs.from_cli_args(args)
        return Gr00tRemoteServerSidePolicy(config)

    # ------------ protocol ------------

    def _build_protocol(self) -> ChunkingActionProtocol:
        proto = ChunkingActionProtocol(
            action_dim=self.action_dim,
            observation_keys=self.required_observation_keys,
            action_chunk_length=self.action_chunk_length,
        )
        print(f"[Gr00tRemoteServerSidePolicy] protocol mode = {proto.mode.value}")
        return proto

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _build_policy_observations(
        self,
        observation: dict[str, Any],
        camera_name: str,
    ) -> dict[str, Any]:
        """Convert packed numpy observation into numpy GR00T policy inputs.

        The client sends a flat dict of numpy arrays.
        ServerSidePolicy.unpack_observation reconstructs the nested structure:
          - observation["camera_obs"][camera_name] : (N, H, W, C) numpy
          - observation["policy"]["robot_joint_pos"]: (N, num_joints) numpy
        """
        nested_obs = self.unpack_observation(observation)
        rgb_np: np.ndarray = nested_obs["camera_obs"][camera_name]
        joint_pos_sim_np: np.ndarray = nested_obs["policy"]["robot_joint_pos"]

        assert self._task_description is not None, "Task description is not set"

        policy_obs_np = build_gr00t_policy_inputs_np(
            rgb_np=rgb_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self._task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
        )
        return policy_obs_np

    # ------------------------------------------------------------------ #
    # ServerSidePolicy interface
    # ------------------------------------------------------------------ #

    def set_task_description(self, task_description: str | None) -> dict[str, Any]:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        self._task_description = task_description
        return {"status": "ok"}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        camera_name = self.policy_config.pov_cam_name_sim

        # 1) Shared numpy-based preprocessing
        policy_observations = self._build_policy_observations(observation, camera_name)

        # 2) GR00T forward pass
        robot_action_policy = self.policy.get_action(policy_observations)

        # 3) postprocessing
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
        )

        assert action_tensor.shape[1] >= self.action_chunk_length

        action_chunk = action_tensor[:, : self.action_chunk_length, :].cpu().numpy()
        action: dict[str, Any] = {"action": action_chunk}
        info: dict[str, Any] = {}
        return action, info

    def reset(self, env_ids: list[int] | None = None, reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
        # GR00T policy is stateless for this closed-loop usage; nothing to reset
        return {"status": "reset_success"}
