# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from gr00t.policy.gr00t_policy import Gr00tPolicy

from isaaclab_arena.remote_policy.action_protocol import ChunkingActionProtocol
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_packed,
    load_gr00t_joint_configs,
    load_gr00t_policy_from_config,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file, to_numpy


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
        self.policy_config: Gr00tClosedloopPolicyConfig = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig
        )
        print(
            "[Gr00tRemoteServerSidePolicy] config:\n"
            f"  model_path        = {self.policy_config.model_path}\n"
            f"  embodiment_tag    = {self.policy_config.embodiment_tag}\n"
            f"  task_mode_name    = {self.policy_config.task_mode_name}\n"
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

        # Modality config
        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action dimensions
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.action_horizon = self.policy_config.action_horizon

        # Underlying GR00T policy
        self.policy: Gr00tPolicy = load_gr00t_policy_from_config(self.policy_config)
        print("[Gr00tRemoteServerSidePolicy] Gr00tPolicy loaded successfully")

        # Required observation keys for protocol (one key per camera)
        self.camera_names: list[str] = self.policy_config.pov_cam_name_sim
        self.required_observation_keys: list[str] = [f"camera_obs.{cam}" for cam in self.camera_names] + [
            "policy.robot_joint_pos"
        ]

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
            action_horizon=self.action_horizon,
        )
        print(f"[Gr00tRemoteServerSidePolicy] protocol mode = {proto.mode.value}")
        return proto

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _build_policy_observations(
        self,
        observation: dict[str, Any],
        camera_names: list[str],
    ) -> dict[str, Any]:
        """Convert packed numpy observation into numpy GR00T policy inputs.

        Uses ``extract_obs_numpy_from_packed`` as the single explicit
        data-extraction boundary for the remote pipeline, then delegates
        to the shared core preprocessing.
        """
        assert self._task_description is not None, "Task description is not set"

        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_packed(
            observation, camera_names, self.unpack_observation
        )

        return build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self._task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

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
        # 1) Shared numpy-based preprocessing
        policy_observations = self._build_policy_observations(observation, self.camera_names)

        # 2) GR00T forward pass
        robot_action_policy, _ = self.policy.get_action(policy_observations)

        # 3) Postprocessing (shared with closedloop)
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        assert action_tensor.shape[1] >= self.action_chunk_length

        action_chunk = to_numpy(action_tensor)
        # NOTE(huikang, 2026-02-06):  Currently, it seems that the output action length is action_horizon,
        # but the action chunk post-process actually handles a length of action_chunk_length.
        # It looks like we can transmit a tensor of length action_chunk_length. At the moment, action_chunk_length and action_horizon are the same.
        action: dict[str, Any] = {"action": action_chunk}
        info: dict[str, Any] = {}
        return action, info

    def reset(self, env_ids: list[int] | None = None, reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
        # placeholder for future reset options from GR00T repo
        self.policy.reset()
        return {"status": "reset_success"}
