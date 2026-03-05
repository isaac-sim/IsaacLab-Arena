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
from isaaclab_arena.remote_policy.client_state import ClientState
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
    """Server-side wrapper around Gr00tPolicy.

    v2: Per-client task descriptions are stored in ``ClientState.instructions``
    instead of the global ``self._task_description`` singleton.  The legacy
    attribute is kept for backward compatibility with v1 callers.
    """

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

    def _resolve_task_description(
        self,
        env_ids: list[int] | None,
        client_state: ClientState | None,
        *,
        batch_size: int = 1,
    ) -> str:
        """Get the task description for a request, preferring client_state (v2).

        **Known limitation**: GR00T's ``build_gr00t_policy_observations()``
        accepts a single ``task_description: str`` for the entire batch.
        When multiple envs have different instructions, only env_ids[0]'s
        instruction can be used.  Mixed-instruction requests are rejected
        explicitly instead of silently using the first instruction.  True
        per-env instructions would require one forward pass per env.
        """
        descriptions = self._resolve_task_descriptions(
            env_ids, client_state, batch_size=batch_size
        )
        if len(set(descriptions)) > 1:
            raise NotImplementedError(
                "GR00T does not support mixed instructions within one get_action request. "
                "Split env_ids into homogeneous instruction groups or dispatch individually."
            )
        return descriptions[0]

    def _resolve_task_descriptions(
        self,
        env_ids: list[int] | None,
        client_state: ClientState | None,
        *,
        batch_size: int,
    ) -> list[str]:
        """Resolve per-slot instructions for a request before GR00T batching."""
        if client_state is None:
            fallback = self._task_description or self.policy_config.language_instruction
            return [fallback] * max(batch_size, 1)

        target_env_ids = env_ids if env_ids is not None else list(range(batch_size))
        if not target_env_ids:
            target_env_ids = [0]

        fallback = self._task_description or self.policy_config.language_instruction
        descriptions: list[str] = []
        for env_id in target_env_ids:
            desc = client_state.instructions[env_id]
            descriptions.append(desc if desc is not None else fallback)
        return descriptions

    def _build_policy_observations(
        self,
        observation: dict[str, Any],
        camera_names: list[str],
        task_description: str,
    ) -> dict[str, Any]:
        """Convert packed numpy observation into numpy GR00T policy inputs."""
        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_packed(
            observation, camera_names, self.unpack_observation
        )

        return build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

    # ------------------------------------------------------------------ #
    # ServerSidePolicy interface
    # ------------------------------------------------------------------ #

    def set_task_description(
        self,
        task_description: str | None,
        *,
        env_ids: list[int] | None = None,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        if task_description is None:
            task_description = self.policy_config.language_instruction

        # Delegate to base class (handles v2 client_state path AND
        # v1 fallback with DeprecationWarning consistently).
        return super().set_task_description(
            task_description, env_ids=env_ids, client_state=client_state,
        )

    def get_action(
        self,
        observation: dict[str, Any],
        *,
        env_ids: list[int] | None = None,
        client_state: ClientState | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batch_size = 1
        for value in observation.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                batch_size = int(value.shape[0])
                break
        task_description = self._resolve_task_description(
            env_ids, client_state, batch_size=batch_size
        )

        # 1) Shared numpy-based preprocessing
        policy_observations = self._build_policy_observations(observation, self.camera_names, task_description)

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
        action: dict[str, Any] = {"action": action_chunk}
        info: dict[str, Any] = {}
        return action, info

    def reset(
        self,
        env_ids: list[int] | None = None,
        reset_options: dict[str, Any] | None = None,
        *,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        self.policy.reset()

        # Preserve instructions across resets (matching v1 semantics).
        # In v1, reset() only resets policy state — the task description
        # persists across episodes.  Clearing instructions here would force
        # the client to call set_task_description() again after every reset,
        # which is a behavior regression.

        return {"status": "reset_success"}
