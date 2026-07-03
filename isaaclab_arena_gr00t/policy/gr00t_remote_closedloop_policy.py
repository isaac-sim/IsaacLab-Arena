# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy using GR00T's native PolicyClient.

This policy connects to a GR00T policy server (launched via
``gr00t/eval/run_gr00t_server.py``) and uses its own observation/action translation pipeline.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import torch
from dataclasses import dataclass
from typing import Any, Literal

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler, ActionScheduler, SyncedBatchActionScheduler
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyCfg,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file


# TODO(xinjieyao, 2026-04-27): Consider adding RemotePolicyCfg and deriving this config from it.
@dataclass
class Gr00tRemoteClosedloopPolicyCfg(Gr00tBasePolicyCfg):
    """Configuration for Gr00tRemoteClosedloopPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyCfg,
    and adds remote server connection parameters and num_envs.
    """

    num_envs: int = 1
    """Number of parallel environments served by the policy."""

    remote_host: str = "localhost"
    """GR00T policy server hostname."""

    remote_port: int = 5555
    """GR00T policy server port."""

    remote_api_token: str | None = None
    """Optional policy-server API token."""

    scheduler: Literal["chunk", "synced_batch"] = "chunk"
    """Action scheduler used to consume inference chunks."""


@register_policy(cfg_type=Gr00tRemoteClosedloopPolicyCfg)
class Gr00tRemoteClosedloopPolicy(PolicyBase[Gr00tRemoteClosedloopPolicyCfg]):
    """GR00T closed-loop policy that delegates inference to a remote GR00T server.

    Uses GR00T's native ``PolicyClient`` (from ``gr00t.policy.server_client``)
    to communicate with a GR00T policy server.
    """

    name = "gr00t_remote_closedloop"

    def __init__(self, config: Gr00tRemoteClosedloopPolicyCfg):
        super().__init__(config)

        action_scheduler_cls: type[ActionScheduler]
        if config.scheduler == "synced_batch":
            action_scheduler_cls = SyncedBatchActionScheduler
        else:
            assert config.scheduler == "chunk", f"Unknown action scheduler: {config.scheduler}"
            action_scheduler_cls = ActionChunkScheduler

        # Policy config (for obs/action translation — no model loading)
        # TODO(xinjieyao, 2026-04-27): to be refactored
        self.policy_config: Gr00tClosedloopPolicyCfg = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyCfg
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (for sim from/to policy joint space remapping)
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length

        self._chunking_state: ActionScheduler | None = action_scheduler_cls(
            num_envs=self.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.policy_config.action_horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=torch.float,
        )

        # Connect to GR00T's native PolicyClient
        client = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        self._client: Gr00tPolicyClient | None = client
        if not client.ping():
            raise ConnectionError(f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}")

        self.task_description: str | None = None

    # TODO(cvolk, 2026-07-03): Remove this deprecated argparse adapter once the CLI builds typed configs directly.

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Gr00t Remote Closedloop Policy",
            "Arguments for GR00T remote closed-loop policy evaluation.",
        )
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
            help="Device for Arena-side tensor operations (default: cuda)",
        )
        group.add_argument("--remote_host", type=str, default="localhost", help="GR00T policy server hostname")
        group.add_argument("--remote_port", type=int, default=5555, help="GR00T policy server port")
        group.add_argument("--remote_api_token", type=str, default=None, help="API token for the policy server")
        group.add_argument(
            "--scheduler",
            type=str,
            default="chunk",
            choices=["chunk", "synced_batch"],
            help=(
                "Action scheduler: 'chunk' fetches a new chunk for any env that needs one;"
                " 'synced_batch' waits until ALL envs need a new chunk and then issues a single"
                " full-batch inference call (envs that finish early hold their current robot state)."
            ),
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicy:
        config = Gr00tRemoteClosedloopPolicyCfg(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_api_token=getattr(args, "remote_api_token", None),
            scheduler=getattr(args, "scheduler", "chunk"),
        )
        return Gr00tRemoteClosedloopPolicy(config)

    # ---------------------- Policy interface -------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the job config, "
                "pass --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self._chunking_state is not None, "GR00T remote policy has been closed"

        def fetch_chunk() -> torch.Tensor:
            return self._get_action_chunk(observation, self.policy_config.pov_cam_name_sim)

        return self._chunking_state.get_action(
            fetch_chunk,
            hold_action=self._extract_hold_action(observation),
        )

    def _extract_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Build the action vector that waiting envs should hold: their current sim joint positions
        copied into the action slots that share a joint name with the state config."""
        joint_pos_sim = observation["policy"]["robot_joint_pos"].to(device=self.device, dtype=torch.float)
        hold_action = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float, device=self.device)
        for joint_name, action_idx in self.robot_action_joints_config.items():
            state_idx = self.robot_state_joints_config.get(joint_name)
            if state_idx is not None:
                hold_action[:, action_idx] = joint_pos_sim[:, state_idx]
        return hold_action

    def _get_action_chunk(
        self, observation: dict[str, Any], camera_names: list[str] | str = "robot_head_cam_rgb"
    ) -> torch.Tensor:
        """Get an action chunk from the remote GR00T server.

        Calls GR00T's PolicyClient to get the action chunk.
        """
        if isinstance(camera_names, str):
            camera_names = [camera_names]

        # 1. Reuse the same obs translation as local policy
        assert self.task_description is not None, "Task description is not set"
        assert self._client is not None, "GR00T remote policy has been closed"
        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_torch(nested_obs=observation, camera_names=camera_names)
        policy_observations = build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

        # 2. Call GR00T's own client
        robot_action_policy, _ = self._client.get_action(policy_observations)

        # 3. Action translation from policy output to sim action tensor
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
        if env_ids is None:
            env_ids = slice(None)
        assert self._client is not None, "GR00T remote policy has been closed"
        assert self._chunking_state is not None, "GR00T remote policy has been closed"
        self._client.reset()
        self._chunking_state.reset(env_ids)

    def close(self) -> None:
        """Release Arena-side resources for the remote GR00T policy client."""
        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                try:
                    if socket is not None:
                        socket.close(linger=0)
                finally:
                    if context is not None:
                        context.term()
        finally:
            self._client = None
            self._chunking_state = None
            self.modality_configs = None
