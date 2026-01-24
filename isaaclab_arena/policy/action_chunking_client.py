# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gymnasium as gym
import torch
from typing import Any

from isaaclab_arena.policy.client_side_policy import ClientSidePolicy
from isaaclab_arena.remote_policy.action_protocol import ChunkingActionProtocol
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class ActionChunkingClientSidePolicy(ClientSidePolicy):
    """Client-side policy that consumes fixed-length action chunks sequentially."""

    def __init__(
        self,
        config: Any,
        num_envs: int,
        device: str,
        remote_config: RemotePolicyConfig,
    ) -> None:
        super().__init__(config=config, remote_config=remote_config, protocol_cls=ChunkingActionProtocol)

        self._num_envs = num_envs
        self._device = device

        self._current_action_chunk = torch.zeros(
            self._num_envs,
            self.protocol.action_chunk_length,
            self.protocol.action_dim,
            dtype=torch.float32,
            device=self._device,
        )
        self._current_action_index = torch.full(
            (self._num_envs,),
            fill_value=-1,
            dtype=torch.int32,
            device=self._device,
        )
        self._env_requires_new_chunk = torch.ones(
            self._num_envs,
            dtype=torch.bool,
            device=self._device,
        )

        self.task_description: str | None = None

    # ---------------------- CLI ----------------------------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add CLI arguments for ActionChunkingClientSidePolicy."""
        # Shared remote policy args.
        parser = ClientSidePolicy.add_remote_args_to_parser(parser)

        # Policy-specific args.
        group = parser.add_argument_group(
            "Action Chunking Client Policy",
            "Arguments for client-side action chunking policy.",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device to use for the policy-related operations.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> ActionChunkingClientSidePolicy:
        """Create an ActionChunkingClientSidePolicy from CLI arguments."""
        remote_config = ClientSidePolicy.build_remote_config_from_args(args)
        return ActionChunkingClientSidePolicy(
            config=None,
            num_envs=args.num_envs,
            device=args.policy_device,
            remote_config=remote_config,
        )

    # ---------------------- Task description ----------------------------

    def set_task_description(self, task_description: str | None) -> str:
        """Set the task description on both client-side and remote policy."""
        self.task_description = task_description
        if task_description is not None:
            self.remote_client.call_endpoint(
                "set_task_description",
                data={"task_description": task_description},
                requires_input=True,
            )
        return self.task_description or ""

    # ---------------------- Chunking logic ------------------------------

    def _request_new_chunk(
        self,
        observation: dict[str, Any],
    ) -> torch.Tensor:
        """Request a new action chunk from the remote policy and validate it."""
        protocol = self.protocol
        packed_obs = self.pack_observation_for_server(observation)

        resp = self.remote_client.get_action(packed_obs)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_action, got {type(resp)!r}")
        if "action" not in resp:
            raise KeyError("Remote response does not contain key 'action' for ActionChunkingClientSidePolicy.")

        raw_chunk = resp["action"]
        if not isinstance(raw_chunk, torch.Tensor):
            raw_chunk = torch.tensor(raw_chunk, dtype=torch.float32, device=self._device)
        else:
            raw_chunk = raw_chunk.to(self._device, dtype=torch.float32)

        if raw_chunk.shape[0] != self._num_envs:
            raise ValueError(f"Expected batch size {self._num_envs}, got {raw_chunk.shape[0]}")
        if raw_chunk.shape[1] != protocol.action_chunk_length:
            raise ValueError(
                f"Expected at least {protocol.action_chunk_length} actions per chunk, got {raw_chunk.shape[1]}"
            )
        if raw_chunk.shape[2] != protocol.action_dim:
            raise ValueError(f"Expected action_dim {protocol.action_dim}, got {raw_chunk.shape[2]}")

        return raw_chunk

    def get_action(
        self,
        env: gym.Env,
        observation: gym.spaces.Dict,
    ) -> torch.Tensor:
        """Return one action per env step, consuming action chunks sequentially."""
        protocol = self.protocol

        if bool(self._env_requires_new_chunk.any()):
            new_chunk = self._request_new_chunk(observation)
            mask = self._env_requires_new_chunk

            self._current_action_chunk[mask] = new_chunk[mask]
            self._current_action_index[mask] = 0
            self._env_requires_new_chunk[mask] = False

        idx = self._current_action_index  # [N]
        batch_idx = torch.arange(self._num_envs, device=self._device)

        action = self._current_action_chunk[batch_idx, idx]

        if action.shape != (self._num_envs, protocol.action_dim):
            raise RuntimeError(
                f"Unexpected action shape {action.shape}, expected {(self._num_envs, protocol.action_dim)}"
            )

        self._current_action_index += 1
        self._env_requires_new_chunk = self._current_action_index >= protocol.action_chunk_length

        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset client-side chunking state and remote policy state."""
        if env_ids is None:
            env_ids = torch.arange(
                self._num_envs,
                device=self._device,
                dtype=torch.long,
            )

        self._current_action_chunk[env_ids] = 0.0
        self._current_action_index[env_ids] = -1
        self._env_requires_new_chunk[env_ids] = True

        # Reset remote state via ClientSidePolicy.
        super().reset(env_ids=env_ids)
