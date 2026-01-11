# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import gymnasium as gym
import torch

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig

def _get_nested(obs: Dict[str, Any], key_path: str) -> Any:
    cur = obs
    for k in key_path.split("."):
        cur = cur[k]
    return cur

def _pack_for_server(
    observation: Dict[str, Any],
    key_paths: List[str],
) -> Dict[str, Any]:
    packed: Dict[str, Any] = {}
    for key_path in key_paths:
        value = _get_nested(observation, key_path)
        # torch.Tensor -> numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        packed[key_path] = value
    return packed

class ActionChunkingClientPolicy(PolicyBase):
    """Client-side policy that requests action chunks from a remote server and
    exposes a step-by-step action interface to the environment.
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        remote_config: Optional[RemotePolicyConfig] = None,
    ) -> None:
        super().__init__(remote_config=remote_config)

        self.num_envs = num_envs
        self.device = device

        if not self.is_remote:
            raise ValueError("ActionChunkingClientPolicy is intended for remote deployment (remote_config must be set).")

        if not self.remote_client.ping():
            cfg = self.remote_config
            raise RuntimeError(
                f"Failed to connect to remote policy server at "
                f"{cfg.host}:{cfg.port}."
            )


        init_info = self.remote_client.get_init_info()
        self.action_dim = init_info.action_dim
        self.action_chunk_length = init_info.action_chunk_length
        self.observation_keys = init_info.observation_keys

        self.current_action_chunk = torch.zeros(
            (self.num_envs, self.action_chunk_length, self.action_dim),
            dtype=torch.float,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.current_action_index = torch.zeros(
            self.num_envs, dtype=torch.int32, device=device
        )

        self.task_description: Optional[str] = None

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Action Chunking Client Policy",
            "Arguments for client-side action chunking policy",
        )

        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device to use for the policy-related operations",
        )
        return parser
    
    @staticmethod
    def from_args(args: argparse.Namespace) -> "ActionChunkingClientPolicy":
        if args.remote_host is None:
            raise ValueError(
                "ActionChunkingClientPolicy requires a remote policy server. "
                "Please set --remote_host (and optionally --remote_port, "
                "--remote_api_token, --remote_timeout_ms)."
            )
    
        remote_config = RemotePolicyConfig(
            host=args.remote_host,
            port=args.remote_port,
            api_token=args.remote_api_token,
            timeout_ms=args.remote_timeout_ms,
        )
    
        return ActionChunkingClientPolicy(
            num_envs=args.num_envs,
            device=args.policy_device,
            remote_config=remote_config,
        )

    def set_task_description(self, task_description: Optional[str]) -> str:
        if task_description is None:
            task_description = ""
        self.task_description = task_description
           
        self.remote_client.call_endpoint(
            "set_task_description",
            data={"task_description": self.task_description},
            requires_input=True,
        )
        return self.task_description

    def _request_new_action_chunk(
        self, observation: Dict[str, Any]
    ) -> torch.Tensor:
        packed_obs = _pack_for_server(observation, self.observation_keys)
        chunk = self.remote_client.get_action(packed_obs)
        # TODO(HuiKang): Currently assumes a simple 'action_chunk' payload.
        # In the future this can be extended to support richer action items
        # (e.g. per-env metadata, hierarchical actions, or multiple action heads).
        action_chunk = chunk["action"]["action_chunk"]
        info = chunk["info"]

        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.to(self.device, dtype=torch.float)
        else:
            action_chunk = torch.tensor(action_chunk, dtype=torch.float, device=self.device)

        if action_chunk.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected action_chunk batch size {self.num_envs}, "
                f"got {action_chunk.shape[0]}"
            )
        if action_chunk.shape[1] < self.action_chunk_length:
            raise ValueError(
                f"Expected at least {self.action_chunk_length} actions per chunk, "
                f"got {action_chunk.shape[1]}"
            )
        if action_chunk.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected action_dim {self.action_dim}, got {action_chunk.shape[2]}"
            )

        return action_chunk[:, : self.action_chunk_length, :]

    def get_action(self, env: gym.Env, observation: Dict[str, Any]) -> torch.Tensor:
        if bool(self.env_requires_new_action_chunk.any()):
            new_chunk = self._request_new_action_chunk(observation)
            self.current_action_chunk[self.env_requires_new_action_chunk] = new_chunk[
                self.env_requires_new_action_chunk
            ]
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        idx = self.current_action_index
        action = self.current_action_chunk[torch.arange(self.num_envs), idx]

        if action.shape != (self.num_envs, self.action_dim):
            raise RuntimeError(
                f"Unexpected action shape {action.shape}, "
                f"expected ({self.num_envs}, {self.action_dim})"
            )

        self.current_action_index += 1

        reset_env_ids = self.current_action_index == self.action_chunk_length
        self.current_action_chunk[reset_env_ids] = 0.0
        self.env_requires_new_action_chunk[reset_env_ids] = True
        self.current_action_index[reset_env_ids] = -1

        return action

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True

        if self.is_remote:
            env_ids_list = env_ids.detach().cpu().tolist()
            self.remote_client.reset(env_ids=env_ids_list, options=None)

