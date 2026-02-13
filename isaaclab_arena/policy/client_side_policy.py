# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import torch
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.remote_policy.action_protocol import ActionMode, ActionProtocol
from isaaclab_arena.remote_policy.policy_client import PolicyClient
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class ClientSidePolicy(PolicyBase):
    """Base class for policies that query a remote policy server.

    Responsibilities:
      - Manage RemotePolicyConfig and PolicyClient.
      - Handshake with the server via get_init_info().
      - Provide observation packing based on observation_keys.
      - Provide shared CLI helpers for remote-related arguments.

    Subclasses:
      - Must implement get_action().
    """

    def __init__(self, config: Any, remote_config: RemotePolicyConfig, protocol_cls: type[ActionProtocol]) -> None:
        super().__init__(config=config)

        if protocol_cls.MODE is None:
            raise ValueError(f"{protocol_cls.__name__}.MODE must be defined as an ActionMode.")

        self.protocol_cls = protocol_cls
        requested_action_mode: ActionMode = protocol_cls.MODE

        self._remote_config = remote_config
        self._client = PolicyClient(config=self._remote_config)

        # 1) Ping server to ensure connectivity.
        if not self._client.ping():
            raise RuntimeError(
                f"Failed to connect to remote policy server at {self._remote_config.host}:{self._remote_config.port}."
            )

        # 2) Handshake: send requested_action_mode, parse response.
        init_resp = self._client.get_init_info(requested_action_mode=requested_action_mode.value)

        if not isinstance(init_resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(init_resp)!r}")

        status = init_resp.get("status", "error")
        if status != "success":
            message = init_resp.get("message", "no message")
            raise RuntimeError(f"Remote policy get_init_info failed with status='{status}': {message}")

        cfg_dict = init_resp.get("config")
        if not isinstance(cfg_dict, dict):
            raise TypeError(
                f"Remote policy get_init_info must return a 'config' dict inside the response, got {type(cfg_dict)!r}"
            )

        self._protocol: ActionProtocol = self.protocol_cls.from_dict(cfg_dict)

    # ---------------------- properties ----------------------------------
    @property
    def protocol(self) -> ActionProtocol:
        return self._protocol

    @property
    def action_mode(self) -> ActionMode:
        return self._protocol.mode

    @property
    def action_dim(self) -> int:
        return self._protocol.action_dim

    @property
    def observation_keys(self) -> list[str]:
        return list(self._protocol.observation_keys)

    @property
    def remote_config(self) -> RemotePolicyConfig:
        return self._remote_config

    @property
    def remote_client(self) -> PolicyClient:
        return self._client

    @property
    def is_remote(self) -> bool:
        return True

    # ---------------------- observation packing -------------------------
    @staticmethod
    def _get_nested_observation(observation: dict[str, Any], key_path: str) -> Any:
        """Get a nested value from a dict using 'a.b.c' path."""
        cur: Any = observation

        for k in key_path.split("."):
            cur = cur[k]
        return cur

    def pack_observation_for_server(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Pack selected observation entries into a flat CPU dict for the server.

        Uses `self.observation_keys` from ClientSidePolicyConfig and:
          - Extracts values using nested key paths.
          - Moves torch.Tensor values to CPU numpy arrays.
        """
        packed: dict[str, Any] = {}
        for key_path in self.observation_keys:
            value = self._get_nested_observation(observation, key_path)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            packed[key_path] = value
        return packed

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Optionally reset remote policy state.

        Client-side state should be reset in subclasses.
        """
        env_ids_list = None
        if env_ids is not None:
            env_ids_list = env_ids.detach().cpu().tolist()
        self._client.reset(env_ids=env_ids_list, options=None)

    def shutdown_remote(self, kill_server: bool = False) -> None:
        """Clean up the remote client and optionally stop the remote server."""
        if kill_server:
            try:
                self._client.call_endpoint("kill", requires_input=False)
            except Exception as exc:
                print(f"[ClientSidePolicy] Failed to send kill to remote server: {exc}")
        self._client.close()

    # ---------------------- shared CLI helpers --------------------------

    @staticmethod
    def add_remote_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add shared remote-policy arguments to the parser.

        This should be called from subclass.add_args_to_parser().
        """
        group = parser.add_argument_group(
            "Remote Policy",
            "Arguments for connecting to a remote policy server.",
        )
        group.add_argument(
            "--remote_host",
            type=str,
            default=None,
            required=True,
            help="Remote policy server host.",
        )
        group.add_argument(
            "--remote_port",
            type=int,
            default=5555,
            help="Remote policy server port.",
        )
        group.add_argument(
            "--remote_api_token",
            type=str,
            default=None,
            help="API token for the remote policy server.",
        )
        group.add_argument(
            "--remote_timeout_ms",
            type=int,
            default=15000,
            help="Timeout (ms) for remote policy requests.",
        )
        group.add_argument(
            "--remote_kill_on_exit",
            action="store_true",
            help="If set, send a 'kill' request to the remote policy server when the run finishes.",
        )
        return parser

    @staticmethod
    def build_remote_config_from_args(args: argparse.Namespace) -> RemotePolicyConfig:
        """Construct RemotePolicyConfig from CLI arguments.

        Assumes add_remote_args_to_parser() has been called on the parser.
        """

        return RemotePolicyConfig(
            host=args.remote_host,
            port=args.remote_port,
            api_token=args.remote_api_token,
            timeout_ms=args.remote_timeout_ms,
        )
