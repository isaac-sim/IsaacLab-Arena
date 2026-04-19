# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from contextlib import suppress
import torch
import warnings
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.remote_policy.mooncake_config import MooncakeTransportConfig, add_client_mooncake_args
from isaaclab_arena.remote_policy.action_protocol import ActionMode, ActionProtocol
from isaaclab_arena.remote_policy.policy_client import PolicyClient
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class ClientSidePolicy(PolicyBase):
    """Base class for policies that query a remote policy server.

    Responsibilities:
      - Manage RemotePolicyConfig and PolicyClient.
      - Handshake with the server via initialize_session().
      - Provide observation packing based on observation_keys.
      - Provide shared CLI helpers for remote-related arguments.

    Subclasses:
      - Must implement get_action().
    """

    def __init__(
        self,
        config: Any,
        remote_config: RemotePolicyConfig,
        protocol_cls: type[ActionProtocol],
        num_envs: int = 1,
        tensor_device: str | None = None,
    ) -> None:
        super().__init__(config=config)

        if protocol_cls.MODE is None:
            raise ValueError(f"{protocol_cls.__name__}.MODE must be defined as an ActionMode.")

        self.protocol_cls = protocol_cls
        requested_action_mode: ActionMode = protocol_cls.MODE

        self._remote_config = remote_config
        self._client = PolicyClient(config=self._remote_config, tensor_device=tensor_device)
        self._num_envs = num_envs

        try:
            # Perform the session handshake once. This covers connectivity,
            # explicit transport/compression validation, and get_init_info in a
            # single round trip.
            init_resp = self._client.initialize_session(
                num_envs=num_envs,
                requested_action_mode=requested_action_mode.value,
            )

            if not isinstance(init_resp, dict):
                raise TypeError(f"Expected dict from get_init_info, got {type(init_resp)!r}")

            status = init_resp.get("status", "error")
            if status != "success":
                message = init_resp.get("message") or init_resp.get("error", "no message")
                raise RuntimeError(f"Remote policy get_init_info failed with status='{status}': {message}")

            cfg_dict = init_resp.get("config")
            if not isinstance(cfg_dict, dict):
                raise TypeError(
                    f"Remote policy get_init_info must return a 'config' dict inside the response, got {type(cfg_dict)!r}"
                )

            self._protocol: ActionProtocol = self.protocol_cls.from_dict(cfg_dict)
        except Exception:
            if getattr(self._client, "session_initialized", False):
                with suppress(Exception):
                    self._client.disconnect()
            if getattr(self._client, "_transport_connected", False):
                with suppress(Exception):
                    self._client.close()
            raise

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
        """Pack selected observation entries into a flat dict for the server.

        This layer only decides *which* observation entries belong in the
        request. Transport-specific conversion (for example CPU numpy vs UCX
        tensor path) is handled later by ``PolicyClient``.
        """
        packed: dict[str, Any] = {}
        for key_path in self.observation_keys:
            packed[key_path] = self._get_nested_observation(observation, key_path)
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
            warnings.warn(
                "--remote_kill_on_exit is deprecated. Prefer disconnect() for normal teardown "
                "and enable --allow_remote_kill explicitly only when server shutdown is intended.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                resp = self._client.kill()
                if isinstance(resp, dict) and resp.get("status") == "rejected":
                    reason = resp.get("reason")
                    if reason == "remote kill disabled":
                        try:
                            self._client.disconnect()
                        except Exception as disconnect_exc:
                            print(
                                "[ClientSidePolicy] Remote kill was rejected and disconnect() also failed: "
                                f"{disconnect_exc}"
                            )
            except Exception as exc:
                print(f"[ClientSidePolicy] Failed to send kill to remote server: {exc}")
                try:
                    self._client.disconnect()
                except Exception as disconnect_exc:
                    print(
                        "[ClientSidePolicy] Remote kill failed and disconnect() also failed: "
                        f"{disconnect_exc}"
                    )
        else:
            try:
                self._client.disconnect()
            except Exception as exc:
                print(f"[ClientSidePolicy] Failed to disconnect remote client cleanly: {exc}")
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
            help="Deprecated: request remote server shutdown on exit. Prefer the default disconnect behavior.",
        )
        group.add_argument(
            "--remote_transport_mode",
            type=str,
            default="zmq",
            choices=["zmq", "zmq_mooncake"],
            help="Remote transport selection for the simple mainline.",
        )
        add_client_mooncake_args(group)
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
            transport_mode=args.remote_transport_mode,
            mooncake=MooncakeTransportConfig.from_public_args(
                local_hostname=args.remote_mooncake_local_hostname,
            ),
        )
