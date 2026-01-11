# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import zmq

from .message_serializer import MessageSerializer
from .remote_policy_config import RemotePolicyConfig, ClientPolicyConfig

class PolicyClient:
    """Synchronous client for talking to a PolicyServer over ZeroMQ."""

    def __init__(self, config: RemotePolicyConfig) -> None:
        self._config = config
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._config.timeout_ms)
        self._socket.connect(f"tcp://{self._config.host}:{self._config.port}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception:
            warnings.warn(
                f"[PolicyClient] Failed to ping remote policy server at "
                f"{self._config.host}:{self._config.port}: {exc}"
            )
            return False

    def reset(self, env_ids=None, options: Optional[Dict[str, Any]] = None) -> Any:
        """Reset remote policy state."""
        resp = self.call_endpoint(
            endpoint="reset",
            data={"env_ids": env_ids, "options": options},
            requires_input=True,
        )
        if isinstance(resp, dict):
            status = resp.get("status")
            if status not in ("reset_success", "ok", "reset_ok", None):
                raise RuntimeError(f"Remote reset failed with status={status}, resp={resp}")
        return resp

    def kill(self) -> Any:
        """Ask remote server to stop main loop."""
        return self.call_endpoint("kill", requires_input=False)

    def get_action(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send policy_observations and get back policy action dict."""
        payload: Dict[str, Any] = {"observation": observation}
    
        resp = self.call_endpoint(
            endpoint="get_action",
            data=payload,
            requires_input=True,
        )
        return resp

    def get_init_info(self) -> Dict[str, Any]:
        """Fetch static initialization info from the remote policy."""
        resp = self.call_endpoint("get_init_info", requires_input=False)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(resp)!r}")
        return resp

    def get_init_info(self) -> ClientPolicyConfig:
        resp = self.call_endpoint("get_init_info", requires_input=False)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(resp)!r}")
        if "action_dim" not in resp or "action_chunk_length" not in resp:
            raise KeyError("Remote policy get_init_info must provide action_dim and action_chunk_length.")
        obs_keys = resp.get("observation_keys", [])
        if not isinstance(obs_keys, list):
            raise TypeError("observation_keys must be a list of strings.")
        return ClientPolicyConfig(
            action_dim=int(resp["action_dim"]),
            action_chunk_length=int(resp["action_chunk_length"]),
            observation_keys=list(obs_keys),
        )

    def set_task_description(self, task_description: Optional[str]) -> Dict[str, Any]:
        """Send task description to the remote policy."""
        payload: Dict[str, Any] = {"task_description": task_description}
        resp = self.call_endpoint(
            endpoint="set_task_description",
            data=payload,
            requires_input=True,
        )
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from set_task_description, got {type(resp)!r}")
        return resp

    def call_endpoint(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        requires_input: bool = True,
    ) -> Any:
        """Generic RPC helper."""
        request: Dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self._config.api_token:
            request["api_token"] = self._config.api_token

        self._socket.send(MessageSerializer.to_bytes(request))
        message = self._socket.recv()
        response = MessageSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def close(self) -> None:
        """Close the underlying ZeroMQ socket and context."""
        self._socket.close()
        self._context.term()

