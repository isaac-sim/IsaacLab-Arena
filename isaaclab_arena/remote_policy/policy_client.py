# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import zmq

from .message_serializer import MessageSerializer
from .remote_policy_config import RemotePolicyConfig 

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
        return self.call_endpoint(
            endpoint="reset",
            data={"env_ids": env_ids, "options": options},
            requires_input=True,
        )

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

