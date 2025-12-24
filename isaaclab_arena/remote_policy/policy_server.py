# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import zmq

from .model_policy import ModelPolicy
from .message_serializer import MessageSerializer


@dataclass
class EndpointHandler:
    handler: Callable[..., Any]
    requires_input: bool = True


class PolicyServer:
    def __init__(
        self,
        policy: ModelPolicy,
        host: str = "*",
        port: int = 5555,
        api_token: Optional[str] = None,
        timeout_ms: int = 15000,
    ) -> None:
        self._policy = policy
        self._running = True
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        bind_addr = f"tcp://{host}:{port}"
        print(f"[PolicyServer] binding on {bind_addr}")
        self._socket.bind(bind_addr)
        self._api_token = api_token
        self._serializer = MessageSerializer

        self._endpoints: Dict[str, EndpointHandler] = {}
        self._register_default_endpoints()

    def _register_default_endpoints(self) -> None:
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._handle_kill, requires_input=False)
        self.register_endpoint("get_action", self._handle_get_action, requires_input=True)
        self.register_endpoint("reset", self._handle_reset, requires_input=True)
        print(f"[PolicyServer] registered endpoints: {list(self._endpoints.keys())}")

    def register_endpoint(
        self,
        name: str,
        handler: Callable[..., Any],
        requires_input: bool = True,
    ) -> None:
        self._endpoints[name] = EndpointHandler(handler=handler, requires_input=requires_input)

    def _handle_ping(self) -> Dict[str, Any]:
        print("[PolicyServer] handle ping")
        return {"status": "ok"}

    def _handle_kill(self) -> Dict[str, Any]:
        print("[PolicyServer] handle kill -> stopping")
        self._running = False
        return {"status": "stopping"}

    def _handle_get_action(
        self,
        observation: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        print("[PolicyServer] handle get_action")
        print(f"  observation keys: {list(observation.keys())}")
        if options is not None:
            print(f"  options keys: {list(options.keys())}")
        action, info = self._policy.get_action(
            observation=observation,
            options=options,
        )
        return {"action": action, "info": info}

    def _handle_reset(self, env_ids=None, options=None, **_: Any) -> Dict[str, Any]:
        print(f"[PolicyServer] handle reset: env_ids={env_ids}, options={options}")
        if hasattr(self._policy, "reset"):
            self._policy.reset(env_ids=env_ids, reset_options=options)
        return {"status": "reset"}

    def _validate_token(self, request: Dict[str, Any]) -> bool:
        if self._api_token is None:
            return True
        ok = request.get("api_token") == self._api_token
        if not ok:
            print("[PolicyServer] invalid api_token in request")
        return ok

    def run(self) -> None:
        addr = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"[PolicyServer] listening on {addr}, api_token={self._api_token!r}")
        while self._running:
            try:
                raw = self._socket.recv()
                print(f"[PolicyServer] received {len(raw)} bytes")
                request = self._serializer.from_bytes(raw)

                if not isinstance(request, dict):
                    raise TypeError(f"Expected dict request, got {type(request)!r}")

                print(f"[PolicyServer] request keys: {list(request.keys())}")

                if not self._validate_token(request):
                    self._socket.send(
                        self._serializer.to_bytes({"error": "Unauthorized: invalid api_token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")
                handler = self._endpoints.get(endpoint)
                if handler is None:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                print(f"[PolicyServer] dispatch endpoint='{endpoint}'")

                data = request.get("data", {}) or {}
                if not isinstance(data, dict):
                    raise TypeError(f"Expected dict data, got {type(data)!r}")

                if handler.requires_input:
                    result = handler.handler(**data)
                else:
                    result = handler.handler()

                resp_bytes = self._serializer.to_bytes(result)
                print(f"[PolicyServer] sending response ({len(resp_bytes)} bytes)")
                self._socket.send(resp_bytes)
            except zmq.Again:
                # timeout, loop again
                continue
            except Exception as exc:
                import traceback

                print(f"[PolicyServer] Error: {exc}")
                print(traceback.format_exc())
                self._socket.send(self._serializer.to_bytes({"error": str(exc)}))

    @staticmethod
    def start(
        policy: ModelPolicy,
        host: str = "*",
        port: int = 5555,
        api_token: Optional[str] = None,
        timeout_ms: int = 15000,
    ) -> None:
        server = PolicyServer(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            timeout_ms=timeout_ms,
        )
        server.run()

