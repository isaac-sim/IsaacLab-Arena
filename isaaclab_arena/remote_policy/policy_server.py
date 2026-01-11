# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import zmq

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy


@dataclass
class EndpointHandler:
    handler: Callable[..., Any]
    requires_input: bool = True


class PolicyServer:
    def __init__(
        self,
        policy: ServerSidePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
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

        self._endpoints: dict[str, EndpointHandler] = {}
        self._register_default_endpoints()

    def _register_default_endpoints(self) -> None:
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._handle_kill, requires_input=False)
        self.register_endpoint("get_action", self._handle_get_action, requires_input=True)
        self.register_endpoint("reset", self._handle_reset, requires_input=True)
        self.register_endpoint("get_init_info", self._handle_get_init_info, requires_input=True)
        self.register_endpoint("set_task_description", self._handle_set_task_description, requires_input=True)
        print(f"[PolicyServer] registered endpoints: {list(self._endpoints.keys())}")

    def register_endpoint(
        self,
        name: str,
        handler: Callable[..., Any],
        requires_input: bool = True,
    ) -> None:
        self._endpoints[name] = EndpointHandler(handler=handler, requires_input=requires_input)

    def _handle_get_init_info(
        self,
        requested_action_mode: str,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle get_init_info: requested_action_mode={requested_action_mode!r}")
        resp = self._policy.get_init_info(requested_action_mode=requested_action_mode)
        if not isinstance(resp, dict):
            raise TypeError(f"Policy.get_init_info() must return dict, got {type(resp)!r}")
        return resp

    def _handle_set_task_description(
        self,
        task_description: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle set_task_description: {task_description!r}")
        resp = self._policy.set_task_description(task_description)
        if not isinstance(resp, dict):
            raise TypeError(f"Policy.set_task_description() must return dict, got {type(resp)!r}")
        return resp

    def _handle_ping(self) -> dict[str, Any]:
        print("[PolicyServer] handle ping")
        return {"status": "ok"}

    def _handle_kill(self) -> dict[str, Any]:
        print("[PolicyServer] handle kill -> stopping")
        self._running = False
        return {"status": "stopping"}

    def _handle_get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print("[PolicyServer] handle get_action")
        if options is not None:
            print(f"  options keys: {list(options.keys())}")
        action, info = self._policy.get_action(
            observation=observation,
            options=options,
        )

        if not isinstance(action, dict):
            raise TypeError(f"Policy.get_action() must return (dict, dict), got action type={type(action)!r}")
        if not isinstance(info, dict):
            raise TypeError(f"Policy.get_action() must return (dict, dict), got info type={type(info)!r}")

        merged: dict[str, Any] = {}
        merged.update(action)
        if any(k in merged for k in info.keys()):
            raise ValueError(f"Policy info keys conflict with action keys: {set(merged.keys()) & set(info.keys())}")
        merged.update(info)

        return merged

    def _handle_reset(self, env_ids=None, options=None, **_: Any) -> dict[str, Any]:
        print(f"[PolicyServer] handle reset: env_ids={env_ids}, options={options}")
        status: dict[str, Any] = {"status": "reset_success"}
        if hasattr(self._policy, "reset"):
            resp = self._policy.reset(env_ids=env_ids, reset_options=options)
            if isinstance(resp, dict):
                status.update(resp)
        return status

    def _validate_token(self, request: dict[str, Any]) -> bool:
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
                request = MessageSerializer.from_bytes(raw)

                if not isinstance(request, dict):
                    raise TypeError(f"Expected dict request, got {type(request)!r}")

                print(f"[PolicyServer] request keys: {list(request.keys())}")

                if not self._validate_token(request):
                    self._socket.send(MessageSerializer.to_bytes({"error": "Unauthorized: invalid api_token"}))
                    continue

                endpoint = request.get("endpoint", "get_action")
                if "endpoint" not in request:
                    self._socket.send(MessageSerializer.to_bytes({"error": "Missing 'endpoint' in request"}))
                    continue

                endpoint = request["endpoint"]

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

                resp_bytes = MessageSerializer.to_bytes(result)
                print(f"[PolicyServer] sending response ({len(resp_bytes)} bytes)")
                self._socket.send(resp_bytes)
            except zmq.Again:
                # timeout, loop again
                continue
            except Exception as exc:
                import traceback

                print(f"[PolicyServer] Error: {exc}")
                print(traceback.format_exc())
                self._socket.send(MessageSerializer.to_bytes({"error": str(exc)}))

    def close(self) -> None:
        """Stop the main loop and close ZMQ resources."""
        self._running = False
        try:
            self._socket.close(0)
        except Exception as exc:
            print(f"[PolicyServer] socket.close() error: {exc}")
        try:
            self._context.term()
        except Exception as exc:
            print(f"[PolicyServer] context.term() error: {exc}")

    @staticmethod
    def start(
        policy: ServerSidePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
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
