# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from typing import Any

import pytest
import zmq

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.policy_client import PolicyClient
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class _DummyServer:
    """Minimal test server that emulates a subset of PolicyServer behavior.

    It only understands the endpoints used by PolicyClient tests and always
    responds with well-formed msgpack-encoded dictionaries.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5557, api_token: str | None = None) -> None:
        self._host = host
        self._port = port
        self._api_token = api_token
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the server loop in a background thread."""
        bind_addr = f"tcp://{self._host}:{self._port}"
        self._socket.bind(bind_addr)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server loop and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._socket.close(0)
        self._context.term()

    def _loop(self) -> None:
        """Event loop that receives one request and sends one response."""
        while self._running:
            try:
                message = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue

            request: dict[str, Any] = MessageSerializer.from_bytes(message)

            # Real code uses "api_token" on the wire.
            if self._api_token is not None:
                if request.get("api_token") != self._api_token:
                    response: dict[str, Any] = {"error": "invalid apitoken"}
                    self._socket.send(MessageSerializer.to_bytes(response))
                    continue

            endpoint = request.get("endpoint", "")
            data = request.get("data", {}) or {}

            if endpoint == "get_action":
                # Return a minimal valid action payload; client expects a dict.
                resp = {"action": [[0.0, 1.0, 2.0]], "info": {"dummy": True}}
            elif endpoint == "get_init_info":
                resp = {"obs_keys": ["rgb", "depth"], "action_dim": 3}
            elif endpoint == "set_task_description":
                desc = data.get("task_description", "")
                resp = {"status": "ok", "echo": desc}
            elif endpoint == "ping":
                resp = {"status": "alive"}
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}

            self._socket.send(MessageSerializer.to_bytes(resp))


@pytest.fixture
def dummy_server() -> _DummyServer:
    """Fixture that starts a dummy server and tears it down after the test."""
    server = _DummyServer(host="127.0.0.1", port=5557, api_token="SECRET")
    server.start()
    # Give the background thread a short time to bind the socket.
    time.sleep(0.1)
    try:
        yield server
    finally:
        server.stop()


def test_policy_client_call_endpoint_and_get_action(dummy_server: _DummyServer) -> None:
    """PolicyClient should be able to call endpoints and parse responses."""
    config = RemotePolicyConfig(host="127.0.0.1", port=5557, api_token="SECRET", timeout_ms=2000)
    client = PolicyClient(config=config)

    # Test ping endpoint without input.
    resp = client.call_endpoint(endpoint="ping", data=None, requires_input=False)
    assert isinstance(resp, dict)
    assert resp.get("status") == "alive"

    # Test get_action endpoint with dummy observation.
    action_resp = client.get_action({
        "rgb": "dummy",  # Content does not matter for this dummy server.
    })
    assert isinstance(action_resp, dict)
    assert "action" in action_resp
    assert "info" in action_resp

    action = action_resp["action"]
    assert isinstance(action, list)
    assert len(action) == 1
    assert len(action[0]) == 3

    client.close()


def test_policy_client_get_init_info_and_set_task_description(dummy_server: _DummyServer) -> None:
    """get_init_info and set_task_description should return dictionaries."""
    config = RemotePolicyConfig(host="127.0.0.1", port=5557, api_token="SECRET", timeout_ms=2000)
    client = PolicyClient(config=config)

    init_info = client.get_init_info({"dummy": True})
    assert isinstance(init_info, dict)
    assert "obs_keys" in init_info
    assert "action_dim" in init_info

    desc = "open the microwave door"
    status = client.set_task_description(desc)
    assert isinstance(status, dict)
    assert status.get("status") == "ok"
    assert status.get("echo") == desc

    client.close()
