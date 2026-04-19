import threading
import time
from typing import Any

import pytest
import zmq

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.policy_client import PolicyClient
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class _DummyServer:
    """Minimal server compatible with the mainline DEALER client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5557, api_token: str | None = None) -> None:
        self._host = host
        self._port = port
        self._api_token = api_token
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        bind_addr = f"tcp://{self._host}:{self._port}"
        self._socket.bind(bind_addr)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._socket.close(0)
        self._context.term()

    def _loop(self) -> None:
        while self._running:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                continue

            if len(parts) < 3 or parts[1] != b"":
                continue

            identity = parts[0]
            payload = parts[2]
            request: dict[str, Any] = MessageSerializer.from_bytes(payload)

            if self._api_token is not None and request.get("api_token") != self._api_token:
                self._socket.send_multipart(
                    [identity, b"", MessageSerializer.to_bytes({"error": "invalid apitoken"})]
                )
                continue

            endpoint = request.get("endpoint", "")
            data = request.get("data", {}) or {}

            if endpoint == "get_action":
                response = {"action": [[0.0, 1.0, 2.0]], "info": {"dummy": True}}
            elif endpoint == "get_init_info":
                response = {
                    "status": "success",
                    "config": {
                        "action_dim": 3,
                        "observation_keys": ["rgb", "depth"],
                        "action_chunk_length": 1,
                        "action_horizon": 1,
                    },
                    "num_envs": data.get("num_envs", 1),
                    "zmq_identity": identity,
                }
            elif endpoint == "set_task_description":
                response = {"task_description": data.get("task_description", "")}
            elif endpoint == "ping":
                response = {"status": "alive"}
            elif endpoint == "disconnect":
                response = {"status": "disconnected"}
            else:
                response = {"error": f"unknown endpoint {endpoint!r}"}

            self._socket.send_multipart([identity, b"", MessageSerializer.to_bytes(response)])


@pytest.fixture
def dummy_server() -> _DummyServer:
    server = _DummyServer(host="127.0.0.1", port=5557, api_token="SECRET")
    server.start()
    time.sleep(0.1)
    try:
        yield server
    finally:
        server.stop()


def test_policy_client_call_endpoint_and_get_action(dummy_server: _DummyServer) -> None:
    del dummy_server
    config = RemotePolicyConfig(host="127.0.0.1", port=5557, api_token="SECRET", timeout_ms=2000)
    client = PolicyClient(config=config)

    response = client.call_endpoint(endpoint="ping", data=None, requires_input=False)
    assert isinstance(response, dict)
    assert response.get("status") == "alive"

    init_response = client.initialize_session(num_envs=1, requested_action_mode="chunk")
    assert init_response["num_envs"] == 1

    action_response = client.get_action({"rgb": "dummy"})
    assert isinstance(action_response, dict)
    assert "action" in action_response
    assert "info" in action_response
    assert len(action_response["action"][0]) == 3

    client.close()


def test_policy_client_initialize_session_and_set_task_description(dummy_server: _DummyServer) -> None:
    del dummy_server
    config = RemotePolicyConfig(host="127.0.0.1", port=5557, api_token="SECRET", timeout_ms=2000)
    client = PolicyClient(config=config)

    init_response = client.initialize_session(num_envs=2, requested_action_mode="chunk")
    assert isinstance(init_response, dict)
    assert init_response["num_envs"] == 2
    assert "config" in init_response

    description = "open the microwave door"
    status = client.set_task_description(description)
    assert isinstance(status, dict)
    assert status.get("task_description") == description

    client.disconnect()
    client.close()
