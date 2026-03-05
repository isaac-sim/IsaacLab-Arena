import inspect
import types
from collections import defaultdict
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

from isaaclab_arena.remote_policy.action_protocol import ActionMode, ActionProtocol
from isaaclab_arena.remote_policy.client_state import ClientState
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.policy_client import PolicyClient, TransportTimeoutError
from isaaclab_arena.remote_policy.policy_server import PolicyServer
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqClientTransport
from isaaclab_arena.remote_policy.transport.zmq_ucx_transport import ZmqUcxServerTransport
from isaaclab_arena.policy.client_side_policy import ClientSidePolicy


class _RecordingServerTransport:
    def __init__(self) -> None:
        self.sent: dict[bytes, list[dict[str, Any]]] = defaultdict(list)

    def bind(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def send(self, client_id: bytes, payload: bytes) -> None:
        self.sent[client_id].append(MessageSerializer.from_bytes(payload))

    def recv(self) -> tuple[bytes, bytes]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def disconnect_client(self, client_id: bytes) -> None:
        pass


class _FallbackPolicy:
    config_class = None

    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        return {"status": "ok", "requested_action_mode": requested_action_mode}

    def get_action(
        self,
        observation: dict[str, Any],
        *,
        env_ids: list[int] | None = None,
        client_state: ClientState | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batch = 1
        for value in observation.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                batch = int(value.shape[0])
                break
        return {"action": np.zeros((batch, 1), dtype=np.float32)}, {"path": "single"}

    def set_task_description(
        self, task_description: str | None, *, env_ids: list[int] | None = None, client_state: ClientState | None = None
    ) -> dict[str, Any]:
        return {"task_description": task_description or ""}

    def reset(
        self, env_ids: list[int] | None = None, reset_options: dict[str, Any] | None = None, *, client_state: ClientState | None = None
    ) -> dict[str, Any]:
        return {"status": "reset_success"}


class _FailingInitPolicy(_FallbackPolicy):
    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        raise RuntimeError("injected get_init_info failure")


class _SuccessInitPolicy(_FallbackPolicy):
    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        return {
            "status": "success",
            "config": {
                "action_dim": 1,
                "observation_keys": ["obs"],
                "action_chunk_length": 1,
                "action_horizon": 1,
            },
        }


class _ClientTransportStub:
    def __init__(
        self,
        *,
        recv_payload: Any = None,
        recv_exc: Exception | None = None,
        send_tensor_exc: Exception | None = None,
    ) -> None:
        self.recv_payload = recv_payload
        self.recv_exc = recv_exc
        self.send_tensor_exc = send_tensor_exc
        self.rebuild_calls = 0
        self.reset_identity_calls = 0
        self.cached_identities: list[bytes] = []
        self.sent_payloads: list[bytes] = []

    def connect(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def send(self, payload: bytes) -> None:
        self.sent_payloads.append(payload)

    def recv(self) -> bytes:
        if self.recv_exc is not None:
            raise self.recv_exc
        if isinstance(self.recv_payload, list):
            assert self.recv_payload
            payload = self.recv_payload.pop(0)
            assert isinstance(payload, bytes)
            return payload
        assert isinstance(self.recv_payload, bytes)
        return self.recv_payload

    def rebuild(self) -> None:
        self.rebuild_calls += 1

    def cache_identity(self, identity: bytes) -> None:
        self.cached_identities.append(identity)

    def reset_identity(self) -> None:
        self.reset_identity_calls += 1

    def close(self) -> None:
        pass

    def send_tensor(self, tensor: Any) -> None:
        if self.send_tensor_exc is not None:
            raise self.send_tensor_exc


class _ConnectTimeoutTransport(_ClientTransportStub):
    def connect_ucx(self, host: str, port: int, client_id: bytes) -> None:
        raise TimeoutError(f"ucx connect timeout to {host}:{port}")


class _ShutdownClientStub:
    def __init__(self, kill_response: Any = None, kill_exc: Exception | None = None) -> None:
        self.kill_response = kill_response
        self.kill_exc = kill_exc
        self.kill_calls = 0
        self.disconnect_calls = 0
        self.close_calls = 0

    def kill(self) -> Any:
        self.kill_calls += 1
        if self.kill_exc is not None:
            raise self.kill_exc
        return self.kill_response

    def disconnect(self) -> Any:
        self.disconnect_calls += 1
        return {"status": "disconnected"}

    def close(self) -> None:
        self.close_calls += 1


class _HandshakeSendFailTransport(_RecordingServerTransport):
    def __init__(self) -> None:
        super().__init__()
        self._send_count = 0

    def send(self, client_id: bytes, payload: bytes) -> None:
        self._send_count += 1
        if self._send_count == 1:
            raise RuntimeError("injected handshake send failure")
        super().send(client_id, payload)


class _UcxCapableRecordingTransport(_RecordingServerTransport):
    def __init__(self, ucx_port: int = 16000) -> None:
        super().__init__()
        self.ucx_port = ucx_port
        self.start_ucx_listener_calls = 0

    def start_ucx_listener(self) -> int:
        self.start_ucx_listener_calls += 1
        return self.ucx_port


class _InitCleanupPolicyClientStub:
    def __init__(self, config: Any) -> None:
        self.ping_calls = 0
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.close_calls = 0

    def ping(self) -> bool:
        self.ping_calls += 1
        return True

    def connect(self, num_envs: int, requested_action_mode: str) -> dict[str, Any]:
        self.connect_calls += 1
        return {"status": "success", "config": {}, "zmq_identity": b"dummy-zmq-id"}

    def disconnect(self) -> dict[str, Any]:
        self.disconnect_calls += 1
        return {"status": "disconnected"}

    def close(self) -> None:
        self.close_calls += 1


class _BrokenProtocol(ActionProtocol):
    MODE: ClassVar[ActionMode] = ActionMode.CHUNK

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionProtocol:
        raise RuntimeError("protocol parse failed")

    def to_dict(self) -> dict[str, Any]:
        return {}


class _DummyClientSidePolicy(ClientSidePolicy):
    def get_action(self, env, observation):
        return torch.zeros((1, 1), dtype=torch.float32)

    @staticmethod
    def add_args_to_parser(parser):
        return parser

    @staticmethod
    def from_args(args):
        raise NotImplementedError


class _SingleRequestRunTransport(_RecordingServerTransport):
    def __init__(self, client_id: bytes, raw: bytes) -> None:
        super().__init__()
        self._client_id = client_id
        self._raw = raw
        self._served = False
        self.server = None

    def recv(self) -> tuple[bytes, bytes]:
        if self._served:
            raise TimeoutError("stop")
        self._served = True
        return self._client_id, self._raw

    def send(self, client_id: bytes, payload: bytes) -> None:
        super().send(client_id, payload)
        if self.server is not None:
            self.server._running = False


def _make_server() -> tuple[PolicyServer, _RecordingServerTransport]:
    transport = _RecordingServerTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=_FallbackPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
    )
    return server, transport


def test_public_remote_policy_constructors_do_not_expose_transport() -> None:
    assert "transport" not in inspect.signature(PolicyClient.__init__).parameters
    assert "transport" not in inspect.signature(PolicyServer.__init__).parameters


def test_single_dispatch_rejects_out_of_range_env_ids() -> None:
    server, transport = _make_server()
    client_id = b"env-id-out-of-range"
    server._client_states[client_id] = ClientState.create(2)
    request = {
        "endpoint": "get_action",
        "data": {
            "observation": {"obs": np.zeros((2, 2), dtype=np.float32)},
            "env_ids": [0, 2],
        },
    }

    server._dispatch_single(client_id, request)

    assert len(transport.sent[client_id]) == 1
    assert "out of range for num_envs=2" in transport.sent[client_id][0]["error"]


def test_single_dispatch_rejects_non_dict_data_with_explicit_error() -> None:
    server, transport = _make_server()
    bad_client = b"bad-data"
    server._client_states[bad_client] = ClientState.create(1)

    server._dispatch_single(bad_client, {"endpoint": "get_action", "data": []})

    assert len(transport.sent[bad_client]) == 1
    assert "Expected dict data" in transport.sent[bad_client][0]["error"]


def test_single_dispatch_rejects_non_dict_observation_with_explicit_error() -> None:
    server, transport = _make_server()
    bad_client = b"bad-obs"
    server._client_states[bad_client] = ClientState.create(1)

    server._dispatch_single(bad_client, {"endpoint": "get_action", "data": {"observation": []}})

    assert len(transport.sent[bad_client]) == 1
    assert "Expected dict observation" in transport.sent[bad_client][0]["error"]


def test_handle_get_action_rejects_non_dict_observation_at_handler_boundary() -> None:
    server, transport = _make_server()
    client_id = b"bad-observation"
    server._client_states[client_id] = ClientState.create(1)

    server._dispatch_single(client_id, {"endpoint": "get_action", "data": {"observation": []}})

    assert len(transport.sent[client_id]) == 1
    assert "Expected dict observation" in transport.sent[client_id][0]["error"]


def test_get_init_info_failure_does_not_leave_half_initialized_session() -> None:
    transport = _RecordingServerTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=_FailingInitPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
    )
    client_id = b"init-fail"

    server._dispatch_single(
        client_id,
        {
            "endpoint": "get_init_info",
            "data": {"requested_action_mode": "chunk", "num_envs": 1},
        },
    )

    assert len(transport.sent[client_id]) == 1
    assert "injected get_init_info failure" in transport.sent[client_id][0]["error"]
    assert client_id not in server._client_states
    assert client_id not in server._client_compression
    assert client_id not in server._client_tensor_compression
    assert client_id not in server._last_seen


def test_get_init_info_returns_zmq_identity() -> None:
    server, _ = _make_server()

    resp = server._handle_get_init_info(
        b"session-init",
        requested_action_mode="chunk",
        num_envs=1,
    )

    assert resp["zmq_identity"] == b"session-init"


def test_get_init_info_rejects_duplicate_live_zmq_identity() -> None:
    server, _ = _make_server()
    zmq_identity = b"already-live"

    first = server._handle_get_init_info(
        zmq_identity,
        requested_action_mode="chunk",
        num_envs=1,
    )
    assert first["zmq_identity"] == zmq_identity

    with pytest.raises(RuntimeError, match="already initialized"):
        server._handle_get_init_info(
            zmq_identity,
            requested_action_mode="chunk",
            num_envs=1,
        )


def test_get_init_info_does_not_negotiate_lz4_when_server_lacks_lz4() -> None:
    server, _ = _make_server()
    server._has_lz4 = False

    resp = server._handle_get_init_info(
        b"no-lz4",
        requested_action_mode="chunk",
        num_envs=1,
        compression_capabilities=["lz4", "none"],
    )

    assert resp["negotiated_zmq_compression"] == "none"


def test_get_init_info_uses_no_zmq_lz4_when_transport_negotiates_ucx() -> None:
    transport = _UcxCapableRecordingTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=_SuccessInitPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
    )
    server._has_ucx = True
    server._has_lz4 = True
    server._has_nvcomp = True

    resp = server._handle_get_init_info(
        b"ucx-no-zmq-lz4",
        requested_action_mode="chunk",
        num_envs=1,
        transport_capabilities=["zmq", "zmq_ucx"],
        compression_capabilities=["none", "lz4", "nvcomp_lz4"],
    )

    assert resp["negotiated_transport"] == "zmq_ucx"
    assert resp["negotiated_zmq_compression"] == "none"
    assert resp["negotiated_tensor_compression"] == "nvcomp_lz4"
    assert resp["negotiated_compression"] == "nvcomp_lz4"
    assert resp["ucx_port"] == transport.ucx_port


def test_get_init_info_negotiated_ucx_requires_ucx_capable_server_transport() -> None:
    server, _ = _make_server()
    server._has_ucx = True

    with pytest.raises(RuntimeError, match="does not support start_ucx_listener"):
        server._handle_get_init_info(
            b"server-no-ucx-transport",
            requested_action_mode="chunk",
            num_envs=1,
            transport_capabilities=["zmq", "zmq_ucx"],
            compression_capabilities=["none"],
        )


def test_get_init_info_send_failure_rolls_back_session_state() -> None:
    transport = _HandshakeSendFailTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=_SuccessInitPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
    )
    client_id = b"handshake-send-fail"

    server._dispatch_single(
        client_id,
        {
            "endpoint": "get_init_info",
            "data": {
                "requested_action_mode": "chunk",
                "num_envs": 1,
                "compression_capabilities": ["none"],
            },
        },
    )

    assert client_id not in server._client_states
    assert client_id not in server._client_compression
    assert client_id not in server._client_tensor_compression
    assert client_id not in server._last_seen
    assert len(transport.sent[client_id]) == 1
    assert "injected handshake send failure" in transport.sent[client_id][0]["error"]


def test_run_replies_to_top_level_non_dict_request() -> None:
    client_id = b"top-level-list"
    transport = _SingleRequestRunTransport(client_id, MessageSerializer.to_bytes(["not-a-dict"]))
    server = PolicyServer._from_transport_for_testing(
        policy=_FallbackPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
    )
    transport.server = server

    server.run()

    assert len(transport.sent[client_id]) == 1
    assert "Malformed request before dispatch" in transport.sent[client_id][0]["error"]
    assert "Expected dict request" in transport.sent[client_id][0]["error"]


def test_run_replies_to_top_level_decode_failure() -> None:
    client_id = b"top-level-decode"
    transport = _SingleRequestRunTransport(client_id, b"\xc1")
    server = PolicyServer._from_transport_for_testing(
        policy=_FallbackPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
    )
    transport.server = server

    server.run()

    assert len(transport.sent[client_id]) == 1
    assert "Malformed request before dispatch" in transport.sent[client_id][0]["error"]


def test_zmq_client_transport_starts_without_cached_identity() -> None:
    transport_a = ZmqClientTransport(timeout_ms=100)
    transport_b = ZmqClientTransport(timeout_ms=100)

    assert transport_a._identity is None
    assert transport_b._identity is None

    transport_a.close()
    transport_b.close()


def test_zmq_ucx_transport_uses_full_identity_for_endpoint_key() -> None:
    client_id = b"ila:test-machine:test-boot:1234:5678:9012"
    assert ZmqUcxServerTransport._ucx_key(client_id) == client_id


def test_gr00t_single_request_mixed_instructions_raise_explicit_error() -> None:
    gr00t_module = pytest.importorskip("isaaclab_arena_gr00t.policy.gr00t_remote_policy")
    Gr00tRemoteServerSidePolicy = gr00t_module.Gr00tRemoteServerSidePolicy
    policy = object.__new__(Gr00tRemoteServerSidePolicy)
    policy.policy_config = types.SimpleNamespace(language_instruction="default instruction")
    policy._task_description = None

    client_state = ClientState.create(2)
    client_state.instructions[0] = "pick cube"
    client_state.instructions[1] = "open drawer"

    with pytest.raises(NotImplementedError, match="mixed instructions"):
        policy._resolve_task_description([0, 1], client_state, batch_size=2)


def test_policy_client_zmq_timeout_does_not_require_reconnect() -> None:
    transport = _ClientTransportStub(recv_exc=TimeoutError("zmq timeout"))
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(TransportTimeoutError) as exc_info:
        client.call_endpoint("ping", requires_input=False)

    assert exc_info.value.source == "zmq_recv"
    assert exc_info.value.must_reconnect is False
    assert transport.rebuild_calls == 1


def test_policy_client_ucx_send_timeout_requires_reconnect() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes({"status": "unused"}),
        send_tensor_exc=TimeoutError("ucx send timeout"),
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )
    client._negotiated_transport = "zmq_ucx"

    with pytest.raises(TransportTimeoutError) as exc_info:
        client._get_action_ucx(
            {"obs": np.zeros((1, 2), dtype=np.float32)},
            None,
            {"rgb": torch.zeros(4, dtype=torch.uint8)},
        )

    assert exc_info.value.source == "ucx_send"
    assert exc_info.value.must_reconnect is True


def test_policy_client_ucx_zmq_recv_timeout_requires_reconnect() -> None:
    transport = _ClientTransportStub(recv_exc=TimeoutError("zmq timeout after ucx"))
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )
    client._negotiated_transport = "zmq_ucx"

    with pytest.raises(TransportTimeoutError) as exc_info:
        client._get_action_ucx(
            {"obs": np.zeros((1, 2), dtype=np.float32)},
            None,
            {"rgb": torch.zeros(4, dtype=torch.uint8)},
        )

    assert exc_info.value.source == "zmq_recv"
    assert exc_info.value.must_reconnect is True
    assert transport.rebuild_calls == 1


def test_policy_client_server_transport_timeout_is_rethrown_structurally() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "error": "UCX recv_tensor timed out while receiving get_action payload. Client should call reconnect().",
                "transport_timeout": True,
                "must_reconnect": True,
                "source": "ucx_recv",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )
    client._negotiated_transport = "zmq_ucx"

    with pytest.raises(TransportTimeoutError) as exc_info:
        client._get_action_ucx(
            {"obs": np.zeros((1, 2), dtype=np.float32)},
            None,
            {"rgb": torch.zeros(4, dtype=torch.uint8)},
        )

    assert exc_info.value.source == "ucx_recv"
    assert exc_info.value.must_reconnect is True


def test_policy_client_missing_client_state_is_rethrown_as_reconnect_required() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "error": "No ClientState for deadbeef on endpoint='get_action'. Client must call connect() / get_init_info first.",
                "must_reconnect": True,
                "source": "client_state",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(TransportTimeoutError) as exc_info:
        client.call_endpoint("get_action", data={"observation": {"obs": np.zeros((1, 2), dtype=np.float32)}}, requires_input=True)

    assert exc_info.value.source == "client_state"
    assert exc_info.value.must_reconnect is True


def test_policy_client_ucx_no_endpoint_is_rethrown_as_reconnect_required() -> None:
    transport = _ClientTransportStub(send_tensor_exc=RuntimeError("UCX endpoint not connected"))
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )
    client._negotiated_transport = "zmq_ucx"

    with pytest.raises(TransportTimeoutError) as exc_info:
        client._get_action_ucx(
            {"obs": np.zeros((1, 2), dtype=np.float32)},
            None,
            {"rgb": torch.zeros(4, dtype=torch.uint8)},
        )

    assert exc_info.value.source == "ucx_send"
    assert exc_info.value.must_reconnect is True


def test_policy_client_reconnect_reuses_last_action_mode() -> None:
    transport = _ClientTransportStub(
        recv_payload=[
            MessageSerializer.to_bytes({"status": "ok", "zmq_identity": b"zmq-id-1"}),
            MessageSerializer.to_bytes({"status": "ok", "zmq_identity": b"zmq-id-2"}),
        ]
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    client.connect(num_envs=2, requested_action_mode="velocity")
    client.reconnect()

    assert len(transport.sent_payloads) == 2
    first = MessageSerializer.from_bytes(transport.sent_payloads[0])
    second = MessageSerializer.from_bytes(transport.sent_payloads[1])
    assert first["data"]["requested_action_mode"] == "velocity"
    assert second["data"]["requested_action_mode"] == "velocity"
    assert transport.reset_identity_calls == 1
    assert transport.cached_identities == [b"zmq-id-1", b"zmq-id-2"]


def test_policy_client_connect_ucx_timeout_is_structured() -> None:
    transport = _ConnectTimeoutTransport(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "ok",
                "negotiated_transport": "zmq_ucx",
                "ucx_port": 15000,
                "zmq_identity": b"client-1",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(TransportTimeoutError) as exc_info:
        client.connect(num_envs=1, requested_action_mode="chunk")

    assert exc_info.value.source == "ucx_connect"
    assert exc_info.value.must_reconnect is True


def test_policy_client_connect_negotiated_ucx_requires_ucx_port() -> None:
    transport = _ConnectTimeoutTransport(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "ok",
                "negotiated_transport": "zmq_ucx",
                "zmq_identity": b"client-1",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(RuntimeError, match="missing 'ucx_port'"):
        client.connect(num_envs=1, requested_action_mode="chunk")


def test_policy_client_connect_negotiated_ucx_requires_zmq_identity() -> None:
    transport = _ConnectTimeoutTransport(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "ok",
                "negotiated_transport": "zmq_ucx",
                "ucx_port": 15000,
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(RuntimeError, match="missing 'zmq_identity'"):
        client.connect(num_envs=1, requested_action_mode="chunk")


def test_policy_client_connect_negotiated_ucx_requires_ucx_capable_transport() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "ok",
                "negotiated_transport": "zmq_ucx",
                "ucx_port": 15000,
                "zmq_identity": b"client-1",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(RuntimeError, match="does not support connect_ucx"):
        client.connect(num_envs=1, requested_action_mode="chunk")


def test_policy_client_connect_requires_server_zmq_identity() -> None:
    transport = _ClientTransportStub(recv_payload=MessageSerializer.to_bytes({"status": "ok"}))
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(RuntimeError, match="missing 'zmq_identity'"):
        client.connect(num_envs=1, requested_action_mode="chunk")


def test_shutdown_remote_kill_rejected_falls_back_to_disconnect() -> None:
    policy = object.__new__(ClientSidePolicy)
    policy._client = _ShutdownClientStub(
        {"status": "rejected", "reason": "remote kill disabled"}
    )

    with pytest.warns(DeprecationWarning):
        policy.shutdown_remote(kill_server=True)

    assert policy._client.kill_calls == 1
    assert policy._client.disconnect_calls == 1
    assert policy._client.close_calls == 1


def test_shutdown_remote_kill_allowed_does_not_fallback_disconnect() -> None:
    policy = object.__new__(ClientSidePolicy)
    policy._client = _ShutdownClientStub({"status": "stopping"})

    with pytest.warns(DeprecationWarning):
        policy.shutdown_remote(kill_server=True)

    assert policy._client.kill_calls == 1
    assert policy._client.disconnect_calls == 0
    assert policy._client.close_calls == 1


def test_shutdown_remote_default_path_prefers_disconnect() -> None:
    policy = object.__new__(ClientSidePolicy)
    policy._client = _ShutdownClientStub()

    policy.shutdown_remote(kill_server=False)

    assert policy._client.kill_calls == 0
    assert policy._client.disconnect_calls == 1
    assert policy._client.close_calls == 1


def test_shutdown_remote_kill_exception_still_attempts_disconnect() -> None:
    policy = object.__new__(ClientSidePolicy)
    policy._client = _ShutdownClientStub(kill_exc=RuntimeError("kill failed"))

    with pytest.warns(DeprecationWarning):
        policy.shutdown_remote(kill_server=True)

    assert policy._client.kill_calls == 1
    assert policy._client.disconnect_calls == 1
    assert policy._client.close_calls == 1


def test_client_side_policy_init_failure_disconnects_and_closes(monkeypatch) -> None:
    stub = _InitCleanupPolicyClientStub(config=None)
    monkeypatch.setattr(
        "isaaclab_arena.policy.client_side_policy.PolicyClient",
        lambda config: stub,
    )

    with pytest.raises(RuntimeError, match="protocol parse failed"):
        _DummyClientSidePolicy(
            config=None,
            remote_config=RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
            protocol_cls=_BrokenProtocol,
            num_envs=1,
        )

    assert stub.ping_calls == 1
    assert stub.connect_calls == 1
    assert stub.disconnect_calls == 1
    assert stub.close_calls == 1
