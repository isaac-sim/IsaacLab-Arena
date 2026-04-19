import pytest

# All tests in this file target the old v2 remote-policy wire protocol which
# included tensor compression. Compression has been removed from the codebase.
# These tests need rewriting against the new compression-free wire protocol
# before they can run again.
pytestmark = pytest.mark.skip(
    reason="Legacy v2 compression tests; obsolete after compression removal."
)

import inspect
import types
from collections import defaultdict
from typing import Any, ClassVar

import numpy as np
import torch

from isaaclab_arena.policy.client_side_policy import ClientSidePolicy
from isaaclab_arena.remote_policy.action_protocol import ActionMode, ActionProtocol
from isaaclab_arena.remote_policy.client_state import ClientState
from isaaclab_arena.remote_policy.compression import TensorPayloadCodec
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.policy_client import PolicyClient, TransportTimeoutError
from isaaclab_arena.remote_policy.policy_server import PolicyServer, ServerTransportTimeoutError
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqClientTransport


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
        del client_id

    @property
    def transport_mode(self) -> str:
        return "zmq"


class _MooncakeServerTransport(_RecordingServerTransport):
    def __init__(self) -> None:
        super().__init__()
        self.handshake_calls = 0
        self.prepare_recv_tensor_calls = 0
        self.recv_tensor_calls = 0

    def get_handshake_metadata(self, client_id: bytes) -> dict[str, Any]:
        del client_id
        self.handshake_calls += 1
        return {
            "mooncake_protocol": "rdma",
            "mooncake_server_session_id": "server-session",
        }

    def prepare_recv_tensor(self, client_id: bytes, source_info: dict[str, Any]) -> None:
        del client_id, source_info
        self.prepare_recv_tensor_calls += 1

    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: Any | None = None) -> Any:
        del client_id, buffer
        self.recv_tensor_calls += 1
        return torch.arange(nbytes, dtype=torch.uint8)

    @property
    def transport_mode(self) -> str:
        return "zmq_mooncake"


class _FailingMooncakeRecvTransport(_MooncakeServerTransport):
    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: Any | None = None) -> Any:
        del client_id, nbytes, buffer
        self.recv_tensor_calls += 1
        raise TimeoutError("mooncake recv timeout")


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
        del env_ids, client_state, kwargs
        batch = 1
        for value in observation.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                batch = int(value.shape[0])
                break
        return {"action": np.zeros((batch, 1), dtype=np.float32)}, {"path": "single"}

    def set_task_description(
        self,
        task_description: str | None,
        *,
        env_ids: list[int] | None = None,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        del env_ids, client_state
        return {"task_description": task_description or ""}

    def reset(
        self,
        env_ids: list[int] | None = None,
        reset_options: dict[str, Any] | None = None,
        *,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        del env_ids, reset_options, client_state
        return {"status": "reset_success"}


class _FailingInitPolicy(_FallbackPolicy):
    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        del requested_action_mode
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
            "requested_action_mode": requested_action_mode,
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
        self.connect_calls = 0
        self.rebuild_calls = 0
        self.reset_identity_calls = 0
        self.cached_identities: list[bytes] = []
        self.sent_payloads: list[bytes] = []
        self.close_calls = 0

    def connect(self, endpoint: str) -> None:
        self.connect_calls += 1
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
        self.close_calls += 1

    def send_tensor(self, tensor: Any) -> None:
        if self.send_tensor_exc is not None:
            raise self.send_tensor_exc
        del tensor

    @property
    def transport_mode(self) -> str:
        return "zmq"


class _MooncakeClientTransportStub(_ClientTransportStub):
    def __init__(
        self,
        *,
        recv_payload: Any = None,
        recv_exc: Exception | None = None,
        connect_backend_exc: Exception | None = None,
        send_tensor_exc: Exception | None = None,
    ) -> None:
        super().__init__(
            recv_payload=recv_payload,
            recv_exc=recv_exc,
            send_tensor_exc=send_tensor_exc,
        )
        self.connect_backend_exc = connect_backend_exc
        self.connect_backend_calls = 0
        self.sent_tensors: list[Any] = []

    def connect_comm_backend(
        self,
        *,
        handshake_response: dict[str, Any],
        server_host: str,
        zmq_identity: bytes | None,
    ) -> None:
        del handshake_response, server_host, zmq_identity
        self.connect_backend_calls += 1
        if self.connect_backend_exc is not None:
            raise self.connect_backend_exc

    def tensor_source_info(self) -> dict[str, Any]:
        return {
            "session_id": "client-session",
            "buffer_ptr": 12345,
            "buffer_bytes": 4096,
        }

    def send_tensor(self, tensor: Any) -> None:
        if self.send_tensor_exc is not None:
            raise self.send_tensor_exc
        self.sent_tensors.append(tensor)

    @property
    def transport_mode(self) -> str:
        return "zmq_mooncake"


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


class _InitCleanupPolicyClientStub:
    def __init__(self, config: Any) -> None:
        del config
        self.initialize_session_calls = 0
        self.disconnect_calls = 0
        self.close_calls = 0
        self.session_initialized = False
        self._transport_connected = False

    def initialize_session(self, num_envs: int, requested_action_mode: str) -> dict[str, Any]:
        del num_envs, requested_action_mode
        self.initialize_session_calls += 1
        self.session_initialized = True
        self._transport_connected = True
        return {"status": "success", "config": {}, "zmq_identity": b"dummy-zmq-id"}

    def disconnect(self) -> dict[str, Any]:
        self.disconnect_calls += 1
        return {"status": "disconnected"}

    def close(self) -> None:
        self.close_calls += 1
        self._transport_connected = False


class _HandshakeRejectPolicyClientStub(_InitCleanupPolicyClientStub):
    def initialize_session(self, num_envs: int, requested_action_mode: str) -> dict[str, Any]:
        del num_envs, requested_action_mode
        self.initialize_session_calls += 1
        self.session_initialized = False
        self._transport_connected = True
        self.close()
        return {
            "status": "rejected",
            "message": "Handshake rejected: transport_mode mismatch. client='zmq_mooncake', server='zmq'.",
            "error": "Handshake rejected: transport_mode mismatch. client='zmq_mooncake', server='zmq'.",
            "source": "handshake",
        }


class _BrokenProtocol(ActionProtocol):
    MODE: ClassVar[ActionMode] = ActionMode.CHUNK

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionProtocol:
        del data
        raise RuntimeError("protocol parse failed")

    def to_dict(self) -> dict[str, Any]:
        return {}


class _DummyClientSidePolicy(ClientSidePolicy):
    def get_action(self, env, observation):
        del env, observation
        return torch.zeros((1, 1), dtype=torch.float32)

    @staticmethod
    def add_args_to_parser(parser):
        return parser

    @staticmethod
    def from_args(args):
        del args
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


def _make_server(
    *,
    policy: Any | None = None,
    transport: Any | None = None,
    tensor_compression: str = "none",
    tensor_device: str | None = None,
) -> tuple[PolicyServer, _RecordingServerTransport]:
    actual_transport = transport or _RecordingServerTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=policy or _FallbackPolicy(),
        transport=actual_transport,
        host="127.0.0.1",
        port=0,
        tensor_compression=tensor_compression,
        tensor_device=tensor_device,
    )
    return server, actual_transport


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
    client_id = b"bad-data"
    server._client_states[client_id] = ClientState.create(1)

    server._dispatch_single(client_id, {"endpoint": "get_action", "data": []})

    assert len(transport.sent[client_id]) == 1
    assert "Expected dict data" in transport.sent[client_id][0]["error"]


def test_single_dispatch_rejects_non_dict_observation_with_explicit_error() -> None:
    server, transport = _make_server()
    client_id = b"bad-observation"
    server._client_states[client_id] = ClientState.create(1)

    server._dispatch_single(client_id, {"endpoint": "get_action", "data": {"observation": []}})

    assert len(transport.sent[client_id]) == 1
    assert "Expected dict observation" in transport.sent[client_id][0]["error"]


def test_get_init_info_failure_does_not_leave_half_initialized_session() -> None:
    server, transport = _make_server(policy=_FailingInitPolicy())
    client_id = b"init-fail"

    server._dispatch_single(
        client_id,
        {
            "endpoint": "get_init_info",
            "data": {
                "requested_action_mode": "chunk",
                "num_envs": 1,
                "transport_mode": "zmq",
                "tensor_compression": "none",
            },
        },
    )

    assert len(transport.sent[client_id]) == 1
    assert "injected get_init_info failure" in transport.sent[client_id][0]["error"]
    assert client_id not in server._client_states
    assert client_id not in server._last_seen


def test_get_init_info_returns_zmq_identity_and_creates_active_session() -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())
    client_id = b"session-init"

    response = server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=2,
        transport_mode="zmq",
        tensor_compression="none",
    )

    assert response["zmq_identity"] == client_id
    assert response["num_envs"] == 2
    assert client_id in server._client_states


def test_get_init_info_rejects_duplicate_live_zmq_identity() -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())
    client_id = b"already-live"

    server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq",
        tensor_compression="none",
    )

    response = server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq",
        tensor_compression="none",
    )

    assert response["status"] == "rejected"
    assert response["source"] == "handshake"
    assert "already has an active session" in response["error"]


def test_get_init_info_rejects_transport_mismatch() -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())

    response = server._handle_get_init_info(
        b"transport-mismatch",
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq_mooncake",
        tensor_compression="none",
    )

    assert response["status"] == "rejected"
    assert response["source"] == "handshake"
    assert "transport_mode mismatch" in response["error"]
    assert response["client_transport_mode"] == "zmq_mooncake"
    assert response["server_transport_mode"] == "zmq"


def test_get_init_info_rejects_tensor_compression_mismatch() -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())

    response = server._handle_get_init_info(
        b"compression-mismatch",
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq",
        tensor_compression="nvcomp_lz4",
    )

    assert response["status"] == "rejected"
    assert response["source"] == "handshake"
    assert "tensor_compression mismatch" in response["error"]
    assert response["client_tensor_compression"] == "nvcomp_lz4"
    assert response["server_tensor_compression"] == "none"


def test_get_init_info_mooncake_returns_handshake_metadata() -> None:
    transport = _MooncakeServerTransport()
    server, _ = _make_server(
        policy=_SuccessInitPolicy(),
        transport=transport,
        tensor_compression="none",
        tensor_device="cuda",
    )

    response = server._handle_get_init_info(
        b"mooncake-client",
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq_mooncake",
        tensor_compression="none",
    )

    assert response["mooncake_protocol"] == "rdma"
    assert response["mooncake_server_session_id"] == "server-session"
    assert transport.handshake_calls == 1


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
                "transport_mode": "zmq",
                "tensor_compression": "none",
            },
        },
    )

    assert client_id not in server._client_states
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


def test_gr00t_single_request_mixed_instructions_raise_explicit_error() -> None:
    gr00t_module = pytest.importorskip("isaaclab_arena_gr00t.policy.gr00t_remote_policy")
    policy = object.__new__(gr00t_module.Gr00tRemoteServerSidePolicy)
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


def test_policy_client_connects_transport_lazily_once() -> None:
    transport = _ClientTransportStub(
        recv_payload=[
            MessageSerializer.to_bytes({"status": "alive"}),
            MessageSerializer.to_bytes({"status": "alive"}),
        ]
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    client.call_endpoint("ping", requires_input=False)
    client.call_endpoint("ping", requires_input=False)

    assert transport.connect_calls == 1


def test_policy_client_initialize_session_caches_identity_and_request_shape() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes({"status": "ok", "zmq_identity": b"client-1"})
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    client.initialize_session(num_envs=2, requested_action_mode="velocity")

    request = MessageSerializer.from_bytes(transport.sent_payloads[0])
    assert request["data"]["requested_action_mode"] == "velocity"
    assert request["data"]["num_envs"] == 2
    assert request["data"]["transport_mode"] == "zmq"
    assert request["data"]["tensor_compression"] == "none"
    assert transport.cached_identities == [b"client-1"]


def test_policy_client_initialize_session_handshake_rejection_closes_transport() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "rejected",
                "error": (
                    "Handshake rejected: transport_mode mismatch. "
                    "client='zmq_mooncake', server='zmq'. "
                    "Adjust the client config to match the server, or restart against a server configured "
                    "for the requested mode."
                ),
                "source": "handshake",
                "client_transport_mode": "zmq_mooncake",
                "server_transport_mode": "zmq",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    response = client.initialize_session(num_envs=1, requested_action_mode="chunk")

    assert response["status"] == "rejected"
    assert "transport_mode mismatch" in response["error"]
    assert response["message"] == response["error"]
    assert transport.close_calls == 1
    assert client._transport_connected is False
    assert client._session_initialized is False


def test_policy_client_initialize_session_mooncake_connects_backend() -> None:
    transport = _MooncakeClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "ok",
                "zmq_identity": b"client-1",
                "mooncake_protocol": "rdma",
                "mooncake_server_session_id": "server-session",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(
            host="127.0.0.1",
            port=5555,
            timeout_ms=100,
            transport_mode="zmq_mooncake",
        ),
        transport,
    )

    client.initialize_session(num_envs=1, requested_action_mode="chunk")

    assert transport.connect_calls == 1
    assert transport.connect_backend_calls == 1


def test_policy_client_rejects_cpu_tensor_device_for_nvcomp() -> None:
    transport = _ClientTransportStub(recv_payload=MessageSerializer.to_bytes({"status": "ok"}))
    with pytest.raises(RuntimeError, match="requires policy_device to resolve to a CUDA device"):
        PolicyClient._from_transport_for_testing(
            RemotePolicyConfig(
                host="127.0.0.1",
                port=5555,
                timeout_ms=100,
                compression="nvcomp_lz4",
            ),
            transport,
            tensor_device="cpu",
        )


def test_policy_client_rejects_cpu_tensor_device_for_mooncake() -> None:
    transport = _MooncakeClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "status": "ok",
                "zmq_identity": b"client-1",
                "mooncake_protocol": "rdma",
                "mooncake_server_session_id": "server-session",
            }
        )
    )
    with pytest.raises(RuntimeError, match="requires policy_device to resolve to a CUDA device"):
        PolicyClient._from_transport_for_testing(
            RemotePolicyConfig(
                host="127.0.0.1",
                port=5555,
                timeout_ms=100,
                transport_mode="zmq_mooncake",
            ),
            transport,
            tensor_device="cpu",
        )


def test_server_rejects_missing_tensor_device_for_nvcomp(monkeypatch) -> None:
    monkeypatch.setattr(PolicyServer, "_has_nvcomp_runtime", staticmethod(lambda: True))

    with pytest.raises(RuntimeError, match="requires policy_device to resolve to a CUDA device"):
        PolicyServer._from_transport_for_testing(
            policy=_SuccessInitPolicy(),
            transport=_RecordingServerTransport(),
            host="127.0.0.1",
            port=0,
            tensor_compression="nvcomp_lz4",
            tensor_device=None,
        )


def test_server_inline_nvcomp_uses_policy_device(monkeypatch) -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())
    server._tensor_device = "cuda:1"

    class _FakeCompressedTensor:
        def clone(self):
            return self

        def to(self, *, device: str):
            return types.SimpleNamespace(device=device)

    class _FakeRawTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

    calls: dict[str, Any] = {}

    monkeypatch.setattr(torch, "frombuffer", lambda *args, **kwargs: _FakeCompressedTensor())

    from isaaclab_arena.remote_policy.compression import gpu_compression as gpu_compression_module

    def _fake_gpu_decompress(tensor: Any, nbytes: int) -> Any:
        calls["device"] = tensor.device
        calls["nbytes"] = nbytes
        return _FakeRawTensor()

    monkeypatch.setattr(gpu_compression_module, "gpu_decompress", _fake_gpu_decompress)

    server._decompress_inline_nvcomp_payload(b"\x00\x01", 2)

    assert calls["device"] == "cuda:1"
    assert calls["nbytes"] == 2


def test_policy_client_initialize_session_mooncake_connect_timeout_best_effort_disconnect() -> None:
    transport = _MooncakeClientTransportStub(
        recv_payload=[
            MessageSerializer.to_bytes(
                {
                    "status": "ok",
                    "zmq_identity": b"client-1",
                    "mooncake_protocol": "rdma",
                    "mooncake_server_session_id": "server-session",
                }
            ),
            MessageSerializer.to_bytes({"status": "disconnected"}),
        ],
        connect_backend_exc=TimeoutError("mooncake connect timeout"),
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(
            host="127.0.0.1",
            port=5555,
            timeout_ms=100,
            transport_mode="zmq_mooncake",
        ),
        transport,
    )

    with pytest.raises(TransportTimeoutError) as exc_info:
        client.initialize_session(num_envs=1, requested_action_mode="chunk")

    assert exc_info.value.source == "mooncake_connect"
    assert exc_info.value.must_reconnect is True
    assert len(transport.sent_payloads) == 2
    second_request = MessageSerializer.from_bytes(transport.sent_payloads[1])
    assert second_request["endpoint"] == "disconnect"


def test_policy_client_get_action_zmq_inlines_tensor_blob() -> None:
    transport = _ClientTransportStub(recv_payload=MessageSerializer.to_bytes({"status": "ok"}))
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    response = client.get_action({"rgb": torch.zeros(4, dtype=torch.uint8)})

    assert response["status"] == "ok"
    request = MessageSerializer.from_bytes(transport.sent_payloads[0])
    assert "rgb" not in request["data"]["observation"]
    assert "inline_tensor_layout" in request["data"]
    assert "inline_tensor_payload" in request["data"]
    assert request["data"]["inline_tensor_compression"] == "none"


def test_policy_client_dedicated_tensor_transport_includes_tensor_source_info() -> None:
    transport = _MooncakeClientTransportStub(
        recv_payload=MessageSerializer.to_bytes({"status": "ok"})
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(
            host="127.0.0.1",
            port=5555,
            timeout_ms=100,
            transport_mode="zmq_mooncake",
        ),
        transport,
    )

    response = client._get_action_dedicated_tensor_transport(
        {"obs": np.zeros((1, 2), dtype=np.float32)},
        None,
        {"rgb": torch.zeros(4, dtype=torch.uint8, device="cpu")},
    )

    assert response["status"] == "ok"
    request = MessageSerializer.from_bytes(transport.sent_payloads[0])
    assert request["data"]["tensor_transport_info"]["session_id"] == "client-session"
    assert request["data"]["tensor_transport_info"]["buffer_ptr"] == 12345
    assert request["data"]["tensor_transport_info"]["buffer_bytes"] == 4096


def test_prepare_tensor_payload_materializes_mixed_cuda_devices_to_target() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("At least two CUDA devices are required for mixed-device validation")

    codec = TensorPayloadCodec(compression="none")

    payload = codec.prepare_tensor_payload(
        {
            "gpu0": torch.zeros(4, dtype=torch.uint8, device="cuda:0"),
            "gpu1": torch.zeros(4, dtype=torch.uint8, device="cuda:1"),
        },
        target_device="cuda:0",
    )

    assert payload.payload_tensor.is_cuda
    assert str(payload.payload_tensor.device) == "cuda:0"


def test_prepare_tensor_payload_nvcomp_materializes_cpu_tensor_to_cuda(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for nvcomp materialization test")

    from isaaclab_arena.remote_policy.compression import gpu_compression as gpu_compression_module

    monkeypatch.setattr(gpu_compression_module, "gpu_compress", lambda tensor: tensor)
    codec = TensorPayloadCodec(compression="nvcomp_lz4")

    payload = codec.prepare_tensor_payload(
        {"rgb": torch.zeros(4, dtype=torch.uint8)},
        target_device="cuda:0",
    )

    assert payload.compression == "nvcomp_lz4"
    assert payload.payload_tensor.is_cuda


def test_prepare_tensor_payload_nvcomp_materializes_mixed_devices_to_one_cuda(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mixed-device nvcomp materialization test")

    from isaaclab_arena.remote_policy.compression import gpu_compression as gpu_compression_module

    monkeypatch.setattr(gpu_compression_module, "gpu_compress", lambda tensor: tensor)
    codec = TensorPayloadCodec(compression="nvcomp_lz4")

    payload = codec.prepare_tensor_payload(
        {
            "cpu_obs": torch.zeros(4, dtype=torch.uint8),
            "gpu_obs": torch.zeros(4, dtype=torch.uint8, device="cuda"),
        },
        target_device="cuda:0",
    )

    assert payload.compression == "nvcomp_lz4"
    assert payload.payload_tensor.is_cuda
    assert payload.original_nbytes == 8


def test_prepare_tensor_payload_returns_device_tensor_and_caller_can_convert_to_bytes() -> None:
    codec = TensorPayloadCodec(compression="none")

    payload = codec.prepare_tensor_payload(
        {"rgb": torch.zeros(4, dtype=torch.uint8)},
        target_device="cpu",
    )

    assert payload.payload_tensor.device.type == "cpu"
    assert payload.payload_tensor.detach().cpu().numpy().tobytes() == b"\x00\x00\x00\x00"


def test_has_nvcomp_requires_cuda_runtime(monkeypatch) -> None:
    from isaaclab_arena.remote_policy.compression import gpu_compression as gpu_compression_module

    monkeypatch.setattr(gpu_compression_module, "_ensure_nvcomp", lambda: True)
    monkeypatch.setattr(gpu_compression_module.torch.cuda, "is_available", lambda: False)

    assert gpu_compression_module.has_nvcomp() is False


def test_handle_get_action_rejects_has_tensor_on_pure_zmq_session() -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())
    client_id = b"zmq-has-tensor"
    server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq",
        tensor_compression="none",
    )

    with pytest.raises(RuntimeError, match="has_tensor=True is not allowed on transport_mode='zmq'"):
        server._handle_get_action(
            client_id,
            observation={"obs": np.zeros((1, 2), dtype=np.float32)},
            has_tensor=True,
            tensor_layout=[{
                "key": "obs",
                "shape": [1, 2],
                "dtype": "torch.float32",
                "offset": 0,
                "nbytes": 8,
            }],
            tensor_nbytes=8,
            tensor_original_nbytes=8,
            tensor_compression="none",
        )


def test_handle_get_action_rejects_tensor_compression_mismatch() -> None:
    transport = _MooncakeServerTransport()
    server, _ = _make_server(
        policy=_SuccessInitPolicy(),
        transport=transport,
        tensor_compression="none",
        tensor_device="cuda",
    )
    client_id = b"mooncake-no-nvcomp"
    server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq_mooncake",
        tensor_compression="none",
    )

    with pytest.raises(RuntimeError, match="dedicated tensor payload compression does not match the active session mode"):
        server._handle_get_action(
            client_id,
            observation={"obs": np.zeros((1, 2), dtype=np.float32)},
            has_tensor=True,
            tensor_layout=[{
                "key": "obs",
                "shape": [1, 2],
                "dtype": "torch.float32",
                "offset": 0,
                "nbytes": 8,
            }],
            tensor_nbytes=8,
            tensor_original_nbytes=8,
            tensor_compression="nvcomp_lz4",
            tensor_transport_info={
                "session_id": "client-session",
                "buffer_ptr": 12345,
                "buffer_bytes": 4096,
            },
        )


def test_handle_get_action_reports_mooncake_recv_source() -> None:
    transport = _FailingMooncakeRecvTransport()
    server, _ = _make_server(
        policy=_SuccessInitPolicy(),
        transport=transport,
        tensor_compression="none",
        tensor_device="cuda",
    )
    client_id = b"mooncake-recv-timeout"
    server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq_mooncake",
        tensor_compression="none",
    )

    with pytest.raises(ServerTransportTimeoutError) as exc_info:
        server._handle_get_action(
            client_id,
            observation={"obs": np.zeros((1, 2), dtype=np.float32)},
            has_tensor=True,
            tensor_layout=[{
                "key": "obs",
                "shape": [1, 2],
                "dtype": "torch.float32",
                "offset": 0,
                "nbytes": 8,
            }],
            tensor_nbytes=8,
            tensor_original_nbytes=8,
            tensor_compression="none",
            tensor_transport_info={
                "session_id": "client-session",
                "buffer_ptr": 12345,
                "buffer_bytes": 4096,
            },
        )

    assert exc_info.value.source == "mooncake_recv"


def test_handle_get_action_unpacks_inline_tensor_payload_for_zmq() -> None:
    server, _ = _make_server(policy=_SuccessInitPolicy())
    client_id = b"zmq-inline"
    server._handle_get_init_info(
        client_id,
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq",
        tensor_compression="none",
    )

    tensor_bytes = torch.arange(8, dtype=torch.uint8).numpy().tobytes()
    response = server._handle_get_action(
        client_id,
        observation={},
        inline_tensor_layout=[{
            "key": "obs",
            "shape": [1, 2],
            "dtype": "torch.float32",
            "offset": 0,
            "nbytes": 8,
        }],
        inline_tensor_payload={"mime": None, "data": tensor_bytes},
        inline_tensor_original_nbytes=8,
        inline_tensor_compression="none",
    )

    assert "action" in response


def test_policy_client_missing_client_state_is_reported_as_session_invalid() -> None:
    transport = _ClientTransportStub(
        recv_payload=MessageSerializer.to_bytes(
            {
                "error": (
                    "No active session for deadbeef on endpoint='get_action'. "
                    "Client must call initialize_session() to start a new session."
                ),
                "source": "session_state",
            }
        )
    )
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(RuntimeError, match="No active session") as exc_info:
        client.call_endpoint(
            "get_action",
            data={"observation": {"obs": np.zeros((1, 2), dtype=np.float32)}},
            requires_input=True,
        )

    assert "initialize_session() to start a new session" in str(exc_info.value)


def test_policy_client_start_new_session_reuses_last_action_mode() -> None:
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

    client.initialize_session(num_envs=2, requested_action_mode="velocity")
    client.start_new_session()

    first = MessageSerializer.from_bytes(transport.sent_payloads[0])
    second = MessageSerializer.from_bytes(transport.sent_payloads[1])
    assert first["data"]["requested_action_mode"] == "velocity"
    assert second["data"]["requested_action_mode"] == "velocity"
    assert transport.reset_identity_calls == 1
    assert transport.cached_identities == [b"zmq-id-1", b"zmq-id-2"]


def test_policy_client_initialize_session_requires_server_zmq_identity() -> None:
    transport = _ClientTransportStub(recv_payload=MessageSerializer.to_bytes({"status": "ok"}))
    client = PolicyClient._from_transport_for_testing(
        RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
        transport,
    )

    with pytest.raises(RuntimeError, match="missing bytes 'zmq_identity'"):
        client.initialize_session(num_envs=1, requested_action_mode="chunk")


def test_shutdown_remote_kill_rejected_falls_back_to_disconnect() -> None:
    policy = object.__new__(ClientSidePolicy)
    policy._client = _ShutdownClientStub({"status": "rejected", "reason": "remote kill disabled"})

    with pytest.warns(DeprecationWarning):
        policy.shutdown_remote(kill_server=True)

    assert policy._client.kill_calls == 1
    assert policy._client.disconnect_calls == 1
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
        lambda *args, **kwargs: stub,
    )

    with pytest.raises(RuntimeError, match="protocol parse failed"):
        _DummyClientSidePolicy(
            config=None,
            remote_config=RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
            protocol_cls=_BrokenProtocol,
            num_envs=1,
        )

    assert stub.initialize_session_calls == 1
    assert stub.disconnect_calls == 1
    assert stub.close_calls == 1


def test_client_side_policy_handshake_rejection_closes_without_disconnect(monkeypatch) -> None:
    stub = _HandshakeRejectPolicyClientStub(config=None)
    monkeypatch.setattr(
        "isaaclab_arena.policy.client_side_policy.PolicyClient",
        lambda *args, **kwargs: stub,
    )

    with pytest.raises(RuntimeError, match="transport_mode mismatch"):
        _DummyClientSidePolicy(
            config=None,
            remote_config=RemotePolicyConfig(host="127.0.0.1", port=5555, timeout_ms=100),
            protocol_cls=_BrokenProtocol,
            num_envs=1,
        )

    assert stub.initialize_session_calls == 1
    assert stub.disconnect_calls == 0
    assert stub.close_calls == 1
