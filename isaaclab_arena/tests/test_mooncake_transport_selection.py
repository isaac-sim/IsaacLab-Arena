from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import pytest
import types

from isaaclab_arena.policy.client_side_policy import ClientSidePolicy
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.policy_client import PolicyClient
from isaaclab_arena.remote_policy.policy_server import PolicyServer
from isaaclab_arena.remote_policy.remote_policy_config import MooncakeTransportConfig, RemotePolicyConfig
from isaaclab_arena.remote_policy.remote_policy_server_runner import build_base_parser


class _SuccessInitPolicy:
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
        del client_id, nbytes, buffer
        self.recv_tensor_calls += 1
        raise AssertionError("recv_tensor should not be called when client state is missing")

    @property
    def transport_mode(self) -> str:
        return "zmq_mooncake"


class _ClientTransportStub:
    def __init__(self, *, recv_payload: Any = None, recv_exc: Exception | None = None) -> None:
        self.recv_payload = recv_payload
        self.recv_exc = recv_exc
        self.sent_payloads: list[bytes] = []
        self.rebuild_calls = 0
        self.cached_identities: list[bytes] = []
        self.reset_identity_calls = 0

    def connect(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def send(self, payload: bytes) -> None:
        self.sent_payloads.append(payload)

    def recv(self) -> bytes:
        if self.recv_exc is not None:
            raise self.recv_exc
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

    @property
    def transport_mode(self) -> str:
        return "zmq"


class _MooncakeClientTransportStub(_ClientTransportStub):
    def __init__(self, *, recv_payload: Any = None, recv_exc: Exception | None = None) -> None:
        super().__init__(recv_payload=recv_payload, recv_exc=recv_exc)
        self.connect_backend_calls = 0

    def connect_comm_backend(
        self,
        *,
        handshake_response: dict[str, Any],
        server_host: str,
        zmq_identity: bytes | None,
    ) -> None:
        del handshake_response, server_host, zmq_identity
        self.connect_backend_calls += 1

    @property
    def transport_mode(self) -> str:
        return "zmq_mooncake"


def test_client_parser_transport_choices_exclude_auto() -> None:
    parser = ClientSidePolicy.add_remote_args_to_parser(argparse.ArgumentParser())
    action = next(a for a in parser._actions if a.dest == "remote_transport_mode")
    assert action.choices == ["zmq", "zmq_mooncake"]


def test_server_parser_transport_choices_exclude_auto() -> None:
    parser = build_base_parser()
    action = next(a for a in parser._actions if a.dest == "transport_mode")
    assert action.choices == ["zmq", "zmq_mooncake"]


def test_client_explicit_mooncake_auto_detects_local_hostname(monkeypatch) -> None:
    monkeypatch.setattr(PolicyClient, "_has_mooncake_runtime", staticmethod(lambda: True))
    monkeypatch.setattr(
        "isaaclab_arena.remote_policy.policy_client.autodetect_local_hostname",
        lambda preferred_remote_host: "10.1.2.3",
    )

    transport = PolicyClient._create_default_transport(
        RemotePolicyConfig(
            host="10.9.8.7",
            port=5555,
            timeout_ms=100,
            transport_mode="zmq_mooncake",
        )
    )

    assert transport.transport_mode == "zmq_mooncake"
    assert transport._local_hostname == "10.1.2.3"
    transport.close()


def test_client_mainline_constructor_rejects_legacy_ucx() -> None:
    with pytest.raises(RuntimeError, match="legacy/debug path"):
        PolicyClient._create_default_transport(
            RemotePolicyConfig(
                host="127.0.0.1",
                port=5555,
                timeout_ms=100,
                transport_mode="zmq_ucx",
            )
        )


def test_server_explicit_mooncake_auto_detects_local_hostname(monkeypatch) -> None:
    monkeypatch.setattr(PolicyServer, "_has_mooncake_runtime", staticmethod(lambda: True))
    monkeypatch.setattr(
        "isaaclab_arena.remote_policy.policy_server.autodetect_local_hostname",
        lambda bind_host: "10.2.3.4",
    )

    transport = PolicyServer._create_default_transport(
        timeout_ms=100,
        bind_host="0.0.0.0",
        transport_mode="zmq_mooncake",
        tensor_device=None,
        mooncake_config=MooncakeTransportConfig(),
    )

    assert transport.transport_mode == "zmq_mooncake"
    assert transport._local_hostname == "10.2.3.4"
    transport.close()


def test_server_mainline_constructor_rejects_legacy_ucx() -> None:
    with pytest.raises(RuntimeError, match="legacy/debug path"):
        PolicyServer._create_default_transport(
            timeout_ms=100,
            bind_host="0.0.0.0",
            transport_mode="zmq_ucx",
            tensor_device=None,
            mooncake_config=MooncakeTransportConfig(),
        )


def test_server_mooncake_rejects_cpu_tensor_device() -> None:
    with pytest.raises(RuntimeError, match="requires policy_device to resolve to a CUDA device"):
        PolicyServer._from_transport_for_testing(
            policy=_SuccessInitPolicy(),
            transport=_MooncakeServerTransport(),
            host="127.0.0.1",
            port=0,
            tensor_device="cpu",
        )


def test_server_mooncake_requires_explicit_tensor_device() -> None:
    with pytest.raises(RuntimeError, match="requires policy_device to resolve to a CUDA device"):
        PolicyServer._from_transport_for_testing(
            policy=_SuccessInitPolicy(),
            transport=_MooncakeServerTransport(),
            host="127.0.0.1",
            port=0,
            tensor_device=None,
        )


def test_server_mooncake_get_init_info_returns_handshake_metadata() -> None:
    transport = _MooncakeServerTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=_SuccessInitPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
        tensor_device="cuda",
    )

    response = server._handle_get_init_info(
        b"client-1",
        requested_action_mode="chunk",
        num_envs=1,
        transport_mode="zmq_mooncake",
    )

    assert response["mooncake_protocol"] == "rdma"
    assert response["mooncake_server_session_id"] == "server-session"
    assert transport.handshake_calls == 1


def test_server_missing_client_state_rejects_before_mooncake_pull() -> None:
    transport = _MooncakeServerTransport()
    server = PolicyServer._from_transport_for_testing(
        policy=_SuccessInitPolicy(),
        transport=transport,
        host="127.0.0.1",
        port=0,
        tensor_device="cuda",
    )

    try:
        server._handle_get_action(
            b"missing-client",
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
            tensor_transport_info={
                "session_id": "client-session",
                "buffer_ptr": 12345,
                "buffer_bytes": 4096,
            },
        )
    except RuntimeError as exc:
        assert "Client must call initialize_session() to start a new session" in str(exc)
    else:
        raise AssertionError("Expected missing client state to raise before tensor pull")

    assert transport.prepare_recv_tensor_calls == 0
    assert transport.recv_tensor_calls == 0


def test_tcp_cuda_recv_uses_transfer_sync_read(monkeypatch) -> None:
    mooncake_module = pytest.importorskip("isaaclab_arena.remote_policy.transport.zmq_mooncake_transport")
    ZmqMooncakeServerTransport = mooncake_module.ZmqMooncakeServerTransport

    calls: list[tuple[str, tuple[Any, ...]]] = []

    class _FakeEngine:
        def transfer_sync_read(self, *args):
            calls.append(("sync", args))
            return 0

        def transfer_read_on_cuda(self, *args):
            calls.append(("cuda", args))

    class _FakeBuffer:
        is_cuda = True
        device = "cuda:0"

        def data_ptr(self):
            return 123456

        def __getitem__(self, item):
            del item
            return self

    monkeypatch.setattr(mooncake_module, "_bind_cuda_context", lambda **kwargs: None)
    monkeypatch.setattr(mooncake_module, "nvtx_range", lambda name: types.SimpleNamespace(__enter__=lambda self=None: None, __exit__=lambda self, exc_type, exc, tb: False))

    transport = object.__new__(ZmqMooncakeServerTransport)
    transport._protocol = "tcp"
    transport._engine = _FakeEngine()
    transport._pending_sources = {b"client": {"session_id": "session", "buffer_ptr": 987654, "buffer_bytes": 4096}}
    transport._ensure_engine = lambda: None

    buffer = _FakeBuffer()
    result = transport.recv_tensor(b"client", 1024, buffer=buffer)

    assert result is buffer
    assert calls == [("sync", ("session", 123456, 987654, 1024))]


def test_client_initialize_session_requires_mooncake_capable_transport() -> None:
    transport = _ClientTransportStub(
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

    with pytest.raises(RuntimeError, match="does not support connect_comm_backend"):
        client.initialize_session(num_envs=1, requested_action_mode="chunk")


def test_client_initialize_session_mooncake_connects_backend() -> None:
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

    response = client.initialize_session(num_envs=1, requested_action_mode="chunk")

    assert response["mooncake_server_session_id"] == "server-session"
    assert transport.connect_backend_calls == 1
