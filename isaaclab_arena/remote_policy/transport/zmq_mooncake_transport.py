# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ZMQ + Mooncake hybrid transport.

ZMQ remains the control plane. Mooncake is only used for the dedicated tensor
path after ``get_init_info`` succeeds and returns the backend's static session
metadata.

The tensor path is server-side pull:

1. The client initializes its Mooncake engine right after the handshake.
2. For ``get_action`` with CUDA tensors, the client stages the flattened bytes
   into a registered local buffer and sends the source metadata over ZMQ.
3. The server receives the control message, then pulls from that source buffer
   into a per-client receive buffer.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from isaaclab_arena.remote_policy.profiling import nvtx_range

from .base import ClientTransport, ServerTransport
from .zmq_transport import ZmqClientTransport, ZmqServerTransport

if TYPE_CHECKING:
    import torch


def _ensure_mooncake_transfer_engine():
    try:
        from mooncake.engine import TransferEngine
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Mooncake transport requires `mooncake-transfer-engine` and its runtime "
            "shared-library dependencies to be importable in this environment."
        ) from exc
    return TransferEngine


def _build_session_id(local_hostname: str, rpc_port: int) -> str:
    if ":" in local_hostname:
        return local_hostname
    return f"{local_hostname}:{rpc_port}"


def _register_buffer_if_needed(engine: Any, protocol: str, pointer: int, nbytes: int, *, force: bool) -> None:
    if protocol != "rdma" and not force:
        return
    ret = engine.register_memory(pointer, nbytes)
    if ret != 0:
        raise RuntimeError(f"Mooncake register_memory failed with code={ret}")


def _unregister_buffer_if_needed(engine: Any, protocol: str, pointer: int, *, force: bool) -> None:
    if protocol != "rdma" and not force:
        return
    ret = engine.unregister_memory(pointer)
    if ret != 0:
        raise RuntimeError(f"Mooncake unregister_memory failed with code={ret}")


def _bind_cuda_context(device: str | None = None, tensor: "torch.Tensor | None" = None) -> None:
    import torch

    if tensor is not None:
        if not tensor.is_cuda:
            return
        index = tensor.device.index
        if index is None:
            index = torch.cuda.current_device()
        torch.cuda.set_device(index)
        return

    if device is None:
        return
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return
    if parsed.index is None:
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        torch.cuda.set_device(parsed.index)


def _current_process_cuda_device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Mooncake CUDA tensor path requires torch.cuda.is_available() to be True.")
    return f"cuda:{torch.cuda.current_device()}"


class ZmqMooncakeServerTransport(ServerTransport):
    """Hybrid ZMQ (control) + Mooncake (tensor pull) server transport."""

    def __init__(
        self,
        timeout_ms: int = 15000,
        *,
        local_hostname: str | None,
        metadata_server: str = "P2PHANDSHAKE",
        protocol: str = "rdma",
        device_name: str = "",
        buffer_bytes: int = 64 * 1024 * 1024,
        tensor_device: str | None = None,
        cuda_device_override: str | None = None,
        force_register: bool = True,
    ) -> None:
        self._zmq = ZmqServerTransport(timeout_ms=timeout_ms)
        self._local_hostname = local_hostname
        self._metadata_server = metadata_server
        self._protocol = protocol
        self._device_name = device_name
        self._buffer_bytes = buffer_bytes
        self._tensor_device = tensor_device
        self._cuda_device_override = cuda_device_override
        self._force_register = force_register

        self._engine = None
        self._session_id: str | None = None
        self._recv_buffers: dict[bytes, torch.Tensor] = {}
        self._recv_buffer_ptrs: dict[bytes, int] = {}
        self._recv_buffer_capacities: dict[bytes, int] = {}
        self._pending_sources: dict[bytes, dict[str, Any]] = {}

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        if not self._local_hostname:
            raise RuntimeError(
                "Mooncake server transport requires a non-empty local hostname. "
                "Pass --mooncake_local_hostname or configure RemotePolicyConfig.mooncake.local_hostname."
            )

        TransferEngine = _ensure_mooncake_transfer_engine()
        engine = TransferEngine()
        ret = engine.initialize(
            self._local_hostname,
            self._metadata_server,
            self._protocol,
            self._device_name,
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed with code={ret}")
        self._engine = engine
        self._session_id = _build_session_id(self._local_hostname, int(engine.get_rpc_port()))

    def _resolve_recv_device(self) -> str:
        if self._tensor_device:
            return self._tensor_device
        if self._cuda_device_override:
            return self._cuda_device_override
        return _current_process_cuda_device()

    def _release_client_buffer(self, client_id: bytes) -> None:
        buffer_ptr = self._recv_buffer_ptrs.pop(client_id, None)
        self._recv_buffer_capacities.pop(client_id, None)
        if buffer_ptr is not None and self._engine is not None:
            try:
                _unregister_buffer_if_needed(
                    self._engine,
                    self._protocol,
                    buffer_ptr,
                    force=self._force_register,
                )
            except Exception as exc:
                warnings.warn(
                    f"[ZmqMooncakeServerTransport] Failed to unregister client buffer for {client_id.hex()[:8]}: {exc}"
                )
        self._recv_buffers.pop(client_id, None)

    def _ensure_client_buffer_capacity(self, client_id: bytes, required_bytes: int) -> "torch.Tensor":
        import torch

        existing = self._recv_buffers.get(client_id)
        existing_capacity = self._recv_buffer_capacities.get(client_id, 0)
        if existing is not None and existing_capacity >= required_bytes:
            return existing

        self._ensure_engine()
        self._release_client_buffer(client_id)
        device = self._resolve_recv_device()
        _bind_cuda_context(device=device)
        capacity = max(self._buffer_bytes, required_bytes, 1)
        buffer = torch.empty(capacity, dtype=torch.uint8, device=device)
        pointer = int(buffer.data_ptr())
        _register_buffer_if_needed(
            self._engine,
            self._protocol,
            pointer,
            capacity,
            force=self._force_register,
        )
        self._recv_buffers[client_id] = buffer
        self._recv_buffer_ptrs[client_id] = pointer
        self._recv_buffer_capacities[client_id] = capacity
        return buffer

    def bind(self, endpoint: str) -> None:
        self._zmq.bind(endpoint)

    def recv(self) -> tuple[bytes, bytes]:
        return self._zmq.recv()

    def send(self, client_id: bytes, payload: bytes) -> None:
        self._zmq.send(client_id, payload)

    def _build_handshake_metadata(self, client_id: bytes) -> dict[str, Any]:
        self._ensure_engine()
        return {
            "mooncake_protocol": self._protocol,
            "mooncake_server_session_id": self._session_id,
        }

    def get_handshake_metadata(self, client_id: bytes) -> dict[str, Any]:
        return self._build_handshake_metadata(client_id)

    def prepare_recv_tensor(self, client_id: bytes, source_info: dict[str, Any]) -> None:
        session_id = source_info.get("session_id")
        buffer_ptr = source_info.get("buffer_ptr")
        buffer_bytes = source_info.get("buffer_bytes")
        if not isinstance(session_id, str):
            raise RuntimeError("Mooncake tensor source info is missing string 'session_id'.")
        if not isinstance(buffer_ptr, int):
            raise RuntimeError("Mooncake tensor source info is missing integer 'buffer_ptr'.")
        if not isinstance(buffer_bytes, int) or buffer_bytes <= 0:
            raise RuntimeError("Mooncake tensor source info is missing positive integer 'buffer_bytes'.")

        self._pending_sources[client_id] = {
            "session_id": session_id,
            "buffer_ptr": buffer_ptr,
            "buffer_bytes": buffer_bytes,
        }

    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: "torch.Tensor | None" = None) -> "torch.Tensor":
        import torch

        self._ensure_engine()
        source = self._pending_sources.get(client_id)
        if source is None:
            raise RuntimeError(f"No pending Mooncake tensor source for client {client_id.hex()[:8]}")
        if nbytes > int(source["buffer_bytes"]):
            raise RuntimeError(
                f"Mooncake tensor payload size {nbytes} exceeds client buffer_bytes={source['buffer_bytes']}"
            )
        if buffer is None:
            buffer = self._ensure_client_buffer_capacity(client_id, nbytes)

        _bind_cuda_context(tensor=buffer)
        # Use transfer_sync_read() for all protocols and buffer types.
        # transfer_read_on_cuda() is not reliably supported across all
        # Mooncake backends (tcp hangs, nvlink_intra untested). The sync
        # API works for both CPU and CUDA destination buffers on every
        # protocol we have validated (tcp, rdma).
        with nvtx_range("mooncake.transfer_sync_read"):
            ret = self._engine.transfer_sync_read(
                source["session_id"],
                int(buffer.data_ptr()),
                int(source["buffer_ptr"]),
                nbytes,
            )
        if ret != 0:
            raise RuntimeError(f"Mooncake transfer_sync_read failed with code={ret}")

        return buffer[:nbytes]

    def disconnect_client(self, client_id: bytes) -> None:
        self._release_client_buffer(client_id)
        self._pending_sources.pop(client_id, None)

    def close(self) -> None:
        for client_id in list(self._recv_buffer_ptrs):
            self.disconnect_client(client_id)
        self._zmq.close()
        self._engine = None
        self._session_id = None

    @property
    def buffer_bytes(self) -> int:
        return self._buffer_bytes

    @property
    def transport_mode(self) -> str:
        return "zmq_mooncake"


class ZmqMooncakeClientTransport(ClientTransport):
    """Hybrid ZMQ (control) + Mooncake (grow-on-demand send buffer) client transport."""

    def __init__(
        self,
        timeout_ms: int = 15000,
        *,
        local_hostname: str | None,
        metadata_server: str = "P2PHANDSHAKE",
        protocol: str = "rdma",
        device_name: str = "",
        buffer_bytes: int = 64 * 1024 * 1024,
        tensor_device: str | None = None,
        cuda_device_override: str | None = None,
        force_register: bool = True,
    ) -> None:
        self._zmq = ZmqClientTransport(timeout_ms=timeout_ms)
        self._local_hostname = local_hostname
        self._metadata_server = metadata_server
        self._protocol = protocol
        self._device_name = device_name
        self._buffer_bytes = buffer_bytes
        self._tensor_device = tensor_device
        self._cuda_device_override = cuda_device_override
        self._force_register = force_register

        self._engine = None
        self._session_id: str | None = None
        self._send_buffers: dict[str, torch.Tensor] = {}
        self._send_buffer_ptrs: dict[str, int] = {}
        self._send_buffer_capacities: dict[str, int] = {}
        self._last_tensor_source_info: dict[str, Any] | None = None

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        if not self._local_hostname:
            raise RuntimeError(
                "Mooncake client transport requires a non-empty local hostname. "
                "Pass --mooncake_local_hostname or configure RemotePolicyConfig.mooncake.local_hostname."
            )

        TransferEngine = _ensure_mooncake_transfer_engine()
        engine = TransferEngine()
        ret = engine.initialize(
            self._local_hostname,
            self._metadata_server,
            self._protocol,
            self._device_name,
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed with code={ret}")
        self._engine = engine
        self._session_id = _build_session_id(self._local_hostname, int(engine.get_rpc_port()))

    def _resolve_send_device(self, tensor: "torch.Tensor") -> str:
        if self._tensor_device:
            return self._tensor_device
        if self._cuda_device_override:
            return self._cuda_device_override
        return str(tensor.device)

    def _release_send_buffer_for_device(self, device: str) -> None:
        pointer = self._send_buffer_ptrs.pop(device, None)
        self._send_buffer_capacities.pop(device, None)
        if pointer is not None and self._engine is not None:
            try:
                _unregister_buffer_if_needed(
                    self._engine,
                    self._protocol,
                    pointer,
                    force=self._force_register,
                )
            except Exception as exc:
                warnings.warn(
                    f"[ZmqMooncakeClientTransport] Failed to unregister send buffer for device {device!r}: {exc}"
                )
        self._send_buffers.pop(device, None)

    def _ensure_send_buffer_capacity(self, device: str, required_bytes: int) -> None:
        import torch

        existing_capacity = self._send_buffer_capacities.get(device, 0)
        if device in self._send_buffers and device in self._send_buffer_ptrs and existing_capacity >= required_bytes:
            return
        self._ensure_engine()
        self._release_send_buffer_for_device(device)
        _bind_cuda_context(device=device)
        capacity = max(self._buffer_bytes, required_bytes, 1)
        buffer = torch.empty(capacity, dtype=torch.uint8, device=device)
        pointer = int(buffer.data_ptr())
        _register_buffer_if_needed(
            self._engine,
            self._protocol,
            pointer,
            capacity,
            force=self._force_register,
        )
        self._send_buffers[device] = buffer
        self._send_buffer_ptrs[device] = pointer
        self._send_buffer_capacities[device] = capacity

    def connect(self, endpoint: str) -> None:
        self._zmq.connect(endpoint)

    def connect_mooncake(self) -> None:
        self._ensure_engine()

    def connect_comm_backend(
        self,
        *,
        handshake_response: dict,
        server_host: str,
        zmq_identity: bytes | None,
    ) -> None:
        del handshake_response, server_host, zmq_identity
        self.connect_mooncake()

    def cache_identity(self, identity: bytes) -> None:
        self._zmq.cache_identity(identity)

    def reset_identity(self) -> None:
        self._zmq.reset_identity()

    def send(self, payload: bytes) -> None:
        self._zmq.send(payload)

    def recv(self) -> bytes:
        return self._zmq.recv()

    def tensor_source_info(self) -> dict[str, Any]:
        if self._session_id is None or self._last_tensor_source_info is None:
            raise RuntimeError("Mooncake tensor transport is not connected")
        return dict(self._last_tensor_source_info)

    def send_tensor(self, tensor: "torch.Tensor") -> None:
        import torch

        if not tensor.is_cuda:
            raise RuntimeError("Mooncake tensor path currently expects CUDA tensors.")

        device = self._resolve_send_device(tensor)
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        nbytes = flat.numel()
        self._ensure_send_buffer_capacity(device, nbytes)
        send_buffer = self._send_buffers[device]
        send_buffer_ptr = self._send_buffer_ptrs[device]
        send_buffer_capacity = self._send_buffer_capacities[device]

        _bind_cuda_context(tensor=flat)
        with nvtx_range("mooncake.copy_into_send_buffer"):
            send_buffer[:nbytes].copy_(flat)
        if flat.is_cuda:
            torch.cuda.current_stream(device=flat.device).synchronize()
        self._last_tensor_source_info = {
            "session_id": self._session_id,
            "buffer_ptr": send_buffer_ptr,
            "buffer_bytes": send_buffer_capacity,
        }

    def recv_tensor(self, nbytes: int, buffer: "torch.Tensor | None" = None) -> "torch.Tensor":
        raise NotImplementedError("Mooncake client transport does not implement recv_tensor in this prototype")

    def reset_tensor_transport(self) -> None:
        for device in list(self._send_buffer_ptrs):
            self._release_send_buffer_for_device(device)
        self._send_buffers = {}
        self._send_buffer_ptrs = {}
        self._send_buffer_capacities = {}
        self._last_tensor_source_info = None
        self._engine = None
        self._session_id = None

    def rebuild(self) -> None:
        self._zmq.rebuild()

    def reset_comm_backend(self) -> None:
        self.reset_tensor_transport()

    def close(self) -> None:
        self.reset_tensor_transport()
        self._zmq.close()

    @property
    def transport_mode(self) -> str:
        return "zmq_mooncake"
