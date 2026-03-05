# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ZMQ + UCX hybrid transport.

Control messages (handshake, metadata, small payloads) travel over ZMQ
ROUTER/DEALER.  Large GPU tensors are sent via UCX zero-copy (RDMA or
shared-memory), bypassing the CPU entirely.

Lifecycle:
  1. Client connects to the server's ZMQ ROUTER endpoint (same as ZMQ-only).
  2. During ``get_init_info``, if both sides advertise ``"zmq_ucx"`` capability,
     the server creates a UCX listener and returns the ``ucx_port`` in the
     response.
  3. The client connects a UCX endpoint to ``server_host:ucx_port``.
  4. For ``get_action`` calls with tensor data, the client sends metadata via
     ZMQ and the raw GPU tensor via UCX.  The server receives both, passes
     the GPU tensor directly to the policy.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import torch

from .base import ClientTransport, ServerTransport
from .zmq_transport import ZmqClientTransport, ZmqServerTransport


def _ensure_ucp():
    """Import ucp lazily; raise clear error if not installed."""
    try:
        import ucp
        return ucp
    except ImportError:
        raise ImportError(
            "ucx-py is required for UCX transport. Install with: pip install ucx-py-cu12"
        )


class ZmqUcxServerTransport(ServerTransport):
    """Hybrid ZMQ (control) + UCX (tensor) server transport.

    ZMQ is used for all control messages.  UCX is used for GPU tensor
    transfers when negotiated during ``get_init_info``.
    """

    @staticmethod
    def _ucx_key(client_id: bytes) -> bytes:
        """Normalize a ZMQ identity to the fixed 16-byte UCX key format.

        ZMQ ROUTER identities are typically 5 bytes, but the UCX handshake
        sends exactly 16 bytes.  This helper pads/truncates so that both
        the store (``_on_connect``) and the lookup (``recv_tensor``,
        ``send_tensor``, ``disconnect_client``) use the same key.
        """
        return client_id[:16].ljust(16, b"\x00")

    def __init__(
        self,
        timeout_ms: int = 15000,
        ucx_port: int | None = None,
    ) -> None:
        self._zmq = ZmqServerTransport(timeout_ms=timeout_ms)
        self._ucx_port = ucx_port
        self._ucx_listener = None
        self._ucx_endpoints: dict[bytes, Any] = {}  # client_id -> ucp.Endpoint
        self._ucx_loop: asyncio.AbstractEventLoop | None = None
        self._ucx_thread: threading.Thread | None = None

    def bind(self, endpoint: str) -> None:
        self._zmq.bind(endpoint)

    def recv(self) -> tuple[bytes, bytes]:
        return self._zmq.recv()

    def recv_with_timeout(self, timeout_ms: int) -> tuple[bytes, bytes]:
        return self._zmq.recv_with_timeout(timeout_ms)

    def send(self, client_id: bytes, payload: bytes) -> None:
        self._zmq.send(client_id, payload)

    # ---- UCX tensor path ----

    def start_ucx_listener(self, host: str = "0.0.0.0", port: int = 0) -> int:
        """Start the UCX listener in a background thread. Returns the bound port."""
        ucp = _ensure_ucp()

        loop = asyncio.new_event_loop()
        self._ucx_loop = loop
        ready_event = threading.Event()
        actual_port = [0]

        async def _listener_coro():
            async def _on_connect(ep):
                # Read the client_id (16 bytes, padded via _ucx_key) so we can map
                cid_raw = await ep.recv(16)
                key = ZmqUcxServerTransport._ucx_key(bytes(cid_raw))
                self._ucx_endpoints[key] = ep

            listener = ucp.create_listener(_on_connect, port=port)
            actual_port[0] = listener.port
            ready_event.set()
            # Keep the listener alive
            while self._ucx_listener is not None:
                await asyncio.sleep(0.1)

        def _run():
            asyncio.set_event_loop(loop)
            self._ucx_listener = True  # sentinel
            loop.run_until_complete(_listener_coro())

        self._ucx_thread = threading.Thread(target=_run, daemon=True)
        self._ucx_thread.start()
        ready_event.wait(timeout=10.0)
        self._ucx_port = actual_port[0]
        return self._ucx_port

    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: torch.Tensor | None = None) -> torch.Tensor:
        """Receive a GPU tensor from a client via UCX."""
        key = self._ucx_key(client_id)
        ep = self._ucx_endpoints.get(key)
        if ep is None:
            raise RuntimeError(f"No UCX endpoint for client {client_id.hex()[:8]}")

        if buffer is None:
            buffer = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

        future = asyncio.run_coroutine_threadsafe(ep.recv(buffer), self._ucx_loop)
        future.result(timeout=30.0)
        return buffer

    def send_tensor(self, client_id: bytes, tensor: torch.Tensor) -> None:
        """Send a GPU tensor to a client via UCX."""
        key = self._ucx_key(client_id)
        ep = self._ucx_endpoints.get(key)
        if ep is None:
            raise RuntimeError(f"No UCX endpoint for client {client_id.hex()[:8]}")

        future = asyncio.run_coroutine_threadsafe(ep.send(tensor), self._ucx_loop)
        future.result(timeout=30.0)

    def disconnect_client(self, client_id: bytes) -> None:
        """Close the UCX endpoint for a specific client."""
        key = self._ucx_key(client_id)
        ep = self._ucx_endpoints.pop(key, None)
        if ep is not None:
            try:
                ep.close()
            except Exception:
                pass

    @property
    def ucx_port(self) -> int | None:
        return self._ucx_port

    def close(self) -> None:
        self._ucx_listener = None
        for ep in self._ucx_endpoints.values():
            try:
                ep.close()
            except Exception:
                pass
        self._ucx_endpoints.clear()
        if self._ucx_loop is not None:
            self._ucx_loop.call_soon_threadsafe(self._ucx_loop.stop)
        self._zmq.close()

    @property
    def transport_mode(self) -> str:
        return "zmq_ucx"


class ZmqUcxClientTransport(ClientTransport):
    """Hybrid ZMQ (control) + UCX (tensor) client transport."""

    def __init__(self, timeout_ms: int = 15000) -> None:
        self._zmq = ZmqClientTransport(timeout_ms=timeout_ms)
        self._ucx_endpoint = None
        self._ucx_loop: asyncio.AbstractEventLoop | None = None
        self._ucx_thread: threading.Thread | None = None
        self._client_id: bytes | None = None

    def connect(self, endpoint: str) -> None:
        self._zmq.connect(endpoint)

    def send(self, payload: bytes) -> None:
        self._zmq.send(payload)

    def recv(self) -> bytes:
        return self._zmq.recv()

    # ---- UCX tensor path ----

    def connect_ucx(self, host: str, port: int, client_id: bytes) -> None:
        """Connect the UCX endpoint to the server's UCX listener."""
        ucp = _ensure_ucp()
        self._client_id = client_id

        loop = asyncio.new_event_loop()
        self._ucx_loop = loop
        ready_event = threading.Event()

        async def _connect_coro():
            ep = await ucp.create_endpoint(host, port)
            # Send our client_id so the server can map us
            await ep.send(client_id[:16].ljust(16, b"\x00"))
            self._ucx_endpoint = ep
            ready_event.set()
            # Keep loop alive
            while self._ucx_endpoint is not None:
                await asyncio.sleep(0.1)

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_connect_coro())

        self._ucx_thread = threading.Thread(target=_run, daemon=True)
        self._ucx_thread.start()
        ready_event.wait(timeout=10.0)
        if self._ucx_endpoint is None:
            raise RuntimeError(f"Failed to connect UCX endpoint to {host}:{port}")

    def send_tensor(self, tensor: torch.Tensor) -> None:
        """Send a GPU tensor to the server via UCX."""
        if self._ucx_endpoint is None:
            raise RuntimeError("UCX endpoint not connected")
        future = asyncio.run_coroutine_threadsafe(
            self._ucx_endpoint.send(tensor), self._ucx_loop
        )
        future.result(timeout=30.0)

    def recv_tensor(self, nbytes: int, buffer: torch.Tensor | None = None) -> torch.Tensor:
        """Receive a GPU tensor from the server via UCX."""
        if self._ucx_endpoint is None:
            raise RuntimeError("UCX endpoint not connected")
        if buffer is None:
            buffer = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        future = asyncio.run_coroutine_threadsafe(
            self._ucx_endpoint.recv(buffer), self._ucx_loop
        )
        future.result(timeout=30.0)
        return buffer

    def rebuild(self) -> None:
        self._zmq.rebuild()

    def close(self) -> None:
        if self._ucx_endpoint is not None:
            try:
                self._ucx_endpoint.close()
            except Exception:
                pass
            self._ucx_endpoint = None
        if self._ucx_loop is not None:
            self._ucx_loop.call_soon_threadsafe(self._ucx_loop.stop)
        self._zmq.close()

    @property
    def transport_mode(self) -> str:
        return "zmq_ucx"
