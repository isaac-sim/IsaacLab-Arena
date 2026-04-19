# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy/debug ZMQ + UCX hybrid transport.

This file is kept for explicit legacy or debugging scenarios. It is not part
of the simple mainline design, which centers on ``zmq`` and ``zmq_mooncake``.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import TYPE_CHECKING, Any

from .base import ClientTransport, ServerTransport
from .zmq_transport import ZmqClientTransport, ZmqServerTransport

if TYPE_CHECKING:
    import torch


def _bind_cuda_context(tensor: "torch.Tensor") -> None:
    """Ensure the calling thread has the CUDA context for this tensor's device."""
    if not tensor.is_cuda:
        return
    import torch

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)


def _ensure_ucp():
    """Import ucp lazily; raise clear error if not installed."""
    try:
        import ucp
        return ucp
    except ImportError:
        raise ImportError(
            "A CUDA-matched ucx-py wheel is required for UCX transport. "
            "Install the package matching your CUDA major version "
            "(for example `pip install ucx-py-cu12` for CUDA 12.x)."
        )


class ZmqUcxServerTransport(ServerTransport):
    """Hybrid ZMQ (control) + UCX (tensor) server transport.

    ZMQ is used for all control messages.  UCX is used for GPU tensor
    transfers when negotiated during ``get_init_info``.
    """

    @staticmethod
    def _ucx_key(client_id: bytes) -> bytes:
        """Use the exact ZMQ identity as the UCX endpoint lookup key."""
        return client_id

    def __init__(
        self,
        timeout_ms: int = 15000,
        ucx_port: int | None = None,
    ) -> None:
        self._zmq = ZmqServerTransport(timeout_ms=timeout_ms)
        self._timeout_s = max(timeout_ms, 1) / 1000.0
        self._ucx_port = ucx_port
        self._ucx_listener = None
        self._ucx_endpoints: dict[bytes, Any] = {}  # client_id -> ucp.Endpoint
        self._ucx_loop: asyncio.AbstractEventLoop | None = None
        self._ucx_thread: threading.Thread | None = None

    def bind(self, endpoint: str) -> None:
        self._zmq.bind(endpoint)

    def recv(self) -> tuple[bytes, bytes]:
        return self._zmq.recv()

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
                # Read the full client identity so UCX endpoint lookup matches
                # the exact ZMQ ROUTER identity without truncation collisions.
                size_buf = bytearray(2)
                await ep.recv(size_buf)
                identity_size = int.from_bytes(bytes(size_buf), "big")
                if identity_size <= 0:
                    raise RuntimeError("UCX client identity size must be positive")
                cid_buf = bytearray(identity_size)
                await ep.recv(cid_buf)
                key = ZmqUcxServerTransport._ucx_key(bytes(cid_buf))
                old_ep = self._ucx_endpoints.pop(key, None)
                if old_ep is not None:
                    try:
                        old_ep.close()
                    except Exception:
                        pass
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
        if not ready_event.wait(timeout=self._timeout_s):
            raise TimeoutError("Timed out waiting for UCX listener to start")
        self._ucx_port = actual_port[0]
        return self._ucx_port

    def get_handshake_metadata(self, client_id: bytes) -> dict[str, Any]:
        del client_id
        existing_ucx_port = self.ucx_port
        if existing_ucx_port is None:
            existing_ucx_port = self.start_ucx_listener()
        return {"ucx_port": existing_ucx_port}

    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: torch.Tensor | None = None) -> torch.Tensor:
        """Receive a GPU tensor from a client via UCX."""
        import torch

        key = self._ucx_key(client_id)
        ep = self._ucx_endpoints.get(key)
        if ep is None:
            raise RuntimeError(f"No UCX endpoint for client {client_id.hex()[:8]}")

        if buffer is None:
            buffer = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

        async def _recv_coro():
            _bind_cuda_context(buffer)
            await ep.recv(buffer)

        future = asyncio.run_coroutine_threadsafe(_recv_coro(), self._ucx_loop)
        future.result(timeout=self._timeout_s)
        return buffer

    def send_tensor(self, client_id: bytes, tensor: torch.Tensor) -> None:
        """Send a GPU tensor to a client via UCX."""
        import torch

        key = self._ucx_key(client_id)
        ep = self._ucx_endpoints.get(key)
        if ep is None:
            raise RuntimeError(f"No UCX endpoint for client {client_id.hex()[:8]}")

        # Ensure all pending CUDA ops have completed before UCX reads GPU memory
        # via DMA (__cuda_array_interface__), which bypasses the CUDA stream.
        if tensor.is_cuda:
            torch.cuda.current_stream().synchronize()

        async def _send_coro():
            _bind_cuda_context(tensor)
            await ep.send(tensor)

        future = asyncio.run_coroutine_threadsafe(_send_coro(), self._ucx_loop)
        future.result(timeout=self._timeout_s)

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
        self._timeout_s = max(timeout_ms, 1) / 1000.0
        self._ucx_endpoint = None
        self._ucx_loop: asyncio.AbstractEventLoop | None = None
        self._ucx_thread: threading.Thread | None = None

    def connect(self, endpoint: str) -> None:
        self._zmq.connect(endpoint)

    def cache_identity(self, identity: bytes) -> None:
        self._zmq.cache_identity(identity)

    def reset_identity(self) -> None:
        self._zmq.reset_identity()

    def _shutdown_ucx_resources(self) -> None:
        if self._ucx_endpoint is not None:
            try:
                close_result = self._ucx_endpoint.close()
                if inspect.isawaitable(close_result):
                    if self._ucx_loop is not None and self._ucx_loop.is_running():
                        asyncio.run_coroutine_threadsafe(close_result, self._ucx_loop).result(timeout=self._timeout_s)
                    else:
                        asyncio.run(close_result)
            except Exception:
                pass
            self._ucx_endpoint = None
        if self._ucx_loop is not None:
            try:
                self._ucx_loop.call_soon_threadsafe(self._ucx_loop.stop)
            except Exception:
                pass

    def send(self, payload: bytes) -> None:
        self._zmq.send(payload)

    def recv(self) -> bytes:
        return self._zmq.recv()

    # ---- UCX tensor path ----

    def connect_ucx(self, host: str, port: int, client_id: bytes) -> None:
        """Connect the UCX endpoint to the server's UCX listener."""
        ucp = _ensure_ucp()

        loop = asyncio.new_event_loop()
        self._ucx_loop = loop
        ready_event = threading.Event()

        async def _connect_coro():
            ep = await ucp.create_endpoint(host, port)
            # Send the full ZMQ identity so the server can map this UCX
            # endpoint to the exact same client without truncation.
            await ep.send(len(client_id).to_bytes(2, "big"))
            await ep.send(client_id)
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
        if not ready_event.wait(timeout=self._timeout_s):
            self._shutdown_ucx_resources()
            raise TimeoutError(f"Timed out connecting UCX endpoint to {host}:{port}")
        if self._ucx_endpoint is None:
            self._shutdown_ucx_resources()
            raise RuntimeError(f"Failed to connect UCX endpoint to {host}:{port}")

    def connect_comm_backend(
        self,
        *,
        handshake_response: dict,
        server_host: str,
        zmq_identity: bytes | None,
    ) -> None:
        if zmq_identity is None:
            raise RuntimeError("connect_comm_backend() requires a non-empty zmq_identity for UCX.")
        if "ucx_port" not in handshake_response:
            raise RuntimeError("connect_comm_backend() requires 'ucx_port' in the handshake response for UCX.")
        self.connect_ucx(server_host, int(handshake_response["ucx_port"]), zmq_identity)

    def send_tensor(self, tensor: torch.Tensor) -> None:
        """Send a GPU tensor to the server via UCX."""
        import torch

        if self._ucx_endpoint is None:
            raise RuntimeError("UCX endpoint not connected")

        # Ensure all pending CUDA ops (e.g. torch.cat, tensor packing) have
        # completed before UCX reads GPU memory via DMA, which bypasses
        # the CUDA stream.
        if tensor.is_cuda:
            torch.cuda.current_stream().synchronize()

        async def _send_coro():
            _bind_cuda_context(tensor)
            await self._ucx_endpoint.send(tensor)

        future = asyncio.run_coroutine_threadsafe(_send_coro(), self._ucx_loop)
        future.result(timeout=self._timeout_s)

    def recv_tensor(self, nbytes: int, buffer: torch.Tensor | None = None) -> torch.Tensor:
        """Receive a GPU tensor from the server via UCX."""
        import torch

        if self._ucx_endpoint is None:
            raise RuntimeError("UCX endpoint not connected")
        if buffer is None:
            buffer = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

        async def _recv_coro():
            _bind_cuda_context(buffer)
            await self._ucx_endpoint.recv(buffer)

        future = asyncio.run_coroutine_threadsafe(_recv_coro(), self._ucx_loop)
        future.result(timeout=self._timeout_s)
        return buffer

    def rebuild(self) -> None:
        self._zmq.rebuild()

    def reset_comm_backend(self) -> None:
        self._shutdown_ucx_resources()

    def close(self) -> None:
        self._shutdown_ucx_resources()
        self._zmq.close()

    @property
    def transport_mode(self) -> str:
        return "zmq_ucx"
