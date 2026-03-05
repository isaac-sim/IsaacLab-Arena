# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import zmq

from .base import ClientTransport, ServerTransport


class ZmqServerTransport(ServerTransport):
    """ZMQ ROUTER-based server transport.

    The ROUTER socket automatically prepends an identity frame to every
    received message, allowing the server to route replies back to the
    correct client.  The wire format is::

        recv:  [identity, b"", payload]
        send:  [client_id, b"", payload]
    """

    def __init__(self, timeout_ms: int = 15000, context: zmq.Context | None = None) -> None:
        self._context = context or zmq.Context()
        self._owns_context = context is None
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._endpoint: str | None = None

    def bind(self, endpoint: str) -> None:
        self._socket.bind(endpoint)
        self._endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def recv(self) -> tuple[bytes, bytes]:
        try:
            parts = self._socket.recv_multipart()
        except zmq.Again:
            raise TimeoutError("ZmqServerTransport.recv() timed out")
        # ROUTER framing: [identity, empty_delimiter, payload]
        if len(parts) < 3 or parts[1] != b"":
            raise ValueError(f"Unexpected ROUTER frame: expected [id, b'', payload], got {len(parts)} parts")
        client_id = cast(bytes, parts[0])
        payload = cast(bytes, parts[2])
        return client_id, payload

    def send(self, client_id: bytes, payload: bytes) -> None:
        self._socket.send_multipart([client_id, b"", payload])

    def close(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
        if self._owns_context:
            try:
                self._context.term()
            except Exception:
                pass

    @property
    def endpoint(self) -> str | None:
        return self._endpoint


class ZmqClientTransport(ClientTransport):
    """ZMQ DEALER-based client transport.

    The DEALER socket must explicitly include the empty delimiter frame
    that the ROUTER expects::

        send:  [b"", payload]
        recv:  [b"", payload]  →  return payload (parts[1])
    """

    def __init__(self, timeout_ms: int = 15000, context: zmq.Context | None = None) -> None:
        self._context = context or zmq.Context()
        self._owns_context = context is None
        self._timeout_ms = timeout_ms
        self._socket: zmq.Socket | None = None
        self._endpoint: str | None = None
        # None means "let ZMQ assign a fresh routing identity on the next
        # connect". After the first handshake, PolicyClient caches the
        # server-observed identity here so rebuild() can preserve the same
        # session-level routing key.
        self._identity: bytes | None = None

    def connect(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._socket = self._context.socket(zmq.DEALER)
        if self._identity is not None:
            self._socket.setsockopt(zmq.IDENTITY, self._identity)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.connect(endpoint)

    def cache_identity(self, identity: bytes) -> None:
        """Cache a server-confirmed ZMQ routing identity for future rebuilds."""
        self._identity = identity

    def reset_identity(self) -> None:
        """Forget the cached identity so the next connect gets a fresh one."""
        self._identity = None

    def send(self, payload: bytes) -> None:
        if self._socket is None:
            raise RuntimeError("ZmqClientTransport: not connected")
        self._socket.send_multipart([b"", payload])

    def recv(self) -> bytes:
        if self._socket is None:
            raise RuntimeError("ZmqClientTransport: not connected")
        try:
            parts = self._socket.recv_multipart()
        except zmq.Again:
            raise TimeoutError("ZmqClientTransport.recv() timed out")
        # DEALER framing: [empty_delimiter, payload]
        if len(parts) < 2 or parts[0] != b"":
            raise ValueError(f"Unexpected DEALER frame: expected [b'', payload], got {len(parts)} parts")
        return cast(bytes, parts[1])

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._owns_context:
            try:
                self._context.term()
            except Exception:
                pass

    def rebuild(self) -> None:
        """Close and reconnect to clear stale DEALER buffers after timeout."""
        if self._endpoint is None:
            return
        endpoint = self._endpoint
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        self.connect(endpoint)
