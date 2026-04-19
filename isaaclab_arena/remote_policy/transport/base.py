# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ServerTransport(ABC):
    """Abstract base for server-side transports.

    A transport handles:
    - control-plane byte send/recv for ``PolicyServer``,
    - optional dedicated tensor send/recv backends,
    - optional handshake metadata for post-ZMQ backends.

    It must not know anything about tensor codecs or compression internals.
    """

    @abstractmethod
    def bind(self, endpoint: str) -> None:
        """Bind to the given endpoint (e.g. ``"tcp://*:5555"``)."""

    @abstractmethod
    def recv(self) -> tuple[bytes, bytes]:
        """Block until a message arrives (uses the default timeout).

        Returns:
            ``(client_id, payload)`` where *client_id* is an opaque identity
            and *payload* is the raw serialized request bytes.

        Raises:
            TimeoutError: if the underlying socket times out.
        """

    @abstractmethod
    def send(self, client_id: bytes, payload: bytes) -> None:
        """Send a response to a specific client."""

    def send_tensor(self, client_id: bytes, tensor: object) -> None:
        """Send a tensor via a dedicated data backend.

        Default raises ``NotImplementedError``; overridden by transports that
        support a dedicated tensor path.
        """
        raise NotImplementedError("send_tensor requires a dedicated tensor backend")

    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: object | None = None) -> object:
        """Receive a tensor via a dedicated data backend.

        Default raises ``NotImplementedError``; overridden by transports that
        support a dedicated tensor path.
        """
        raise NotImplementedError("recv_tensor requires a dedicated tensor backend")

    def get_handshake_metadata(self, client_id: bytes) -> dict[str, Any]:
        """Return backend metadata that should be attached to ``get_init_info``.

        ZMQ-only transports can keep the default empty dict. Dedicated tensor
        transports (for example Mooncake or legacy UCX) can override this and
        return backend-specific static metadata for the client.
        """
        del client_id
        return {}

    def disconnect_client(self, client_id: bytes) -> None:
        """Clean up resources for a specific client (e.g. UCX endpoint).

        Default is a no-op.  UCX transports should close the endpoint.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @property
    def transport_mode(self) -> str:
        """Return the transport mode identifier."""
        return "zmq"


class ClientTransport(ABC):
    """Abstract base for client-side transports."""

    @abstractmethod
    def connect(self, endpoint: str) -> None:
        """Connect to the given endpoint (e.g. ``"tcp://host:5555"``)."""

    @abstractmethod
    def send(self, payload: bytes) -> None:
        """Send a request payload to the server."""

    @abstractmethod
    def recv(self) -> bytes:
        """Block until a response arrives.

        Returns:
            The raw serialized response bytes.

        Raises:
            TimeoutError: if the underlying socket times out.
        """

    def send_tensor(self, tensor: object) -> None:
        """Send a tensor to the server via a dedicated data backend."""
        raise NotImplementedError("send_tensor requires a dedicated tensor backend")

    def recv_tensor(self, nbytes: int, buffer: object | None = None) -> object:
        """Receive a tensor from the server via a dedicated data backend."""
        raise NotImplementedError("recv_tensor requires a dedicated tensor backend")

    def connect_comm_backend(
        self,
        *,
        handshake_response: dict,
        server_host: str,
        zmq_identity: bytes | None,
    ) -> None:
        """Connect or initialize a post-handshake backend.

        ZMQ-only transports can keep the default behavior. Dedicated tensor
        transports (Mooncake, legacy UCX, future backends) should override this.
        """
        raise NotImplementedError("connect_comm_backend requires a dedicated communication backend")

    def reset_comm_backend(self) -> None:
        """Reset any non-ZMQ backend state before starting a new session."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    def rebuild(self) -> None:
        """Rebuild the transport after a timeout (DEALER recovery).

        Default is a no-op.  ZMQ DEALER transports should close and
        reconnect to clear stale buffers.
        """

    @property
    def transport_mode(self) -> str:
        return "zmq"
