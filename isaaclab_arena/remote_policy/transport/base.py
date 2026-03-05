# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod


class ServerTransport(ABC):
    """Abstract base for server-side transports.

    A transport handles the low-level send/recv of byte payloads between
    the PolicyServer and its clients.  Each received message is tagged with
    a ``client_id`` (opaque bytes) so the server can route responses back.
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
        """Send a GPU tensor to a specific client (UCX path).

        Default raises ``NotImplementedError``; overridden by UCX transports.
        """
        raise NotImplementedError("send_tensor requires a UCX-capable transport")

    def recv_tensor(self, client_id: bytes, nbytes: int, buffer: object | None = None) -> object:
        """Receive a GPU tensor from a specific client (UCX path).

        Default raises ``NotImplementedError``; overridden by UCX transports.
        """
        raise NotImplementedError("recv_tensor requires a UCX-capable transport")

    def disconnect_client(self, client_id: bytes) -> None:
        """Clean up resources for a specific client (e.g. UCX endpoint).

        Default is a no-op.  UCX transports should close the endpoint.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @property
    def transport_mode(self) -> str:
        """Return the transport mode identifier (e.g. ``"zmq"``, ``"zmq_ucx"``)."""
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
        """Send a GPU tensor to the server (UCX path)."""
        raise NotImplementedError("send_tensor requires a UCX-capable transport")

    def recv_tensor(self, nbytes: int, buffer: object | None = None) -> object:
        """Receive a GPU tensor from the server (UCX path)."""
        raise NotImplementedError("recv_tensor requires a UCX-capable transport")

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
