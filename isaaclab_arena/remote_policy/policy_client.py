# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig
from isaaclab_arena.remote_policy.transport.base import ClientTransport
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqClientTransport


class PolicyClient:
    """v2 synchronous client for talking to a PolicyServer.

    Changes from v1:
      - Uses ``ClientTransport`` (DEALER by default) instead of raw zmq.REQ.
      - ``connect(num_envs)`` performs capability negotiation via ``get_init_info``.
      - Supports ``env_ids`` in ``get_action`` and ``set_task_description``.
      - DEALER timeout recovery via ``transport.rebuild()``.
    """

    def __init__(
        self,
        config: RemotePolicyConfig,
        transport: ClientTransport | None = None,
    ) -> None:
        self._config = config
        self._compression: str = config.compression
        self._zmq_compression: str = config.compression  # updated by connect()
        self._tensor_compression: str = "none"  # updated by connect()
        self._negotiated_transport: str = "zmq"
        self._num_envs: int | None = None
        self._ucx_client_id: bytes | None = None

        # Transport
        if transport is not None:
            self._transport = transport
        else:
            self._transport = ZmqClientTransport(timeout_ms=config.timeout_ms)

        endpoint = f"tcp://{config.host}:{config.port}"
        self._transport.connect(endpoint)

    # ------------------------------------------------------------------ #
    # Capability detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_compression_capabilities() -> list[str]:
        caps = ["none"]
        try:
            import lz4.frame  # noqa: F401

            caps.append("lz4")
        except ImportError:
            pass
        try:
            from nvidia.nvcomp import Codec  # noqa: F401

            caps.append("nvcomp_lz4")
        except ImportError:
            pass
        return caps

    @staticmethod
    def _detect_transport_capabilities() -> list[str]:
        caps = ["zmq"]
        try:
            import ucp  # noqa: F401

            caps.append("zmq_ucx")
        except ImportError:
            pass
        return caps

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception as exc:
            warnings.warn(
                f"[PolicyClient] Failed to ping remote policy server at "
                f"{self._config.host}:{self._config.port}: {exc}"
            )
            return False

    def connect(self, num_envs: int, requested_action_mode: str) -> dict[str, Any]:
        """Perform the v2 handshake: capability negotiation + get_init_info.

        Args:
            num_envs: Number of environments this client manages.
            requested_action_mode: ActionMode value string (e.g. ``"chunk"``).

        Returns:
            The server's ``get_init_info`` response dict.
        """
        self._num_envs = num_envs
        payload = {
            "requested_action_mode": requested_action_mode,
            "num_envs": num_envs,
            "transport_capabilities": self._detect_transport_capabilities(),
            "compression_capabilities": self._detect_compression_capabilities(),
        }
        resp = self.call_endpoint("get_init_info", data=payload, requires_input=True)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(resp)!r}")

        # Store negotiated settings — separate ZMQ and tensor compression
        self._compression = resp.get("negotiated_compression", self._compression)
        self._zmq_compression = resp.get("negotiated_zmq_compression", self._zmq_compression)
        self._tensor_compression = resp.get("negotiated_tensor_compression", self._tensor_compression)
        self._negotiated_transport = resp.get("negotiated_transport", "zmq")

        # If UCX was negotiated, connect the UCX endpoint
        if self._negotiated_transport == "zmq_ucx" and "ucx_port" in resp:
            ucx_port = resp["ucx_port"]
            server_host = self._config.host
            if hasattr(self._transport, "connect_ucx"):
                # Use the ZMQ identity from the server so UCX endpoint
                # mapping matches ZMQ routing identity (fixes P1-2)
                zmq_identity = resp.get("zmq_identity", b"")
                if isinstance(zmq_identity, str):
                    zmq_identity = zmq_identity.encode("latin-1")
                self._ucx_client_id = zmq_identity
                self._transport.connect_ucx(server_host, ucx_port, zmq_identity)

        return resp

    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        """v1-compatible handshake (no capability negotiation).

        Prefer ``connect(num_envs, requested_action_mode)`` for v2.
        """
        payload = {"requested_action_mode": requested_action_mode}
        resp = self.call_endpoint("get_init_info", data=payload, requires_input=True)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(resp)!r}")
        return resp

    def get_action(
        self,
        observation: dict[str, Any],
        env_ids: list[int] | None = None,
        gpu_tensors: dict[str, "torch.Tensor"] | None = None,
    ) -> dict[str, Any]:
        """Send observations and get back actions.

        Args:
            observation: CPU-side observation dict (numpy arrays).
            env_ids: Optional environment indices.
            gpu_tensors: Optional dict of GPU tensors to send via UCX.
                When provided and UCX is negotiated, these are sent as a
                single flat GPU buffer (optionally nvcomp-compressed).
                The ZMQ control message carries shape/dtype metadata.
        """
        # UCX tensor path: send GPU tensors via UCX, metadata via ZMQ
        if (
            gpu_tensors is not None
            and self._negotiated_transport == "zmq_ucx"
            and hasattr(self._transport, "send_tensor")
        ):
            return self._get_action_ucx(observation, env_ids, gpu_tensors)

        # Standard ZMQ path
        payload: dict[str, Any] = {"observation": observation}
        if env_ids is not None:
            payload["env_ids"] = env_ids

        resp = self.call_endpoint("get_action", data=payload, requires_input=True)
        return resp

    def _get_action_ucx(
        self,
        observation: dict[str, Any],
        env_ids: list[int] | None,
        gpu_tensors: dict[str, "torch.Tensor"],
    ) -> dict[str, Any]:
        """UCX path: send GPU tensors via zero-copy, control via ZMQ."""
        import torch

        # Flatten all GPU tensors into a single contiguous buffer
        tensor_layout: list[dict[str, Any]] = []
        flat_parts = []
        offset = 0
        for key, t in gpu_tensors.items():
            flat = t.contiguous().view(torch.uint8).reshape(-1)
            tensor_layout.append({
                "key": key,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "offset": offset,
                "nbytes": flat.numel(),
            })
            flat_parts.append(flat)
            offset += flat.numel()

        flat_buffer = torch.cat(flat_parts) if len(flat_parts) > 1 else flat_parts[0]
        original_nbytes = flat_buffer.numel()

        # Optionally compress with nvcomp
        tensor_compressed = False
        if self._tensor_compression == "nvcomp_lz4":
            from isaaclab_arena.remote_policy.gpu_compression import gpu_compress
            flat_buffer = gpu_compress(flat_buffer)
            tensor_compressed = True

        # Send control message via ZMQ
        control_payload: dict[str, Any] = {
            "observation": observation,
            "has_tensor": True,
            "tensor_layout": tensor_layout,
            "tensor_nbytes": flat_buffer.numel(),
            "tensor_original_nbytes": original_nbytes,
            "tensor_compressed": tensor_compressed,
        }
        if env_ids is not None:
            control_payload["env_ids"] = env_ids

        request: dict[str, Any] = {"endpoint": "get_action", "data": control_payload}
        if self._config.api_token:
            request["api_token"] = self._config.api_token
        # Use negotiated ZMQ compression for the control message
        self._transport.send(MessageSerializer.to_bytes(request, compression_method=self._zmq_compression))

        # Send tensor via UCX
        self._transport.send_tensor(flat_buffer)

        # Receive response via ZMQ (with timeout recovery)
        try:
            raw = self._transport.recv()
        except TimeoutError:
            warnings.warn("[PolicyClient] Timeout in UCX get_action ZMQ recv, rebuilding transport")
            self._transport.rebuild()
            raise
        response = MessageSerializer.from_bytes(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def set_task_description(
        self,
        task_description: str | None,
        env_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Send task description to the remote policy."""
        payload: dict[str, Any] = {"task_description": task_description}
        if env_ids is not None:
            payload["env_ids"] = env_ids

        resp = self.call_endpoint("set_task_description", data=payload, requires_input=True)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from set_task_description, got {type(resp)!r}")
        return resp

    def reset(self, env_ids: list[int] | None = None, options: dict[str, Any] | None = None) -> Any:
        """Reset remote policy state."""
        resp = self.call_endpoint(
            endpoint="reset",
            data={"env_ids": env_ids, "options": options},
            requires_input=True,
        )
        if isinstance(resp, dict):
            status = resp.get("status")
            if status not in ("reset_success", "ok", "reset_ok", None):
                raise RuntimeError(f"Remote reset failed with status={status}, resp={resp}")
        return resp

    def kill(self) -> Any:
        """Ask remote server to stop main loop."""
        return self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
    ) -> Any:
        """Generic RPC helper."""
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self._config.api_token:
            request["api_token"] = self._config.api_token

        compression = self._zmq_compression if endpoint != "get_init_info" else "none"
        self._transport.send(MessageSerializer.to_bytes(request, compression_method=compression))

        try:
            raw = self._transport.recv()
        except TimeoutError:
            # DEALER timeout recovery: rebuild socket to clear stale buffer
            warnings.warn(f"[PolicyClient] Timeout on endpoint={endpoint!r}, rebuilding transport")
            self._transport.rebuild()
            raise

        response = MessageSerializer.from_bytes(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def close(self) -> None:
        """Close the underlying transport."""
        self._transport.close()

    @property
    def compression(self) -> str:
        return self._compression

    @property
    def negotiated_transport(self) -> str:
        return self._negotiated_transport
