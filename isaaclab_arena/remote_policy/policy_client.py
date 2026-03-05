# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig
from isaaclab_arena.remote_policy.transport.base import ClientTransport
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqClientTransport

if TYPE_CHECKING:
    import torch


class TransportTimeoutError(TimeoutError):
    """Unified timeout error with recovery guidance.

    Attributes:
        must_reconnect: If ``True``, the UCX session is likely broken and
            the caller should invoke ``PolicyClient.reconnect()``.
            If ``False``, only ZMQ was affected and ``rebuild()`` was
            already called internally — the session may still be usable.
        source: Which layer or session state triggered recovery
            (for example ``"zmq_recv"``, ``"ucx_connect"``,
            ``"ucx_send"``, ``"ucx_recv"``, ``"client_state"``).
    """

    def __init__(self, message: str, *, must_reconnect: bool, source: str):
        super().__init__(message)
        self.must_reconnect = must_reconnect
        self.source = source


class PolicyClient:
    """v2 synchronous client for talking to a PolicyServer.

    Changes from v1:
      - Uses ``ClientTransport`` (DEALER by default) instead of raw zmq.REQ.
      - ``connect(num_envs)`` performs capability negotiation via ``get_init_info``.
      - Supports ``env_ids`` in ``get_action`` and ``set_task_description``.
      - DEALER timeout recovery via ``transport.rebuild()``.
      - Caches the server-observed ``zmq_identity`` for transport rebuilds.
    """

    @staticmethod
    def _has_ucx_runtime() -> bool:
        try:
            import ucp  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _create_default_transport(timeout_ms: int) -> ClientTransport:
        """Create the production transport from locally available runtimes."""
        if PolicyClient._has_ucx_runtime():
            from isaaclab_arena.remote_policy.transport.zmq_ucx_transport import ZmqUcxClientTransport

            return ZmqUcxClientTransport(timeout_ms=timeout_ms)
        return ZmqClientTransport(timeout_ms=timeout_ms)

    def __init__(
        self,
        config: RemotePolicyConfig,
    ) -> None:
        self._initialize_with_transport(
            config=config,
            transport=self._create_default_transport(timeout_ms=config.timeout_ms),
        )

    @classmethod
    def _from_transport_for_testing(
        cls,
        config: RemotePolicyConfig,
        transport: ClientTransport,
    ) -> PolicyClient:
        """Create a client with an injected transport for tests/benchmarks only.

        Production code should call ``PolicyClient(config)`` so transport
        auto-detection stays behind the public constructor.
        """
        client = cls.__new__(cls)
        client._initialize_with_transport(config=config, transport=transport)
        return client

    def _initialize_with_transport(
        self,
        config: RemotePolicyConfig,
        transport: ClientTransport,
    ) -> None:
        self._config = config
        self._compression: str = config.compression
        self._zmq_compression: str = config.compression  # updated by connect()
        self._tensor_compression: str = "none"  # updated by connect()
        self._negotiated_transport: str = "zmq"
        self._num_envs: int | None = None
        self._last_requested_action_mode: str | None = None

        self._transport = transport
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
        if PolicyClient._has_ucx_runtime():
            caps.append("zmq_ucx")
        return caps

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def _raise_if_error_response(response: Any) -> None:
        if not isinstance(response, dict):
            return
        if response.get("must_reconnect"):
            raise TransportTimeoutError(
                str(response.get("error", "Reconnect required")),
                must_reconnect=True,
                source=str(response.get("source", "unknown")),
            )
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

    def _prepare_observation_for_transport(
        self,
        observation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, "torch.Tensor"] | None]:
        """Prepare an already-packed observation for the negotiated transport."""
        import torch

        use_ucx_tensor_path = (
            self._negotiated_transport == "zmq_ucx"
            and hasattr(self._transport, "send_tensor")
        )

        control_observation: dict[str, Any] = {}
        tensor_observation: dict[str, torch.Tensor] = {}
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if use_ucx_tensor_path and value.is_cuda:
                    tensor_observation[key] = value
                    continue
                control_observation[key] = value.cpu().numpy()
                continue
            control_observation[key] = value

        return control_observation, (tensor_observation or None)

    def _cache_zmq_identity_from_handshake(self, response: dict[str, Any], *, required: bool) -> bytes | None:
        zmq_identity = response.get("zmq_identity")
        if zmq_identity is None:
            if required:
                raise RuntimeError("Protocol error: get_init_info response is missing 'zmq_identity'.")
            return None

        if isinstance(zmq_identity, str):
            zmq_identity = zmq_identity.encode("latin-1")
        if not isinstance(zmq_identity, bytes):
            raise RuntimeError(f"Protocol error: expected bytes 'zmq_identity', got {type(zmq_identity)!r}.")

        transport = cast(Any, self._transport)
        if hasattr(transport, "cache_identity"):
            transport.cache_identity(zmq_identity)
        return zmq_identity

    def _build_request(
        self,
        endpoint: str,
        data: dict[str, Any] | None,
        *,
        requires_input: bool,
    ) -> dict[str, Any]:
        """Build a request envelope with auth metadata."""
        request: dict[str, Any] = {"endpoint": endpoint}
        request_data: dict[str, Any] | None = dict(data or {}) if requires_input else None

        if requires_input:
            request["data"] = request_data or {}
        if self._config.api_token:
            request["api_token"] = self._config.api_token
        return request

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
        zmq_identity = self._cache_zmq_identity_from_handshake(resp, required=True)

        # Store negotiated settings — separate ZMQ and tensor compression
        self._compression = resp.get("negotiated_compression", self._compression)
        self._zmq_compression = resp.get("negotiated_zmq_compression", self._zmq_compression)
        self._tensor_compression = resp.get("negotiated_tensor_compression", self._tensor_compression)
        self._negotiated_transport = resp.get("negotiated_transport", "zmq")

        # If UCX was negotiated, the response and the injected transport must
        # both satisfy the UCX contract. Anything else is a protocol/config bug.
        if self._negotiated_transport == "zmq_ucx":
            if "ucx_port" not in resp:
                raise RuntimeError(
                    "Protocol error: negotiated_transport='zmq_ucx' but handshake response "
                    "is missing 'ucx_port'."
                )
            if "zmq_identity" not in resp:
                raise RuntimeError(
                    "Protocol error: negotiated_transport='zmq_ucx' but handshake response "
                    "is missing 'zmq_identity'."
                )
            if not hasattr(self._transport, "connect_ucx"):
                raise RuntimeError(
                    "Protocol/config mismatch: server negotiated 'zmq_ucx' but client "
                    f"transport {type(self._transport).__name__} does not support connect_ucx()."
                )

            ucx_port = resp["ucx_port"]
            server_host = self._config.host
            # Use the ZMQ identity from the server so UCX endpoint mapping
            # matches ZMQ routing identity (fixes P1-2).
            if zmq_identity is None:
                raise RuntimeError("Protocol error: negotiated 'zmq_ucx' but handshake has no zmq_identity.")
            try:
                cast(Any, self._transport).connect_ucx(server_host, ucx_port, zmq_identity)
            except TimeoutError as exc:
                raise TransportTimeoutError(
                    f"UCX connect timed out for {server_host}:{ucx_port}. Call reconnect().",
                    must_reconnect=True,
                    source="ucx_connect",
                ) from exc

        self._last_requested_action_mode = requested_action_mode
        return resp

    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        """v1-compatible handshake (no capability negotiation).

        Prefer ``connect(num_envs, requested_action_mode)`` for v2.
        """
        payload = {"requested_action_mode": requested_action_mode}
        resp = self.call_endpoint("get_init_info", data=payload, requires_input=True)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(resp)!r}")
        self._cache_zmq_identity_from_handshake(resp, required=False)
        return resp

    def get_action(
        self,
        observation: dict[str, Any],
        env_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Send observations and get back actions.

        Args:
            observation: Already-packed flat observation dict. ``PolicyClient``
                automatically decides how tensor entries travel: CUDA tensors
                may use the negotiated UCX path, while CPU tensors are
                serialized over ZMQ.
            env_ids: Optional environment indices.
        """
        control_observation, tensor_observation = self._prepare_observation_for_transport(observation)

        # UCX tensor path: send CUDA tensors via UCX, metadata via ZMQ
        if tensor_observation is not None:
            return self._get_action_ucx(control_observation, env_ids, tensor_observation)

        # Standard ZMQ path
        payload: dict[str, Any] = {"observation": control_observation}
        if env_ids is not None:
            payload["env_ids"] = env_ids

        resp = self.call_endpoint("get_action", data=payload, requires_input=True)
        return resp

    def _get_action_ucx(
        self,
        observation: dict[str, Any],
        env_ids: list[int] | None,
        tensor_entries: dict[str, "torch.Tensor"],
    ) -> dict[str, Any]:
        """Internal UCX path: send selected tensor payloads via zero-copy."""
        import torch

        # Flatten all GPU tensors into a single contiguous buffer
        tensor_layout: list[dict[str, Any]] = []
        flat_parts = []
        offset = 0
        for key, t in tensor_entries.items():
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

        tensor_compressed = self._tensor_compression == "nvcomp_lz4"
        if tensor_compressed:
            from isaaclab_arena.remote_policy.gpu_compression import gpu_compress
            flat_buffer = gpu_compress(flat_buffer)

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

        request = self._build_request("get_action", control_payload, requires_input=True)
        # Use negotiated ZMQ compression for the control message
        self._transport.send(MessageSerializer.to_bytes(request, compression_method=self._zmq_compression))

        # Send tensor via UCX
        try:
            self._transport.send_tensor(flat_buffer)
        except (TimeoutError, RuntimeError) as exc:
            raise TransportTimeoutError(
                f"UCX send_tensor failed: {exc}",
                must_reconnect=True,
                source="ucx_send",
            ) from exc

        # Receive response via ZMQ
        try:
            raw = self._transport.recv()
        except TimeoutError:
            self._transport.rebuild()
            raise TransportTimeoutError(
                "ZMQ recv timed out during UCX get_action. "
                "UCX session is likely stale — call reconnect().",
                must_reconnect=True,
                source="zmq_recv",
            )
        response = MessageSerializer.from_bytes(raw)
        self._raise_if_error_response(response)
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

    def reconnect(self, num_envs: int | None = None, requested_action_mode: str | None = None) -> dict[str, Any]:
        """Full reconnect: rebuild ZMQ + re-handshake + re-establish UCX.

        Use this after a UCX timeout or when the server has GC'd this
        client's state.  A simple ``rebuild()`` only recovers ZMQ; this
        method performs a complete re-initialization.

        Args:
            num_envs: Override num_envs for the new session (default: reuse).
            requested_action_mode: Action mode for the handshake. Defaults to
                the last successful handshake's mode, or ``"chunk"`` if none.
        """
        n = num_envs if num_envs is not None else (self._num_envs or 1)
        action_mode = requested_action_mode or self._last_requested_action_mode or "chunk"

        # Close UCX endpoint if present
        transport = cast(Any, self._transport)
        if hasattr(transport, "_ucx_endpoint") and transport._ucx_endpoint is not None:
            try:
                transport._ucx_endpoint.close()
            except Exception:
                pass
            transport._ucx_endpoint = None

        if hasattr(transport, "reset_identity"):
            transport.reset_identity()

        # Rebuild ZMQ and let ZMQ assign a fresh identity.
        self._transport.rebuild()

        # Re-handshake
        return self.connect(n, action_mode)

    def disconnect(self) -> Any:
        """Disconnect this client from the server (cleans up server-side state)."""
        return self.call_endpoint("disconnect", requires_input=False)

    def kill(self) -> Any:
        """Ask remote server to stop main loop.

        Only works if the server was started with ``allow_remote_kill=True``.
        Prefer :meth:`disconnect` for normal client teardown.
        """
        return self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
    ) -> Any:
        """Generic RPC helper."""
        request = self._build_request(endpoint, data, requires_input=requires_input)

        compression = self._zmq_compression if endpoint != "get_init_info" else "none"
        self._transport.send(MessageSerializer.to_bytes(request, compression_method=compression))

        try:
            raw = self._transport.recv()
        except TimeoutError:
            # DEALER timeout recovery: rebuild socket to clear stale buffer.
            # ZMQ-only timeouts do NOT require reconnect() — rebuild is
            # sufficient (stable identity preserved, no UCX state affected).
            self._transport.rebuild()
            raise TransportTimeoutError(
                f"ZMQ recv timed out on endpoint={endpoint!r}. "
                "ZMQ socket rebuilt. Session is still usable for ZMQ-only requests.",
                must_reconnect=False,
                source="zmq_recv",
            )

        response = MessageSerializer.from_bytes(raw)
        self._raise_if_error_response(response)
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
