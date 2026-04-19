# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import warnings
from contextlib import suppress
from typing import Any, cast

import torch

from isaaclab_arena.remote_policy.compression import (
    TensorPayloadCodec,
    build_control_observation_for_tensor_transport,
    split_observation_entries,
)
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.metrics import record_remote_metric
from isaaclab_arena.remote_policy.mooncake_config import autodetect_local_hostname
from isaaclab_arena.remote_policy.protocol_enums import TransportMode
from isaaclab_arena.remote_policy.profiling import nvtx_range
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig
from isaaclab_arena.remote_policy.transport.base import ClientTransport
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqClientTransport


class TransportTimeoutError(TimeoutError):
    """Unified timeout error with recovery guidance."""

    def __init__(self, message: str, *, must_reconnect: bool, source: str):
        super().__init__(message)
        self.must_reconnect = must_reconnect
        self.source = source


class PolicyClient:
    """Synchronous client for talking to a ``PolicyServer``."""

    @staticmethod
    def _normalize_tensor_device(device: str | None) -> str | None:
        if device is None:
            return None
        if device == "cpu":
            return "cpu"

        try:
            parsed = torch.device(device)
        except Exception:
            return device

        if parsed.type != "cuda":
            return str(parsed)
        if parsed.index is None:
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
            return "cuda"
        return str(parsed)

    @staticmethod
    def _has_mooncake_runtime() -> bool:
        try:
            import mooncake.engine  # noqa: F401
        except (ImportError, OSError):
            return False
        return True


    @staticmethod
    def _create_default_transport(config: RemotePolicyConfig) -> ClientTransport:
        """Create the production transport from explicit config."""
        timeout_ms = config.timeout_ms
        mode = TransportMode.parse(config.transport_mode)
        mooncake = config.mooncake

        if mode == TransportMode.ZMQ:
            return ZmqClientTransport(timeout_ms=timeout_ms)

        if mode == TransportMode.ZMQ_UCX:
            raise RuntimeError(
                "transport_mode='zmq_ucx' is a legacy/debug path and is no longer available through the mainline "
                "PolicyClient constructor."
            )

        if mode != TransportMode.ZMQ_MOONCAKE:
            raise ValueError(f"Unsupported transport_mode={mode!r}")

        if not PolicyClient._has_mooncake_runtime():
            raise RuntimeError(
                "transport_mode='zmq_mooncake' was requested but the Mooncake runtime is unavailable."
            )

        resolved_local_hostname = mooncake.local_hostname or autodetect_local_hostname(config.host)
        if not resolved_local_hostname:
            raise RuntimeError(
                "transport_mode='zmq_mooncake' requires a local hostname/IP that peers can reach. "
                "Pass --remote_mooncake_local_hostname to override."
            )

        from isaaclab_arena.remote_policy.transport.zmq_mooncake_transport import ZmqMooncakeClientTransport

        return ZmqMooncakeClientTransport(
            timeout_ms=timeout_ms,
            local_hostname=resolved_local_hostname,
            metadata_server=mooncake.metadata_backend,
            protocol=mooncake.protocol,
            device_name=mooncake.device_name or "",
            buffer_bytes=mooncake.staging_buffer_bytes,
            tensor_device=None,
            cuda_device_override=mooncake.cuda_device_override,
            force_register=mooncake.force_register,
        )

    def __init__(self, config: RemotePolicyConfig, tensor_device: str | None = None) -> None:
        self._initialize_with_transport(
            config=config,
            transport=self._create_default_transport(config),
            tensor_device=tensor_device,
        )

    @classmethod
    def _from_transport_for_testing(
        cls,
        config: RemotePolicyConfig,
        transport: ClientTransport,
        tensor_device: str | None = None,
    ) -> PolicyClient:
        """Create a client with an injected transport for tests only."""
        client = cls.__new__(cls)
        client._initialize_with_transport(config=config, transport=transport, tensor_device=tensor_device)
        return client

    def _initialize_with_transport(
        self,
        config: RemotePolicyConfig,
        transport: ClientTransport,
        tensor_device: str | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._transport_endpoint = f"tcp://{config.host}:{config.port}"
        self._transport_connected = False
        self._transport_mode = TransportMode.parse(transport.transport_mode)
        self._tensor_device = self._normalize_tensor_device(tensor_device)
        self._session_initialized = False
        self._num_envs: int | None = None
        self._last_requested_action_mode: str | None = None

        if self._transport_mode == TransportMode.ZMQ_MOONCAKE and self._tensor_device == "cpu":
            raise RuntimeError(
                "transport_mode='zmq_mooncake' requires policy_device to resolve to a CUDA device."
            )

        transport_any = cast(Any, self._transport)
        if hasattr(transport_any, "_tensor_device") and self._tensor_device is not None:
            transport_any._tensor_device = self._tensor_device

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

    def _ensure_transport_connected(self) -> None:
        if self._transport_connected:
            return
        self._transport.connect(self._transport_endpoint)
        self._transport_connected = True

    def _build_request(
        self,
        endpoint: str,
        data: dict[str, Any] | None,
        *,
        requires_input: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = dict(data or {})
        if self._config.api_token:
            request["api_token"] = self._config.api_token
        return request

    def _send_request(
        self,
        endpoint: str,
        data: dict[str, Any] | None,
        *,
        requires_input: bool,
    ) -> None:
        self._ensure_transport_connected()
        request = self._build_request(endpoint, data, requires_input=requires_input)
        serialize_started_ns = time.perf_counter_ns()
        payload = MessageSerializer.to_bytes(request)
        serialize_elapsed_ms = (time.perf_counter_ns() - serialize_started_ns) / 1e6

        send_started_ns = time.perf_counter_ns()
        self._transport.send(payload)
        send_elapsed_ms = (time.perf_counter_ns() - send_started_ns) / 1e6
        record_remote_metric(
            "client_request_send",
            endpoint=endpoint,
            transport_mode=self._transport_mode.value,
            requires_input=requires_input,
            serialized_nbytes=len(payload),
            serialize_ms=serialize_elapsed_ms,
            send_ms=send_elapsed_ms,
        )

    def _recv_get_action_response(
        self,
        *,
        timeout_message: str,
        must_start_new_session: bool,
    ) -> dict[str, Any]:
        recv_started_ns = time.perf_counter_ns()
        try:
            raw = self._transport.recv()
        except TimeoutError:
            self._transport.rebuild()
            raise TransportTimeoutError(
                timeout_message,
                must_reconnect=must_start_new_session,
                source="zmq_recv",
            )
        recv_elapsed_ms = (time.perf_counter_ns() - recv_started_ns) / 1e6
        decode_started_ns = time.perf_counter_ns()
        response = MessageSerializer.from_bytes(raw)
        decode_elapsed_ms = (time.perf_counter_ns() - decode_started_ns) / 1e6
        record_remote_metric(
            "client_get_action_response",
            transport_mode=self._transport_mode.value,
            must_start_new_session=must_start_new_session,
            response_nbytes=len(raw),
            recv_ms=recv_elapsed_ms,
            decode_ms=decode_elapsed_ms,
            response_keys=sorted(response.keys()) if isinstance(response, dict) else None,
        )
        self._raise_if_error_response(response)
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict response, got {type(response)!r}")
        return response

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception as exc:
            warnings.warn(
                f"[PolicyClient] Failed to ping remote policy server at {self._config.host}:{self._config.port}: {exc}"
            )
            return False

    def initialize_session(self, num_envs: int, requested_action_mode: str) -> dict[str, Any]:
        """Perform the explicit ``get_init_info`` handshake."""
        self._send_request(
            "get_init_info",
            data={
                "requested_action_mode": requested_action_mode,
                "num_envs": num_envs,
                "transport_mode": self._transport_mode.value,
            },
            requires_input=True,
        )

        try:
            raw = self._transport.recv()
        except TimeoutError:
            self._transport.rebuild()
            raise TransportTimeoutError(
                "ZMQ recv timed out on endpoint='get_init_info'. "
                "ZMQ socket rebuilt. Retry initialize_session().",
                must_reconnect=False,
                source="zmq_recv",
            )

        response = MessageSerializer.from_bytes(raw)
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(response)!r}")
        if "error" in response or response.get("status") == "rejected":
            self.close()
            return response

        zmq_identity = response.get("zmq_identity")
        if isinstance(zmq_identity, str):
            zmq_identity = zmq_identity.encode("latin-1")
        if not isinstance(zmq_identity, bytes):
            raise RuntimeError("Protocol error: get_init_info response is missing bytes 'zmq_identity'.")

        transport = cast(Any, self._transport)
        if hasattr(transport, "cache_identity"):
            # The live DEALER socket already owns this routing identity for all
            # future sends on the current connection. Cache the server-observed
            # value only so rebuild() can rebind a replacement socket to the
            # same session key after a timeout.
            transport.cache_identity(zmq_identity)

        if self._transport_mode == TransportMode.ZMQ_MOONCAKE:
            for field in ("mooncake_protocol", "mooncake_server_session_id"):
                if field not in response:
                    raise RuntimeError(
                        "Protocol error: transport_mode='zmq_mooncake' but get_init_info response "
                        f"is missing {field!r}."
                    )
            try:
                self._transport.connect_comm_backend(
                    handshake_response=response,
                    server_host=self._config.host,
                    zmq_identity=zmq_identity,
                )
            except (NotImplementedError, AttributeError):
                raise RuntimeError(
                    f"Protocol/config mismatch: transport_mode='zmq_mooncake' but client transport "
                    f"{type(self._transport).__name__} does not support connect_comm_backend()."
                )
            except TimeoutError as exc:
                with suppress(Exception):
                    self.disconnect()
                raise TransportTimeoutError(
                    "Mooncake backend connect timed out. Best-effort disconnect() was sent to clear server-side state.",
                    must_reconnect=True,
                    source="mooncake_connect",
                ) from exc
            except Exception as exc:
                with suppress(Exception):
                    self.disconnect()
                raise RuntimeError(
                    "Failed to connect Mooncake backend after get_init_info. "
                    "Best-effort disconnect() was sent to clear server-side state."
                ) from exc

        self._num_envs = int(response.get("num_envs", num_envs))
        self._last_requested_action_mode = requested_action_mode
        self._session_initialized = True
        return response

    def get_action(
        self,
        observation: dict[str, Any],
        env_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Send observations and get back actions."""
        control_entries, tensor_entries = split_observation_entries(observation)

        if self._transport_mode == TransportMode.ZMQ_MOONCAKE:
            control_observation, transport_tensor_entries = build_control_observation_for_tensor_transport(
                control_entries,
                tensor_entries,
            )
            if transport_tensor_entries is not None:
                return self._get_action_dedicated_tensor_transport(
                    control_observation,
                    env_ids,
                    transport_tensor_entries,
                )

            payload: dict[str, Any] = {"observation": control_observation}
            if env_ids is not None:
                payload["env_ids"] = env_ids
            response = self.call_endpoint("get_action", data=payload, requires_input=True)
            if not isinstance(response, dict):
                raise TypeError(f"Expected dict response, got {type(response)!r}")
            return response

        if tensor_entries is not None:
            return self._get_action_inline_tensor_payload(control_entries, env_ids, tensor_entries)

        payload = {"observation": control_entries}
        if env_ids is not None:
            payload["env_ids"] = env_ids
        response = self.call_endpoint("get_action", data=payload, requires_input=True)
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict response, got {type(response)!r}")
        return response

    @staticmethod
    def _build_dedicated_tensor_control_payload(
        observation: dict[str, Any],
        env_ids: list[int] | None,
        *,
        tensor_layout: list[dict[str, Any]],
        tensor_nbytes: int,
        tensor_transport_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "observation": observation,
            "has_tensor": True,
            "tensor_layout": tensor_layout,
            "tensor_nbytes": tensor_nbytes,
        }
        if tensor_transport_info is not None:
            payload["tensor_transport_info"] = tensor_transport_info
        if env_ids is not None:
            payload["env_ids"] = env_ids
        return payload

    def _require_local_tensor_device(self, *, context: str) -> str:
        if self._tensor_device is None:
            raise RuntimeError(f"{context} requires policy_device to resolve to a local tensor device.")
        return self._tensor_device

    def _inline_tensor_target_device(self) -> str:
        return "cpu"

    def _prepare_request_tensor_payload(
        self,
        tensor_entries: dict[str, "torch.Tensor"],
        *,
        target_device: str,
    ):
        # TensorPayloadCodec stays transport-agnostic. PolicyClient chooses the
        # target device from the final wire shape: inline ZMQ always uses CPU
        # bytes, while dedicated tensor transports keep a device-resident buffer
        # for send_tensor().
        codec = TensorPayloadCodec()
        prepared_payload = codec.prepare_tensor_payload(
            tensor_entries,
            target_device=target_device,
        )
        record_remote_metric(
            "client_tensor_payload_prepared",
            transport_mode=self._transport_mode.value,
            target_device=target_device,
            tensor_keys=sorted(tensor_entries.keys()),
            original_nbytes=prepared_payload.original_nbytes,
        )
        return prepared_payload

    def _get_action_inline_tensor_payload(
        self,
        control_entries: dict[str, Any],
        env_ids: list[int] | None,
        tensor_entries: dict[str, "torch.Tensor"],
    ) -> dict[str, Any]:
        """Pure-ZMQ tensor path using one packed inline tensor blob."""
        total_started_ns = time.perf_counter_ns()
        prepared_payload = self._prepare_request_tensor_payload(
            tensor_entries,
            target_device=self._inline_tensor_target_device(),
        )
        materialize_started_ns = time.perf_counter_ns()
        inline_tensor_payload = prepared_payload.flat_buffer.detach().cpu().numpy().tobytes()
        materialize_elapsed_ms = (time.perf_counter_ns() - materialize_started_ns) / 1e6
        record_remote_metric(
            "client_inline_payload_materialized",
            transport_mode=self._transport_mode.value,
            payload_nbytes=len(inline_tensor_payload),
            original_nbytes=prepared_payload.original_nbytes,
            materialize_ms=materialize_elapsed_ms,
        )

        payload: dict[str, Any] = {
            "observation": dict(control_entries),
            "inline_tensor_layout": prepared_payload.tensor_layout,
            "inline_tensor_payload": inline_tensor_payload,
            "inline_tensor_nbytes": len(inline_tensor_payload),
        }
        if env_ids is not None:
            payload["env_ids"] = env_ids

        self._send_request("get_action", payload, requires_input=True)
        response = self._recv_get_action_response(
            timeout_message=(
                "ZMQ recv timed out during packed-ZMQ get_action. "
                "ZMQ socket rebuilt. Retry or reconnect if the server session was lost."
            ),
            must_start_new_session=False,
        )
        record_remote_metric(
            "client_get_action_total",
            path="inline_zmq",
            transport_mode=self._transport_mode.value,
            total_ms=(time.perf_counter_ns() - total_started_ns) / 1e6,
            original_nbytes=prepared_payload.original_nbytes,
        )
        return response

    def _get_action_dedicated_tensor_transport(
        self,
        observation: dict[str, Any],
        env_ids: list[int] | None,
        tensor_entries: dict[str, "torch.Tensor"],
    ) -> dict[str, Any]:
        """Dedicated tensor transport path (currently ZMQ+Mooncake)."""
        total_started_ns = time.perf_counter_ns()
        transport = cast(Any, self._transport)
        if not hasattr(transport, "tensor_source_info"):
            raise RuntimeError(
                "Protocol/config mismatch: transport_mode='zmq_mooncake' but client transport "
                f"{type(self._transport).__name__} does not expose tensor_source_info()."
            )

        prepared_payload = self._prepare_request_tensor_payload(
            tensor_entries,
            target_device=self._require_local_tensor_device(context="transport_mode='zmq_mooncake'"),
        )
        transport_payload = prepared_payload.flat_buffer

        send_tensor_started_ns = time.perf_counter_ns()
        try:
            with nvtx_range("mooncake.stage_send_tensor"):
                self._transport.send_tensor(transport_payload)
        except (TimeoutError, RuntimeError) as exc:
            raise TransportTimeoutError(
                f"Mooncake send_tensor failed: {exc}",
                must_reconnect=True,
                source="mooncake_send",
            ) from exc
        record_remote_metric(
            "client_dedicated_tensor_send",
            transport_mode=self._transport_mode.value,
            original_nbytes=prepared_payload.original_nbytes,
            send_tensor_ms=(time.perf_counter_ns() - send_tensor_started_ns) / 1e6,
        )

        control_payload = self._build_dedicated_tensor_control_payload(
            observation,
            env_ids,
            tensor_layout=prepared_payload.tensor_layout,
            tensor_nbytes=prepared_payload.original_nbytes,
            tensor_transport_info=transport.tensor_source_info(),
        )
        self._send_request("get_action", control_payload, requires_input=True)
        response = self._recv_get_action_response(
            timeout_message=(
                "ZMQ recv timed out during Mooncake get_action. "
                "Mooncake session is likely stale — call start_new_session()."
            ),
            must_start_new_session=True,
        )
        record_remote_metric(
            "client_get_action_total",
            path="dedicated_tensor_transport",
            transport_mode=self._transport_mode.value,
            total_ms=(time.perf_counter_ns() - total_started_ns) / 1e6,
            original_nbytes=prepared_payload.original_nbytes,
        )
        return response

    def set_task_description(
        self,
        task_description: str | None,
        env_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_description": task_description}
        if env_ids is not None:
            payload["env_ids"] = env_ids

        response = self.call_endpoint("set_task_description", data=payload, requires_input=True)
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict from set_task_description, got {type(response)!r}")
        return response

    def reset(self, env_ids: list[int] | None = None, options: dict[str, Any] | None = None) -> Any:
        response = self.call_endpoint(
            endpoint="reset",
            data={"env_ids": env_ids, "options": options},
            requires_input=True,
        )
        if isinstance(response, dict):
            status = response.get("status")
            if status not in ("reset_success", "ok", "reset_ok", None):
                raise RuntimeError(f"Remote reset failed with status={status}, resp={response}")
        return response

    def start_new_session(self, num_envs: int | None = None, requested_action_mode: str | None = None) -> dict[str, Any]:
        """Start a fresh logical session."""
        n = num_envs if num_envs is not None else (self._num_envs or 1)
        action_mode = requested_action_mode or self._last_requested_action_mode or "chunk"

        with suppress(Exception):
            self._transport.reset_comm_backend()

        transport = cast(Any, self._transport)
        if hasattr(transport, "reset_identity"):
            transport.reset_identity()

        self._session_initialized = False
        self._transport.rebuild()
        return self.initialize_session(n, action_mode)

    def disconnect(self) -> Any:
        """Disconnect this client from the server."""
        return self.call_endpoint("disconnect", requires_input=False)

    def kill(self) -> Any:
        """Ask remote server to stop main loop."""
        return self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
    ) -> Any:
        self._send_request(endpoint, data, requires_input=requires_input)

        try:
            raw = self._transport.recv()
        except TimeoutError:
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
        self._transport_connected = False
        self._session_initialized = False

    @property
    def transport_mode(self) -> str:
        return self._transport_mode.value

    @property
    def tensor_device(self) -> str | None:
        return self._tensor_device

    @property
    def session_initialized(self) -> bool:
        return self._session_initialized
