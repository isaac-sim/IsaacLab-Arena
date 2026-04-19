# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from isaaclab_arena.remote_policy.client_state import ClientState
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.metrics import record_remote_metric
from isaaclab_arena.remote_policy.mooncake_config import autodetect_local_hostname
from isaaclab_arena.remote_policy.protocol_enums import TransportMode
from isaaclab_arena.remote_policy.profiling import nvtx_range
from isaaclab_arena.remote_policy.remote_policy_config import MooncakeTransportConfig
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena.remote_policy.transport.base import ServerTransport
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqServerTransport


@dataclass
class EndpointHandler:
    handler: Callable[..., Any]
    requires_input: bool = True


class ServerTransportTimeoutError(TimeoutError):
    """Structured server-side transport timeout for client-facing recovery."""

    def __init__(self, message: str, *, must_reconnect: bool, source: str) -> None:
        super().__init__(message)
        self.must_reconnect = must_reconnect
        self.source = source


class SessionRequiredError(RuntimeError):
    """Structured server-side error indicating there is no active session."""

    def __init__(self, message: str, *, source: str) -> None:
        super().__init__(message)
        self.source = source


class PolicyServer:
    """Policy server with a simple explicit handshake."""

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
        return str(parsed)

    @staticmethod
    def _has_mooncake_runtime() -> bool:
        try:
            import mooncake.engine  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    @staticmethod
    def _create_default_transport(
        timeout_ms: int,
        *,
        bind_host: str,
        transport_mode: str,
        tensor_device: str | None,
        mooncake_config: MooncakeTransportConfig | None,
    ) -> ServerTransport:
        """Create the production transport from explicit config."""
        mooncake = mooncake_config or MooncakeTransportConfig()
        requested_mode = TransportMode.parse(transport_mode)

        if requested_mode == TransportMode.ZMQ:
            print("[PolicyServer] transport_mode='zmq' — using ZMQ-only transport")
            return ZmqServerTransport(timeout_ms=timeout_ms)

        if requested_mode == TransportMode.ZMQ_UCX:
            raise RuntimeError(
                "transport_mode='zmq_ucx' is a legacy/debug path and is no longer available through the mainline "
                "PolicyServer constructor."
            )

        if requested_mode != TransportMode.ZMQ_MOONCAKE:
            raise ValueError(f"Unsupported transport_mode={transport_mode!r}")

        if not PolicyServer._has_mooncake_runtime():
            raise RuntimeError(
                "transport_mode='zmq_mooncake' was requested but the Mooncake runtime is unavailable."
            )

        resolved_local_hostname = mooncake.local_hostname or autodetect_local_hostname(bind_host)
        if not resolved_local_hostname:
            raise RuntimeError(
                "transport_mode='zmq_mooncake' requires a local hostname/IP that peers can reach. "
                "Pass --mooncake_local_hostname to override."
            )

        from isaaclab_arena.remote_policy.transport.zmq_mooncake_transport import ZmqMooncakeServerTransport

        print("[PolicyServer] transport_mode='zmq_mooncake' — using ZMQ+Mooncake transport")
        return ZmqMooncakeServerTransport(
            timeout_ms=timeout_ms,
            local_hostname=resolved_local_hostname,
            metadata_server=mooncake.metadata_backend,
            protocol=mooncake.protocol,
            device_name=mooncake.device_name or "",
            buffer_bytes=mooncake.staging_buffer_bytes,
            tensor_device=tensor_device,
            cuda_device_override=mooncake.cuda_device_override,
            force_register=mooncake.force_register,
        )

    def __init__(
        self,
        policy: ServerSidePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
        timeout_ms: int = 15000,
        idle_timeout_s: float = 600.0,
        allow_remote_kill: bool = False,
        transport_mode: str = "zmq",
        tensor_device: str | None = None,
        mooncake_config: MooncakeTransportConfig | None = None,
    ) -> None:
        self._initialize_with_transport(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            transport=self._create_default_transport(
                timeout_ms=timeout_ms,
                bind_host=host,
                transport_mode=transport_mode,
                tensor_device=tensor_device,
                mooncake_config=mooncake_config,
            ),
            idle_timeout_s=idle_timeout_s,
            allow_remote_kill=allow_remote_kill,
            tensor_device=tensor_device,
        )

    @classmethod
    def _from_transport_for_testing(
        cls,
        policy: ServerSidePolicy,
        transport: ServerTransport,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
        idle_timeout_s: float = 600.0,
        allow_remote_kill: bool = False,
        tensor_device: str | None = None,
    ) -> PolicyServer:
        server = cls.__new__(cls)
        server._initialize_with_transport(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            transport=transport,
            idle_timeout_s=idle_timeout_s,
            allow_remote_kill=allow_remote_kill,
            tensor_device=tensor_device,
        )
        return server

    def _initialize_with_transport(
        self,
        policy: ServerSidePolicy,
        host: str,
        port: int,
        api_token: str | None,
        transport: ServerTransport,
        idle_timeout_s: float,
        allow_remote_kill: bool,
        tensor_device: str | None,
    ) -> None:
        self._policy = policy
        self._running = True
        self._api_token = api_token
        self._idle_timeout_s = idle_timeout_s
        self._allow_remote_kill = allow_remote_kill
        self._transport = transport
        self._transport_mode = TransportMode.parse(transport.transport_mode)
        self._tensor_device = self._normalize_tensor_device(tensor_device)

        if self._transport_mode == TransportMode.ZMQ_MOONCAKE and self._tensor_device in (None, "cpu"):
            raise RuntimeError(
                "transport_mode='zmq_mooncake' requires policy_device to resolve to a CUDA device on the server."
            )

        self._client_states: dict[bytes, ClientState] = {}
        self._last_seen: dict[bytes, float] = {}

        bind_addr = f"tcp://{host}:{port}"
        print(f"[PolicyServer] binding on {bind_addr}")
        self._transport.bind(bind_addr)

        self._endpoints: dict[str, EndpointHandler] = {}
        self._register_default_endpoints()

    def _register_default_endpoints(self) -> None:
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("disconnect", self._handle_disconnect, requires_input=False)
        self.register_endpoint("kill", self._handle_kill, requires_input=False)
        self.register_endpoint("get_action", self._handle_get_action, requires_input=True)
        self.register_endpoint("reset", self._handle_reset, requires_input=True)
        self.register_endpoint("get_init_info", self._handle_get_init_info, requires_input=True)
        self.register_endpoint("set_task_description", self._handle_set_task_description, requires_input=True)
        print(f"[PolicyServer] registered endpoints: {list(self._endpoints.keys())}")

    def register_endpoint(
        self,
        name: str,
        handler: Callable[..., Any],
        requires_input: bool = True,
    ) -> None:
        self._endpoints[name] = EndpointHandler(handler=handler, requires_input=requires_input)

    @staticmethod
    def _client_state_required_message(zmq_identity: bytes, endpoint: str) -> str:
        return (
            f"No active session for {zmq_identity.hex()[:8]} on endpoint={endpoint!r}. "
            "Client must call initialize_session() to start a new session."
        )

    def _require_client_state(self, zmq_identity: bytes, endpoint: str) -> ClientState:
        client_state = self._client_states.get(zmq_identity)
        if client_state is None:
            raise SessionRequiredError(
                self._client_state_required_message(zmq_identity, endpoint),
                source="session_state",
            )
        return client_state

    def _create_client_state(self, zmq_identity: bytes, *, num_envs: int) -> None:
        self._client_states[zmq_identity] = ClientState.create(num_envs)
        self._last_seen[zmq_identity] = time.monotonic()

    def _drop_client_state(self, zmq_identity: bytes) -> None:
        self._client_states.pop(zmq_identity, None)
        self._last_seen.pop(zmq_identity, None)
        self._transport.disconnect_client(zmq_identity)

    def _mark_seen(self, zmq_identity: bytes) -> None:
        if zmq_identity in self._client_states:
            self._last_seen[zmq_identity] = time.monotonic()

    def _send_error_response(
        self,
        zmq_identity: bytes,
        message: str,
        *,
        must_reconnect: bool = False,
        source: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"error": message}
        if must_reconnect or source is not None:
            payload["must_reconnect"] = must_reconnect
            payload["source"] = source or "unknown"
        self._transport.send(zmq_identity, MessageSerializer.to_bytes(payload))

    @staticmethod
    def _build_handshake_reject_response(
        message: str,
        **details: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "rejected",
            "message": message,
            "error": message,
            "source": "handshake",
        }
        payload.update(details)
        return payload

    def _effective_cuda_tensor_device(self, *, context: str) -> str:
        if self._tensor_device is None or self._tensor_device == "cpu":
            raise RuntimeError(f"{context} requires policy_device to resolve to a CUDA device on the server.")
        return self._tensor_device

    @staticmethod
    def _validate_env_ids(
        client_state: ClientState,
        env_ids: list[int] | None,
        endpoint: str,
    ) -> None:
        if env_ids is None:
            return
        for idx, env_id in enumerate(env_ids):
            if not 0 <= env_id < client_state.num_envs:
                raise IndexError(
                    f"env_ids[{idx}]={env_id} out of range for num_envs={client_state.num_envs} "
                    f"on endpoint={endpoint!r}."
                )

    def _handle_ping(self, zmq_identity: bytes) -> dict[str, Any]:
        if zmq_identity in self._client_states:
            self._mark_seen(zmq_identity)
        return {"status": "ok"}

    def _handle_disconnect(self, zmq_identity: bytes) -> dict[str, Any]:
        self._drop_client_state(zmq_identity)
        return {"status": "disconnected"}

    def _handle_kill(self, zmq_identity: bytes) -> dict[str, Any]:
        self._require_client_state(zmq_identity, "kill")
        if not self._allow_remote_kill:
            return {"status": "rejected", "reason": "remote kill disabled"}
        self._running = False
        return {"status": "stopping"}

    def _handle_get_init_info(
        self,
        zmq_identity: bytes,
        requested_action_mode: str,
        num_envs: int = 1,
        transport_mode: str = "zmq",
        **_: Any,
    ) -> dict[str, Any]:
        print(
            f"[PolicyServer] handle get_init_info from {zmq_identity.hex()[:8]}: "
            f"requested_action_mode={requested_action_mode!r}, num_envs={num_envs}, "
            f"transport_mode={transport_mode!r}"
        )

        if zmq_identity in self._client_states:
            return self._build_handshake_reject_response(
                (
                    f"Handshake rejected: client {zmq_identity.hex()[:8]} already has an active session. "
                    "Do not call initialize_session() twice on a live session."
                ),
                client_id=zmq_identity.hex(),
            )

        requested_transport = TransportMode.parse(transport_mode)
        if requested_transport != self._transport_mode:
            return self._build_handshake_reject_response(
                (
                    "Handshake rejected: transport_mode mismatch. "
                    f"client={requested_transport.value!r}, server={self._transport_mode.value!r}. "
                    "Adjust the client config to match the server, or restart against a server configured "
                    "for the requested mode."
                ),
                client_transport_mode=requested_transport.value,
                server_transport_mode=self._transport_mode.value,
            )

        response = self._policy.get_init_info(requested_action_mode=requested_action_mode)
        if not isinstance(response, dict):
            raise TypeError(f"Policy.get_init_info() must return dict, got {type(response)!r}")

        if self._transport_mode != TransportMode.ZMQ:
            response.update(self._transport.get_handshake_metadata(zmq_identity))
        response["num_envs"] = num_envs
        response["zmq_identity"] = zmq_identity

        self._create_client_state(zmq_identity, num_envs=num_envs)
        return response

    def _validate_tensor_transport_request(
        self,
        zmq_identity: bytes,
        *,
        has_tensor: bool,
    ) -> None:
        if not has_tensor:
            return

        if self._transport_mode == TransportMode.ZMQ:
            raise RuntimeError(
                "Protocol error: has_tensor=True is not allowed on transport_mode='zmq'. "
                "Pure ZMQ sessions must use the inline_tensor_payload path."
            )

    def _validate_inline_tensor_request(
        self,
        zmq_identity: bytes,
        *,
        inline_tensor_layout: list[dict[str, Any]] | None,
        inline_tensor_payload: Any,
    ) -> None:
        if inline_tensor_layout is None and inline_tensor_payload is None:
            return
        if inline_tensor_layout is None or inline_tensor_payload is None:
            raise RuntimeError(
                "Protocol error: inline tensor payload requires both inline_tensor_layout and inline_tensor_payload."
            )

        if self._transport_mode != TransportMode.ZMQ:
            raise RuntimeError(
                "Protocol error: inline_tensor_payload is only supported on transport_mode='zmq'."
            )

    @staticmethod
    def _extract_inline_tensor_bytes(inline_tensor_payload: Any) -> bytes:
        if isinstance(inline_tensor_payload, bytes):
            return inline_tensor_payload
        if isinstance(inline_tensor_payload, dict):
            payload_bytes = inline_tensor_payload.get("data")
            if isinstance(payload_bytes, bytes):
                return payload_bytes
        raise RuntimeError(
            f"Protocol error: inline_tensor_payload must decode to bytes/blob, got {type(inline_tensor_payload)!r}."
        )

    def _handle_get_action(
        self,
        zmq_identity: bytes,
        observation: dict[str, Any],
        env_ids: list[int] | None = None,
        options: dict[str, Any] | None = None,
        has_tensor: bool = False,
        tensor_layout: list[dict[str, Any]] | None = None,
        tensor_nbytes: int = 0,
        tensor_transport_info: dict[str, Any] | None = None,
        inline_tensor_layout: list[dict[str, Any]] | None = None,
        inline_tensor_payload: Any = None,
        inline_tensor_nbytes: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        total_started_ns = time.perf_counter_ns()
        del inline_tensor_nbytes
        if not isinstance(observation, dict):
            raise TypeError(f"Expected dict observation, got {type(observation)!r}")

        client_state = self._require_client_state(zmq_identity, "get_action")
        self._validate_env_ids(client_state, env_ids, "get_action")
        self._validate_tensor_transport_request(
            zmq_identity,
            has_tensor=has_tensor,
        )
        self._validate_inline_tensor_request(
            zmq_identity,
            inline_tensor_layout=inline_tensor_layout,
            inline_tensor_payload=inline_tensor_payload,
        )
        self._mark_seen(zmq_identity)

        path = "control_only"
        if inline_tensor_layout and inline_tensor_payload is not None:
            path = "inline_zmq"
            observation = self._unpack_inline_tensor_payload(
                observation,
                inline_tensor_layout,
                inline_tensor_payload,
            )

        if has_tensor and tensor_layout and hasattr(self._transport, "recv_tensor"):
            path = "dedicated_tensor_transport"
            observation = self._recv_and_unpack_gpu_tensors(
                zmq_identity,
                observation,
                tensor_layout,
                tensor_nbytes,
                tensor_transport_info=tensor_transport_info,
            )

        policy_started_ns = time.perf_counter_ns()
        action, info = self._policy.get_action(
            observation=observation,
            options=options,
            env_ids=env_ids,
            client_state=client_state,
        )
        record_remote_metric(
            "server_policy_forward",
            endpoint="get_action",
            path=path,
            transport_mode=self._transport_mode.value,
            policy_class=type(self._policy).__name__,
            env_count=len(env_ids) if env_ids is not None else client_state.num_envs,
            policy_forward_ms=(time.perf_counter_ns() - policy_started_ns) / 1e6,
        )
        if not isinstance(action, dict):
            raise TypeError(f"Policy.get_action() must return (dict, dict), got action type={type(action)!r}")
        if not isinstance(info, dict):
            raise TypeError(f"Policy.get_action() must return (dict, dict), got info type={type(info)!r}")

        merged: dict[str, Any] = {}
        merged.update(action)
        merged.update(info)
        record_remote_metric(
            "server_get_action_total",
            path=path,
            transport_mode=self._transport_mode.value,
            has_tensor=has_tensor,
            has_inline_tensor=inline_tensor_layout is not None and inline_tensor_payload is not None,
            total_ms=(time.perf_counter_ns() - total_started_ns) / 1e6,
        )
        return merged

    def _unpack_inline_tensor_payload(
        self,
        observation: dict[str, Any],
        tensor_layout: list[dict[str, Any]],
        inline_tensor_payload: Any,
    ) -> dict[str, Any]:
        unpack_started_ns = time.perf_counter_ns()
        payload_bytes = self._extract_inline_tensor_bytes(inline_tensor_payload)
        raw_buffer = torch.frombuffer(memoryview(payload_bytes), dtype=torch.uint8).clone()

        for entry in tensor_layout:
            key = entry["key"]
            shape = entry["shape"]
            dtype = getattr(torch, entry["dtype"].replace("torch.", ""))
            offset = entry["offset"]
            nbytes = entry["nbytes"]
            tensor_bytes = raw_buffer[offset : offset + nbytes]
            tensor = tensor_bytes.view(dtype).reshape(shape)
            observation[key] = tensor.clone().numpy()

        record_remote_metric(
            "server_inline_tensor_unpack",
            transport_mode=self._transport_mode.value,
            payload_nbytes=len(payload_bytes),
            tensor_count=len(tensor_layout),
            unpack_ms=(time.perf_counter_ns() - unpack_started_ns) / 1e6,
        )
        return observation

    def _recv_and_unpack_gpu_tensors(
        self,
        zmq_identity: bytes,
        observation: dict[str, Any],
        tensor_layout: list[dict[str, Any]],
        tensor_nbytes: int,
        tensor_transport_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if tensor_transport_info is not None and hasattr(self._transport, "prepare_recv_tensor"):
            getattr(self._transport, "prepare_recv_tensor")(zmq_identity, tensor_transport_info)

        recv_started_ns = time.perf_counter_ns()
        try:
            with nvtx_range("mooncake.server_recv_tensor"):
                raw_buffer = self._transport.recv_tensor(zmq_identity, tensor_nbytes)
        except (TimeoutError, RuntimeError) as exc:
            raise ServerTransportTimeoutError(
                "Tensor transport recv_tensor failed while receiving get_action payload. "
                "Client should call start_new_session().",
                must_reconnect=True,
                source="mooncake_recv",
            ) from exc
        recv_elapsed_ms = (time.perf_counter_ns() - recv_started_ns) / 1e6

        raw_buffer = cast(torch.Tensor, raw_buffer)
        for entry in tensor_layout:
            key = entry["key"]
            shape = entry["shape"]
            dtype = getattr(torch, entry["dtype"].replace("torch.", ""))
            offset = entry["offset"]
            nbytes = entry["nbytes"]
            tensor_bytes = raw_buffer[offset : offset + nbytes]
            tensor = tensor_bytes.view(dtype).reshape(shape)
            observation[key] = tensor

        record_remote_metric(
            "server_dedicated_tensor_recv",
            transport_mode=self._transport_mode.value,
            tensor_nbytes=tensor_nbytes,
            tensor_count=len(tensor_layout),
            recv_tensor_ms=recv_elapsed_ms,
        )
        return observation

    def _handle_set_task_description(
        self,
        zmq_identity: bytes,
        task_description: str | None = None,
        env_ids: list[int] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        client_state = self._require_client_state(zmq_identity, "set_task_description")
        self._validate_env_ids(client_state, env_ids, "set_task_description")
        self._mark_seen(zmq_identity)
        response = self._policy.set_task_description(
            task_description,
            env_ids=env_ids,
            client_state=client_state,
        )
        if not isinstance(response, dict):
            raise TypeError(f"Policy.set_task_description() must return dict, got {type(response)!r}")
        return response

    def _handle_reset(
        self,
        zmq_identity: bytes,
        env_ids: list[int] | None = None,
        options: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        client_state = self._require_client_state(zmq_identity, "reset")
        self._validate_env_ids(client_state, env_ids, "reset")
        self._mark_seen(zmq_identity)

        status: dict[str, Any] = {"status": "reset_success"}
        response = self._policy.reset(env_ids=env_ids, reset_options=options, client_state=client_state)
        if isinstance(response, dict):
            status.update(response)
        return status

    def _gc_stale_clients(self) -> None:
        now = time.monotonic()
        stale_clients = [
            (zmq_identity, now - last_seen)
            for zmq_identity, last_seen in self._last_seen.items()
            if now - last_seen > self._idle_timeout_s
        ]
        for zmq_identity, idle_for_s in stale_clients:
            print(f"[PolicyServer] GC stale client {zmq_identity.hex()[:8]} (idle {idle_for_s:.0f}s)")
            self._drop_client_state(zmq_identity)

    def _validate_token(self, request: dict[str, Any]) -> bool:
        if self._api_token is None:
            return True
        return request.get("api_token") == self._api_token

    def _dispatch_single(self, zmq_identity: bytes, request: dict[str, Any]) -> None:
        try:
            if not self._validate_token(request):
                self._send_error_response(zmq_identity, "Unauthorized: invalid api_token")
                return

            if "endpoint" not in request:
                self._send_error_response(zmq_identity, "Missing 'endpoint' in request")
                return

            endpoint = request["endpoint"]
            handler = self._endpoints.get(endpoint)
            if handler is None:
                raise ValueError(f"Unknown endpoint: {endpoint}")

            data = request.get("data", {}) or {}
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict data, got {type(data)!r}")

            result = handler.handler(zmq_identity, **data) if handler.requires_input else handler.handler(zmq_identity)

            try:
                serialize_started_ns = time.perf_counter_ns()
                payload = MessageSerializer.to_bytes(result)
                serialize_elapsed_ms = (time.perf_counter_ns() - serialize_started_ns) / 1e6
                send_started_ns = time.perf_counter_ns()
                self._transport.send(zmq_identity, payload)
                send_elapsed_ms = (time.perf_counter_ns() - send_started_ns) / 1e6
                record_remote_metric(
                    "server_response_send",
                    endpoint=endpoint,
                    transport_mode=self._transport_mode.value,
                    response_nbytes=len(payload),
                    serialize_ms=serialize_elapsed_ms,
                    send_ms=send_elapsed_ms,
                )
            except Exception:
                if endpoint == "get_init_info":
                    self._drop_client_state(zmq_identity)
                raise

        except ServerTransportTimeoutError as exc:
            self._send_error_response(
                zmq_identity,
                str(exc),
                must_reconnect=exc.must_reconnect,
                source=exc.source,
            )
        except SessionRequiredError as exc:
            self._send_error_response(
                zmq_identity,
                str(exc),
                source=exc.source,
            )
        except Exception as exc:
            import traceback

            print(f"[PolicyServer] Error: {exc}")
            print(traceback.format_exc())
            try:
                self._send_error_response(zmq_identity, str(exc))
            except Exception:
                pass

    def run(self) -> None:
        print(f"[PolicyServer] listening, api_token={self._api_token!r}, mode='single-request'")
        gc_interval = max(self._idle_timeout_s / 4, 30.0)
        last_gc = time.monotonic()

        while self._running:
            zmq_identity: bytes | None = None
            try:
                recv_started_ns = time.perf_counter_ns()
                zmq_identity, raw = self._transport.recv()
                recv_elapsed_ms = (time.perf_counter_ns() - recv_started_ns) / 1e6
                decode_started_ns = time.perf_counter_ns()
                request = MessageSerializer.from_bytes(raw)
                decode_elapsed_ms = (time.perf_counter_ns() - decode_started_ns) / 1e6
                if not isinstance(request, dict):
                    raise TypeError(f"Expected dict request, got {type(request)!r}")
                record_remote_metric(
                    "server_request_received",
                    endpoint=request.get("endpoint"),
                    transport_mode=self._transport_mode.value,
                    request_nbytes=len(raw),
                    recv_ms=recv_elapsed_ms,
                    decode_ms=decode_elapsed_ms,
                )
                self._dispatch_single(zmq_identity, request)
            except TimeoutError:
                pass
            except Exception as exc:
                print(f"[PolicyServer] Error in main loop: {exc}")
                if zmq_identity is not None:
                    try:
                        self._send_error_response(
                            zmq_identity,
                            f"Malformed request before dispatch: {exc}",
                            source="request_decode",
                        )
                    except Exception:
                        pass

            if time.monotonic() - last_gc > gc_interval:
                self._gc_stale_clients()
                last_gc = time.monotonic()

    def close(self) -> None:
        self._running = False
        try:
            self._transport.close()
        except Exception:
            pass

    @staticmethod
    def start(
        policy: ServerSidePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
        timeout_ms: int = 15000,
        idle_timeout_s: float = 600.0,
        allow_remote_kill: bool = False,
        transport_mode: str = "zmq",
        tensor_device: str | None = None,
        mooncake_config: MooncakeTransportConfig | None = None,
    ) -> None:
        server = PolicyServer(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            timeout_ms=timeout_ms,
            idle_timeout_s=idle_timeout_s,
            allow_remote_kill=allow_remote_kill,
            transport_mode=transport_mode,
            tensor_device=tensor_device,
            mooncake_config=mooncake_config,
        )
        server.run()
