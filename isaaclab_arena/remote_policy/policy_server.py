# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from isaaclab_arena.remote_policy.client_state import ClientState
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
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


class ReconnectRequiredError(RuntimeError):
    """Structured server-side error indicating the client must reconnect."""

    def __init__(self, message: str, *, source: str) -> None:
        super().__init__(message)
        self.source = source


class PolicyServer:
    """v2 policy server with ROUTER/DEALER multi-client support.

    Changes from v1:
      - Uses ``ServerTransport`` (ROUTER by default) instead of raw zmq.REP.
      - Tracks per-client ``ClientState`` keyed by ZMQ identity.
      - Injects ``env_ids`` and ``client_state`` into policy method calls.
      - Garbage-collects stale clients after ``idle_timeout_s``.
    """

    @staticmethod
    def _has_ucx_runtime() -> bool:
        try:
            import ucp  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _create_default_transport(timeout_ms: int) -> ServerTransport:
        """Create the production transport from locally available runtimes."""
        if PolicyServer._has_ucx_runtime():
            from isaaclab_arena.remote_policy.transport.zmq_ucx_transport import ZmqUcxServerTransport

            print("[PolicyServer] auto-detected UCX — using ZMQ+UCX transport")
            return ZmqUcxServerTransport(timeout_ms=timeout_ms)

        print("[PolicyServer] UCX not available — using ZMQ-only transport")
        return ZmqServerTransport(timeout_ms=timeout_ms)

    def __init__(
        self,
        policy: ServerSidePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
        timeout_ms: int = 15000,
        idle_timeout_s: float = 600.0,
        allow_remote_kill: bool = False,
    ) -> None:
        self._initialize_with_transport(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            transport=self._create_default_transport(timeout_ms=timeout_ms),
            idle_timeout_s=idle_timeout_s,
            allow_remote_kill=allow_remote_kill,
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
    ) -> PolicyServer:
        """Create a server with an injected transport for tests/benchmarks only.

        Production code should call ``PolicyServer(...)`` so transport
        auto-detection stays behind the public constructor. Configure timeout
        behavior on the injected transport itself.
        """
        server = cls.__new__(cls)
        server._initialize_with_transport(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            transport=transport,
            idle_timeout_s=idle_timeout_s,
            allow_remote_kill=allow_remote_kill,
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
    ) -> None:
        self._policy = policy
        self._running = True
        self._api_token = api_token
        self._idle_timeout_s = idle_timeout_s
        self._allow_remote_kill = allow_remote_kill

        # Per-client state keyed by ZMQ ROUTER identity.
        self._client_states: dict[bytes, ClientState] = {}
        self._client_compression: dict[bytes, str] = {}  # ZMQ message compression
        self._client_tensor_compression: dict[bytes, str] = {}  # UCX tensor compression
        self._last_seen: dict[bytes, float] = {}

        # Detect capabilities once at init
        self._has_ucx = self._has_ucx_runtime()

        self._has_nvcomp = False
        try:
            from nvidia.nvcomp import Codec  # noqa: F401
            self._has_nvcomp = True
        except ImportError:
            pass

        self._has_lz4 = False
        try:
            import lz4.frame  # noqa: F401
            self._has_lz4 = True
        except ImportError:
            pass

        self._transport = transport
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

    # ------------------------------------------------------------------
    # Client state lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _client_state_required_message(zmq_identity: bytes, endpoint: str) -> str:
        return (
            f"No ClientState for {zmq_identity.hex()[:8]} on endpoint={endpoint!r}. "
            "Client must call connect() / get_init_info first."
        )

    def _require_client_state(self, zmq_identity: bytes, endpoint: str) -> ClientState:
        """Look up ClientState or raise if the client is not initialized.

        After ``disconnect()`` or stale-client GC, the server no longer
        holds state for the client.  Rather than silently falling back to
        v1 legacy behavior, we return an error so the client knows to
        reconnect via ``get_init_info`` / ``connect()``.
        """
        cs = self._client_states.get(zmq_identity)
        if cs is None:
            raise ReconnectRequiredError(
                self._client_state_required_message(zmq_identity, endpoint),
                source="client_state",
            )
        return cs

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
        compression = self._client_compression.get(zmq_identity, "none")
        self._transport.send(
            zmq_identity,
            MessageSerializer.to_bytes(payload, compression_method=compression),
        )

    @staticmethod
    def _infer_observation_batch_size(observation: dict[str, Any]) -> int:
        """Historical helper for shape-based batch-size inference.

        NOTE:
        The shared v2 server no longer relies on this helper in the production
        ``get_action`` path.  The reason is that a generic observation dict does
        not guarantee that ``value.shape[0]`` always means "env batch size":
        some policies may send tensors whose leading dimension is time/history,
        sensor-specific structure, or another policy-local convention.

        We keep the helper for future policy-specific validators and for review
        discussion context, but the shared infra currently avoids enforcing a
        transport-level "first dimension must equal env batch" rule.
        """
        batch_size: int | None = None
        for value in observation.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                current = int(value.shape[0])
                if batch_size is None:
                    batch_size = current
                elif current != batch_size:
                    raise ValueError(
                        f"Inconsistent observation batch dimensions: saw {batch_size} and {current}."
                    )
        return batch_size if batch_size is not None else 1

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

    def _validate_get_action_targets(
        self,
        client_state: ClientState,
        observation: dict[str, Any],
        env_ids: list[int] | None,
        endpoint: str,
    ) -> None:
        """Validate explicit env targeting without inferring batch semantics.

        Current shared-infra policy:
        - If the client explicitly provides ``env_ids``, we still validate that
          those ids are within the ``ClientState.num_envs`` range.
        - We intentionally do *not* infer an observation batch size from
          ``observation.shape[0]`` here.  That inference is not universally
          correct across policies, and enforcing it at the transport/shared
          server layer can reject otherwise valid policy-specific layouts.

        In other words, the shared server validates explicit indices, but it
        leaves deeper "does this observation layout match these env targets?"
        checks to policy-specific code or to a future protocol field that
        describes batch semantics explicitly.
        """
        del observation
        if env_ids is not None:
            self._validate_env_ids(client_state, env_ids, endpoint)

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    def _handle_ping(self, zmq_identity: bytes) -> dict[str, Any]:
        print(f"[PolicyServer] handle ping from {zmq_identity.hex()[:8]}")
        if zmq_identity in self._client_states:
            self._mark_seen(zmq_identity)
        return {"status": "ok"}

    def _handle_disconnect(self, zmq_identity: bytes) -> dict[str, Any]:
        """Disconnect a single client: clean up its state only."""
        print(f"[PolicyServer] handle disconnect from {zmq_identity.hex()[:8]}")
        self._drop_client_state(zmq_identity)
        return {"status": "disconnected"}

    def _drop_client_state(self, zmq_identity: bytes) -> None:
        self._client_states.pop(zmq_identity, None)
        self._client_compression.pop(zmq_identity, None)
        self._client_tensor_compression.pop(zmq_identity, None)
        self._last_seen.pop(zmq_identity, None)
        self._transport.disconnect_client(zmq_identity)

    def _mark_seen(self, zmq_identity: bytes) -> None:
        self._last_seen[zmq_identity] = time.monotonic()

    def _handle_kill(self, zmq_identity: bytes) -> dict[str, Any]:
        """Shut down the entire server.  Gated by ``allow_remote_kill``."""
        self._require_client_state(zmq_identity, "kill")
        if not self._allow_remote_kill:
            print(
                f"[PolicyServer] WARN: kill from {zmq_identity.hex()[:8]} REJECTED "
                "(allow_remote_kill=False). Use 'disconnect' to clean up client state, "
                "or send SIGTERM to shut down the server."
            )
            return {"status": "rejected", "reason": "remote kill disabled"}
        print(f"[PolicyServer] handle kill from {zmq_identity.hex()[:8]} -> stopping")
        self._running = False
        return {"status": "stopping"}

    def _handle_get_init_info(
        self,
        zmq_identity: bytes,
        requested_action_mode: str,
        num_envs: int = 1,
        transport_capabilities: list[str] | None = None,
        compression_capabilities: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(
            f"[PolicyServer] handle get_init_info from {zmq_identity.hex()[:8]}: "
            f"requested_action_mode={requested_action_mode!r}, num_envs={num_envs}"
        )
        if zmq_identity in self._client_states:
            raise RuntimeError(
                f"Client {zmq_identity.hex()[:8]} is already initialized. "
                "Do not call get_init_info twice on a live session; reuse the current connection "
                "or reconnect with a fresh zmq_identity."
            )
        new_client_state = ClientState.create(num_envs)

        # Negotiate transport
        client_trans = set(transport_capabilities or [])
        server_trans = {"zmq"}
        if self._has_ucx:
            server_trans.add("zmq_ucx")
        negotiated_transport = "zmq"
        for pref in ["zmq_ucx", "zmq"]:
            if pref in client_trans and pref in server_trans:
                negotiated_transport = pref
                break

        # Negotiate compression. When tensors move over UCX, the remaining ZMQ
        # control payloads are typically small metadata messages, so default the
        # control plane back to no compression and reserve compression budget for
        # the tensor path.
        client_comp = set(compression_capabilities or [])
        server_comp = {"none"}
        if self._has_lz4:
            server_comp.add("lz4")
        if self._has_nvcomp:
            server_comp.add("nvcomp_lz4")

        zmq_compression = "none"
        if negotiated_transport != "zmq_ucx":
            for pref in ["lz4", "none"]:
                if pref in client_comp and pref in server_comp:
                    zmq_compression = pref
                    break

        tensor_compression = "none"
        for pref in ["nvcomp_lz4", "none"]:
            if pref in client_comp and pref in server_comp:
                tensor_compression = pref
                break

        negotiated_compression = tensor_compression if tensor_compression != "none" else zmq_compression

        resp = self._policy.get_init_info(requested_action_mode=requested_action_mode)
        if not isinstance(resp, dict):
            raise TypeError(f"Policy.get_init_info() must return dict, got {type(resp)!r}")

        ucx_port = None
        if negotiated_transport == "zmq_ucx":
            if not hasattr(self._transport, "start_ucx_listener"):
                raise RuntimeError(
                    "Protocol/config mismatch: negotiated_transport='zmq_ucx' but server "
                    f"transport {type(self._transport).__name__} does not support start_ucx_listener()."
                )
            existing_ucx_port = getattr(self._transport, "ucx_port", None)
            if existing_ucx_port is None:
                ucx_port = getattr(self._transport, "start_ucx_listener")()
            else:
                ucx_port = existing_ucx_port

        resp["negotiated_compression"] = negotiated_compression
        resp["negotiated_zmq_compression"] = zmq_compression
        resp["negotiated_tensor_compression"] = tensor_compression
        resp["negotiated_transport"] = negotiated_transport
        resp["num_envs"] = num_envs
        if ucx_port is not None:
            resp["ucx_port"] = ucx_port
        resp["zmq_identity"] = zmq_identity

        # Commit handshake state only after all steps succeed.
        self._client_states[zmq_identity] = new_client_state
        self._client_compression[zmq_identity] = zmq_compression
        self._client_tensor_compression[zmq_identity] = tensor_compression
        self._mark_seen(zmq_identity)
        return resp

    def _handle_get_action(
        self,
        zmq_identity: bytes,
        observation: dict[str, Any],
        env_ids: list[int] | None = None,
        options: dict[str, Any] | None = None,
        has_tensor: bool = False,
        tensor_layout: list[dict[str, Any]] | None = None,
        tensor_nbytes: int = 0,
        tensor_original_nbytes: int = 0,
        tensor_compressed: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle get_action from {zmq_identity.hex()[:8]}")
        if not isinstance(observation, dict):
            raise TypeError(f"Expected dict observation, got {type(observation)!r}")

        # UCX tensor path
        if has_tensor and tensor_layout and hasattr(self._transport, "recv_tensor"):
            observation = self._recv_and_unpack_gpu_tensors(
                zmq_identity, observation, tensor_layout,
                tensor_nbytes, tensor_original_nbytes, tensor_compressed,
            )

        client_state = self._require_client_state(zmq_identity, "get_action")
        self._validate_get_action_targets(client_state, observation, env_ids, "get_action")
        self._mark_seen(zmq_identity)

        action, info = self._policy.get_action(
            observation=observation,
            options=options,
            env_ids=env_ids,
            client_state=client_state,
        )

        if not isinstance(action, dict):
            raise TypeError(f"Policy.get_action() must return (dict, dict), got action type={type(action)!r}")
        if not isinstance(info, dict):
            raise TypeError(f"Policy.get_action() must return (dict, dict), got info type={type(info)!r}")

        merged: dict[str, Any] = {}
        merged.update(action)
        merged.update(info)
        return merged

    def _recv_and_unpack_gpu_tensors(
        self,
        zmq_identity: bytes,
        observation: dict[str, Any],
        tensor_layout: list[dict[str, Any]],
        tensor_nbytes: int,
        tensor_original_nbytes: int,
        tensor_compressed: bool,
    ) -> dict[str, Any]:
        """Receive a GPU tensor via UCX and unpack into observation dict."""
        import torch

        try:
            raw_buffer = self._transport.recv_tensor(zmq_identity, tensor_nbytes)
        except (TimeoutError, RuntimeError) as exc:
            raise ServerTransportTimeoutError(
                "UCX recv_tensor failed while receiving get_action payload. "
                "Client should call reconnect().",
                must_reconnect=True,
                source="ucx_recv",
            ) from exc

        if tensor_compressed:
            from isaaclab_arena.remote_policy.gpu_compression import gpu_decompress
            raw_buffer = gpu_decompress(raw_buffer, tensor_original_nbytes)
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

        return observation

    def _handle_set_task_description(
        self,
        zmq_identity: bytes,
        task_description: str | None = None,
        env_ids: list[int] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle set_task_description from {zmq_identity.hex()[:8]}: {task_description!r}")
        client_state = self._require_client_state(zmq_identity, "set_task_description")
        self._validate_env_ids(client_state, env_ids, "set_task_description")
        self._mark_seen(zmq_identity)
        resp = self._policy.set_task_description(
            task_description,
            env_ids=env_ids,
            client_state=client_state,
        )
        if not isinstance(resp, dict):
            raise TypeError(f"Policy.set_task_description() must return dict, got {type(resp)!r}")
        return resp

    def _handle_reset(
        self,
        zmq_identity: bytes,
        env_ids: list[int] | None = None,
        options: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle reset from {zmq_identity.hex()[:8]}: env_ids={env_ids}, options={options}")
        client_state = self._require_client_state(zmq_identity, "reset")
        self._validate_env_ids(client_state, env_ids, "reset")
        self._mark_seen(zmq_identity)

        status: dict[str, Any] = {"status": "reset_success"}
        resp = self._policy.reset(env_ids=env_ids, reset_options=options, client_state=client_state)
        if isinstance(resp, dict):
            status.update(resp)
        return status

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _gc_stale_clients(self) -> None:
        """Remove state for clients that haven't been seen within idle_timeout_s."""
        now = time.monotonic()
        stale = [zmq_identity for zmq_identity, ts in self._last_seen.items() if now - ts > self._idle_timeout_s]
        for zmq_identity in stale:
            print(
                f"[PolicyServer] GC stale client {zmq_identity.hex()[:8]} "
                f"(idle {now - self._last_seen[zmq_identity]:.0f}s)"
            )
            self._drop_client_state(zmq_identity)

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def _validate_token(self, request: dict[str, Any]) -> bool:
        if self._api_token is None:
            return True
        ok = request.get("api_token") == self._api_token
        if not ok:
            print("[PolicyServer] invalid api_token in request")
        return ok

    def _dispatch_single(self, zmq_identity: bytes, request: dict[str, Any]) -> None:
        """Dispatch a single request (non-batch path)."""
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
            print(f"[PolicyServer] dispatch endpoint='{endpoint}' for {zmq_identity.hex()[:8]}")

            data = request.get("data", {}) or {}
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict data, got {type(data)!r}")

            if handler.requires_input:
                result = handler.handler(zmq_identity, **data)
            else:
                result = handler.handler(zmq_identity)

            compression = self._client_compression.get(zmq_identity, "none")
            try:
                resp_bytes = MessageSerializer.to_bytes(result, compression_method=compression)
                self._transport.send(zmq_identity, resp_bytes)
            except Exception:
                if endpoint == "get_init_info":
                    self._drop_client_state(zmq_identity)
                raise

        except ServerTransportTimeoutError as exc:
            print(f"[PolicyServer] Transport timeout: {exc}")
            self._send_error_response(
                zmq_identity,
                str(exc),
                must_reconnect=exc.must_reconnect,
                source=exc.source,
            )
        except ReconnectRequiredError as exc:
            print(f"[PolicyServer] Reconnect required: {exc}")
            self._send_error_response(
                zmq_identity,
                str(exc),
                must_reconnect=True,
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"[PolicyServer] listening, api_token={self._api_token!r}, mode='single-request'")
        gc_interval = max(self._idle_timeout_s / 4, 30.0)
        last_gc = time.monotonic()

        while self._running:
            zmq_identity: bytes | None = None
            try:
                zmq_identity, raw = self._transport.recv()

                request = MessageSerializer.from_bytes(raw)
                if not isinstance(request, dict):
                    raise TypeError(f"Expected dict request, got {type(request)!r}")

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
    ) -> None:
        server = PolicyServer(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            timeout_ms=timeout_ms,
            idle_timeout_s=idle_timeout_s,
            allow_remote_kill=allow_remote_kill,
        )
        server.run()
