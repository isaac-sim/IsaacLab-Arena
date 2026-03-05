# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.remote_policy.client_state import ClientState
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena.remote_policy.transport.base import ServerTransport
from isaaclab_arena.remote_policy.transport.zmq_transport import ZmqServerTransport


@dataclass
class EndpointHandler:
    handler: Callable[..., Any]
    requires_input: bool = True


class PolicyServer:
    """v2 policy server with ROUTER/DEALER multi-client support.

    Changes from v1:
      - Uses ``ServerTransport`` (ROUTER by default) instead of raw zmq.REP.
      - Tracks per-client ``ClientState`` keyed by ZMQ identity.
      - Injects ``env_ids`` and ``client_state`` into policy method calls.
      - Garbage-collects stale clients after ``idle_timeout_s``.
    """

    def __init__(
        self,
        policy: ServerSidePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
        timeout_ms: int = 15000,
        transport: ServerTransport | None = None,
        idle_timeout_s: float = 600.0,
        max_batch_size: int = 1,
        batch_wait_ms: int = 5,
    ) -> None:
        self._policy = policy
        self._running = True
        self._api_token = api_token
        self._idle_timeout_s = idle_timeout_s
        self._max_batch_size = max_batch_size
        self._batch_wait_ms = batch_wait_ms

        # Per-client state
        self._client_states: dict[bytes, ClientState] = {}
        self._client_compression: dict[bytes, str] = {}  # ZMQ message compression
        self._client_tensor_compression: dict[bytes, str] = {}  # UCX tensor compression
        self._last_seen: dict[bytes, float] = {}

        # Transport
        if transport is not None:
            self._transport = transport
        else:
            self._transport = ZmqServerTransport(timeout_ms=timeout_ms)

        # Detect capabilities once at init
        self._has_ucx = False
        if self._transport.transport_mode == "zmq_ucx":
            self._has_ucx = True
        else:
            try:
                import ucp  # noqa: F401
                self._has_ucx = True
            except ImportError:
                pass

        self._has_nvcomp = False
        try:
            from nvidia.nvcomp import Codec  # noqa: F401
            self._has_nvcomp = True
        except ImportError:
            pass

        bind_addr = f"tcp://{host}:{port}"
        print(f"[PolicyServer] binding on {bind_addr}")
        self._transport.bind(bind_addr)

        self._endpoints: dict[str, EndpointHandler] = {}
        self._register_default_endpoints()

    def _register_default_endpoints(self) -> None:
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
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
    # Endpoint handlers
    # ------------------------------------------------------------------

    def _handle_ping(self, client_id: bytes) -> dict[str, Any]:
        print(f"[PolicyServer] handle ping from {client_id.hex()[:8]}")
        return {"status": "ok"}

    def _handle_kill(self, client_id: bytes) -> dict[str, Any]:
        print(f"[PolicyServer] handle kill from {client_id.hex()[:8]} -> stopping")
        self._running = False
        return {"status": "stopping"}

    def _handle_get_init_info(
        self,
        client_id: bytes,
        requested_action_mode: str,
        num_envs: int = 1,
        transport_capabilities: list[str] | None = None,
        compression_capabilities: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(
            f"[PolicyServer] handle get_init_info from {client_id.hex()[:8]}: "
            f"requested_action_mode={requested_action_mode!r}, num_envs={num_envs}"
        )

        # Create per-client state
        self._client_states[client_id] = ClientState.create(num_envs)
        self._last_seen[client_id] = time.monotonic()

        # Negotiate compression
        # ZMQ path: lz4 CPU compression via MessageSerializer
        # UCX path: nvcomp_lz4 GPU compression (handled separately in tensor send/recv)
        client_comp = set(compression_capabilities or [])
        server_comp = {"none", "lz4"}
        if self._has_nvcomp:
            server_comp.add("nvcomp_lz4")

        # For ZMQ message-level compression, prefer lz4
        zmq_compression = "none"
        for pref in ["lz4", "none"]:
            if pref in client_comp and pref in server_comp:
                zmq_compression = pref
                break
        self._client_compression[client_id] = zmq_compression

        # For UCX tensor-level compression, prefer nvcomp_lz4
        tensor_compression = "none"
        for pref in ["nvcomp_lz4", "none"]:
            if pref in client_comp and pref in server_comp:
                tensor_compression = pref
                break

        self._client_tensor_compression[client_id] = tensor_compression

        # Report the best overall compression to the client
        negotiated_compression = tensor_compression if tensor_compression != "none" else zmq_compression

        # Negotiate transport (prefer zmq_ucx > zmq)
        client_trans = set(transport_capabilities or [])
        server_trans = {"zmq"}
        if self._has_ucx:
            server_trans.add("zmq_ucx")
        negotiated_transport = "zmq"
        for pref in ["zmq_ucx", "zmq"]:
            if pref in client_trans and pref in server_trans:
                negotiated_transport = pref
                break

        # Delegate to policy
        resp = self._policy.get_init_info(requested_action_mode=requested_action_mode)
        if not isinstance(resp, dict):
            raise TypeError(f"Policy.get_init_info() must return dict, got {type(resp)!r}")

        # If UCX negotiated, start listener and return port
        ucx_port = None
        if negotiated_transport == "zmq_ucx" and hasattr(self._transport, "start_ucx_listener"):
            if self._transport.ucx_port is None:
                ucx_port = self._transport.start_ucx_listener()
            else:
                ucx_port = self._transport.ucx_port

        # Inject negotiation results — send both ZMQ and tensor compression
        # so the client can use the correct one for each path
        resp["negotiated_compression"] = negotiated_compression  # backward compat
        resp["negotiated_zmq_compression"] = zmq_compression
        resp["negotiated_tensor_compression"] = tensor_compression
        resp["negotiated_transport"] = negotiated_transport
        resp["num_envs"] = num_envs
        if ucx_port is not None:
            resp["ucx_port"] = ucx_port
        # Send the ZMQ identity so the client can use it as UCX client_id
        # (ensures UCX endpoint mapping matches ZMQ routing identity)
        resp["zmq_identity"] = client_id
        return resp

    def _handle_get_action(
        self,
        client_id: bytes,
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
        print(f"[PolicyServer] handle get_action from {client_id.hex()[:8]}")
        if options is not None:
            print(f"  options keys: {list(options.keys())}")

        # UCX tensor path: receive GPU tensor and merge into observation
        if has_tensor and tensor_layout and hasattr(self._transport, "recv_tensor"):
            observation = self._recv_and_unpack_gpu_tensors(
                client_id, observation, tensor_layout,
                tensor_nbytes, tensor_original_nbytes, tensor_compressed,
            )

        client_state = self._client_states.get(client_id)

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
        if any(k in merged for k in info.keys()):
            raise ValueError(f"Policy info keys conflict with action keys: {set(merged.keys()) & set(info.keys())}")
        merged.update(info)
        return merged

    def _recv_and_unpack_gpu_tensors(
        self,
        client_id: bytes,
        observation: dict[str, Any],
        tensor_layout: list[dict[str, Any]],
        tensor_nbytes: int,
        tensor_original_nbytes: int,
        tensor_compressed: bool,
    ) -> dict[str, Any]:
        """Receive a GPU tensor via UCX and unpack into observation dict."""
        import torch

        # Receive the flat GPU buffer
        raw_buffer = self._transport.recv_tensor(client_id, tensor_nbytes)

        # Decompress if nvcomp was used
        if tensor_compressed:
            from isaaclab_arena.remote_policy.gpu_compression import gpu_decompress
            raw_buffer = gpu_decompress(raw_buffer, tensor_original_nbytes)

        # Unpack individual tensors from the flat buffer
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
        client_id: bytes,
        task_description: str | None = None,
        env_ids: list[int] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle set_task_description from {client_id.hex()[:8]}: {task_description!r}")
        client_state = self._client_states.get(client_id)
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
        client_id: bytes,
        env_ids: list[int] | None = None,
        options: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        print(f"[PolicyServer] handle reset from {client_id.hex()[:8]}: env_ids={env_ids}, options={options}")
        client_state = self._client_states.get(client_id)

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
        stale = [cid for cid, ts in self._last_seen.items() if now - ts > self._idle_timeout_s]
        for cid in stale:
            print(f"[PolicyServer] GC stale client {cid.hex()[:8]} (idle {now - self._last_seen[cid]:.0f}s)")
            self._client_states.pop(cid, None)
            self._client_compression.pop(cid, None)
            self._client_tensor_compression.pop(cid, None)
            self._last_seen.pop(cid, None)
            self._transport.disconnect_client(cid)

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

    # ------------------------------------------------------------------
    # Batch processing (Phase 7)
    # ------------------------------------------------------------------

    def _collect_batch(self) -> list[tuple[bytes, dict[str, Any]]]:
        """Collect up to ``max_batch_size`` get_action requests for batching.

        Blocks for the first request.  If the first request is NOT get_action
        or ``max_batch_size == 1``, returns immediately with that single request.
        Otherwise polls for up to ``batch_wait_ms`` to collect more requests,
        short-circuiting when we have enough or all known clients have sent.
        """
        batch: list[tuple[bytes, dict[str, Any]]] = []

        # Block for the first request
        client_id, raw = self._transport.recv()  # may raise TimeoutError
        self._last_seen[client_id] = time.monotonic()
        request = MessageSerializer.from_bytes(raw)
        if not isinstance(request, dict):
            raise TypeError(f"Expected dict request, got {type(request)!r}")
        batch.append((client_id, request))

        # Only batch get_action requests
        if request.get("endpoint") != "get_action" or self._max_batch_size <= 1:
            return batch

        # Try to collect more within batch_wait_ms
        deadline = time.monotonic() + self._batch_wait_ms / 1000.0
        num_active = len(self._client_states)

        while len(batch) < self._max_batch_size and time.monotonic() < deadline:
            # Short-circuit: collected from all active clients
            if num_active > 0 and len(batch) >= num_active:
                break
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                cid2, raw2 = self._transport.recv_with_timeout(remaining_ms)
                self._last_seen[cid2] = time.monotonic()
                req2 = MessageSerializer.from_bytes(raw2)
                if isinstance(req2, dict) and req2.get("endpoint") == "get_action":
                    batch.append((cid2, req2))
                else:
                    # Non-get_action request — process it immediately, don't batch
                    batch.append((cid2, req2))
                    break
            except TimeoutError:
                break

        return batch

    def _process_batch(self, batch: list[tuple[bytes, dict[str, Any]]]) -> None:
        """Process a collected batch of requests.

        All requests (including get_action) are dispatched individually
        to preserve per-client state isolation (ClientState, env_ids).
        Batch collection reduces ZMQ polling overhead when multiple clients
        send requests within the ``batch_wait_ms`` window.
        """
        # Separate get_action vs other requests
        action_reqs = []
        other_reqs = []
        for client_id, request in batch:
            if request.get("endpoint") == "get_action":
                action_reqs.append((client_id, request))
            else:
                other_reqs.append((client_id, request))

        # Handle non-get_action requests individually
        for client_id, request in other_reqs:
            self._dispatch_single(client_id, request)

        if not action_reqs:
            return

        # Dispatch all get_action requests individually to preserve
        # per-client state isolation (ClientState, env_ids).
        # Batch collection reduces ZMQ polling overhead; the policy
        # is called once per client to maintain correct state semantics.
        # TODO: True batched inference (concat obs → single policy call →
        # split results) is a future optimization once per-client state
        # handling in the policy interface is resolved.
        for client_id, request in action_reqs:
            self._dispatch_single(client_id, request)

    def _dispatch_single(self, client_id: bytes, request: dict[str, Any]) -> None:
        """Dispatch a single request (non-batch path)."""
        try:
            if not self._validate_token(request):
                resp_bytes = MessageSerializer.to_bytes({"error": "Unauthorized: invalid api_token"})
                self._transport.send(client_id, resp_bytes)
                return

            if "endpoint" not in request:
                resp_bytes = MessageSerializer.to_bytes({"error": "Missing 'endpoint' in request"})
                self._transport.send(client_id, resp_bytes)
                return

            endpoint = request["endpoint"]
            handler = self._endpoints.get(endpoint)
            if handler is None:
                raise ValueError(f"Unknown endpoint: {endpoint}")
            print(f"[PolicyServer] dispatch endpoint='{endpoint}' for {client_id.hex()[:8]}")

            data = request.get("data", {}) or {}
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict data, got {type(data)!r}")

            if handler.requires_input:
                result = handler.handler(client_id, **data)
            else:
                result = handler.handler(client_id)

            compression = self._client_compression.get(client_id, "none")
            resp_bytes = MessageSerializer.to_bytes(result, compression_method=compression)
            print(f"[PolicyServer] sending response ({len(resp_bytes)} bytes) to {client_id.hex()[:8]}")
            self._transport.send(client_id, resp_bytes)

        except Exception as exc:
            import traceback
            print(f"[PolicyServer] Error: {exc}")
            print(traceback.format_exc())
            try:
                self._transport.send(client_id, MessageSerializer.to_bytes({"error": str(exc)}))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"[PolicyServer] listening, api_token={self._api_token!r}, max_batch_size={self._max_batch_size}")
        gc_interval = max(self._idle_timeout_s / 4, 30.0)
        last_gc = time.monotonic()

        while self._running:
            try:
                if self._max_batch_size > 1:
                    # Batch path
                    batch = self._collect_batch()
                    self._process_batch(batch)
                else:
                    # Single-request path (zero batching overhead)
                    client_id, raw = self._transport.recv()
                    self._last_seen[client_id] = time.monotonic()
                    print(f"[PolicyServer] received {len(raw)} bytes from {client_id.hex()[:8]}")

                    request = MessageSerializer.from_bytes(raw)
                    if not isinstance(request, dict):
                        raise TypeError(f"Expected dict request, got {type(request)!r}")

                    self._dispatch_single(client_id, request)

            except TimeoutError:
                pass
            except Exception as exc:
                print(f"[PolicyServer] Error in main loop: {exc}")

            # Periodic GC
            if time.monotonic() - last_gc > gc_interval:
                self._gc_stale_clients()
                last_gc = time.monotonic()

    def close(self) -> None:
        """Stop the main loop and close transport resources.

        Closing the transport will cause a blocked ``recv()`` to raise,
        breaking out of the main loop.
        """
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
        max_batch_size: int = 1,
        batch_wait_ms: int = 5,
    ) -> None:
        server = PolicyServer(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token,
            timeout_ms=timeout_ms,
            idle_timeout_s=idle_timeout_s,
            max_batch_size=max_batch_size,
            batch_wait_ms=batch_wait_ms,
        )
        server.run()
