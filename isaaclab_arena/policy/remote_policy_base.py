# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import gymnasium as gym
import numpy as np
import threading
import torch
from abc import ABC, abstractmethod
from typing import Any

import websockets.exceptions
from openpi_client import websocket_client_policy

from isaaclab_arena.policy.action_scheduling import LatencyActionScheduler
from isaaclab_arena.policy.inference_latency import hold_action_from_action_manager, inference_latency_steps
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.policy.websocket_client import WebsocketClientPolicy
from isaaclab_arena.utils.timer import Timer

MAX_RECONNECT_ATTEMPTS = 3

NUM_INFERENCE_WORKERS = 4
"""Threads sending chunk requests concurrently when an inference latency is modeled, one connection each."""

BATCH_REQUEST_KEY = "batch"
"""Wire key under which a batched request (and its response) carries one entry per environment."""


class EmbodimentAdapter(ABC):
    """Translates between Arena's gym observation dict and an openpi-protocol server's
    wire format for a specific physical embodiment (DROID, ALOHA, ...).

    Subclasses declare the embodiment-specific action layout and observation keys.
    """

    action_dim: int

    @abstractmethod
    def extract(self, observation: dict[str, Any], env_id: int) -> Any:
        """Pull a single env's tensors gym observation dict and pack into a dataclass."""

    @abstractmethod
    def pack_request(self, extracted: Any, language_instruction: str) -> dict[str, Any]:
        """Build the wire-format request payload the server expects."""


class RemoteChunkReplayPolicy:
    """Mixin implementing openpi-protocol closed-loop control, parameterized by an embodiment adapter.

    Action handling is chunk replay: fetch one ``(chunk, action_dim)`` prediction
    per env, keep the first ``open_loop_horizon`` actions, and yield them in order before
    refetching. A ``LatencyActionScheduler`` decides when each env's request is sent and when its
    chunk becomes usable, according to ``config.inference_latency_ms``.

    The envs that request a chunk on the same step share one ``{"batch": [obs, ...]}`` request when
    ``config.batch_chunk_fetches`` is set and the server advertises ``batch_inference`` in its connect
    metadata, which the batched openpi server in ``isaaclab_arena_openpi/server`` does; otherwise each
    env gets its own request.

    Concrete policies inherit (RemoteChunkReplayPolicy, PolicyBase[ConcreteCfg]).
    The PolicyBase[ConcreteCfg] is the concrete config class for the policy, parameterized
    by a concrete Cfg class.
    """

    server_response_actions_key: str | None = None
    """Key under which the server returns the action chunk.

    Subclasses override this to match the server's wire format.
    """

    def __init__(self, config: PolicyCfg, adapter: EmbodimentAdapter, open_loop_horizon: int) -> None:
        super().__init__(config)  # PolicyBase.__init__ via the concrete class's MRO.
        assert self.server_response_actions_key is not None, (
            f"{type(self).__name__} must set server_response_actions_key to the key under which"
            " the remote server returns the action chunk"
        )
        self._adapter = adapter
        self._open_loop_horizon = open_loop_horizon
        self._batch_chunk_fetches: bool = getattr(config, "batch_chunk_fetches", True)
        self._max_batch_size: int | None = getattr(config, "max_batch_size", None)
        assert (
            self._max_batch_size is None or self._max_batch_size >= 1
        ), f"max_batch_size must be at least 1 or None, got {self._max_batch_size}"
        self.device = config.policy_device

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port
        self._ping_interval = config.ping_interval
        self._ping_timeout = config.ping_timeout

        name = type(self).__name__
        print(f"[{name}] Connecting to server at {self._remote_host}:{self._remote_port} ...")
        # One connection per thread that sends requests. The main thread's connection is opened
        # now so a missing server fails construction; worker threads open theirs on first use.
        self._thread_clients = threading.local()
        self._all_clients: list[WebsocketClientPolicy] = []
        self._clients_lock = threading.Lock()
        self._websocket_client: WebsocketClientPolicy | None = self._client_for_current_thread()
        print(f"[{name}] Connected.")
        if self._batch_chunk_fetches and not self.server_batches_requests():
            print(f"[{name}] Server does not advertise batch_inference; sending one request per environment.")
        # Server calls made so far, over every thread; compared with the chunks requested in the summary.
        self._server_calls = 0
        self._server_calls_lock = threading.Lock()

        # Per-env chunk bookkeeping. Created on the first get_action call, when num_envs and the
        # control step that converts the latency to steps are known.
        self._scheduler: LatencyActionScheduler | None = None
        self._flush_requested = False
        self.task_description: str | None = None

    @property
    def handles_inference_latency(self) -> bool:
        """The chunk scheduler applies ``config.inference_latency_ms`` itself."""
        return True

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            f"{type(self).__name__} requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )
        assert self._websocket_client is not None, f"{type(self).__name__} has been closed"

        scheduler = self._scheduler_for_env(env)
        if self._flush_requested:
            # A connection was re-established; the reconnected server may have lost state, so
            # every env requests a fresh chunk for its current observation.
            scheduler.reset()
            self._flush_requested = False

        hold_action = hold_action_from_action_manager(env) if scheduler.latency_steps > 0 else None
        return scheduler.get_action(functools.partial(self._prepare_requests, observation), hold_action=hold_action)

    def _prepare_requests(self, observation: dict[str, Any], env_ids: list[int]) -> list[functools.partial]:
        """Snapshot the given envs' observations and return the calls that fetch their chunks, in env order.

        The observation is read here, on the calling thread and for this step, because the environment
        overwrites its buffers on the next step while the requests are in flight. With a batching server
        the envs are grouped into requests of at most ``max_batch_size``; otherwise each env gets its own.
        """
        requests = [
            self._adapter.pack_request(self._adapter.extract(observation, env_id), self.task_description)
            for env_id in env_ids
        ]
        if not self.server_batches_requests():
            return [functools.partial(self._fetch_action_chunk, request) for request in requests]
        batch_size = self._batch_size_limit()
        return [
            functools.partial(self._fetch_action_chunks_batched, requests[start : start + batch_size])
            for start in range(0, len(requests), batch_size)
        ]

    def server_batches_requests(self) -> bool:
        """Return whether chunk requests of several envs go to the server as one batched request.

        True when ``config.batch_chunk_fetches`` is set and the connected server advertises
        ``batch_inference`` in its connect metadata.
        """
        if not self._batch_chunk_fetches or self._websocket_client is None:
            return False
        return bool(self._websocket_client.get_server_metadata().get("batch_inference", False))

    def _batch_size_limit(self) -> int:
        """Return the most envs one batched request may carry: the config limit or the server's advertised one."""
        assert self._websocket_client is not None
        server_limit = self._websocket_client.get_server_metadata().get("max_batch_size")
        if self._max_batch_size is not None and server_limit is not None:
            return min(self._max_batch_size, int(server_limit))
        if self._max_batch_size is not None:
            return self._max_batch_size
        assert server_limit is not None, "a server advertising batch_inference must advertise max_batch_size"
        return int(server_limit)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._scheduler is None:
            return  # not yet initialized; nothing to clear
        self._scheduler.reset(env_ids)

    def close(self) -> None:
        """Release the local websocket connections to the server.

        Does NOT stop the server process that runs in a separate container (or machine)
        and outlives this client.
        """
        if self._scheduler is not None:
            self._print_latency_summary()
            self._scheduler.close()
            self._scheduler = None
        with self._clients_lock:
            for client in self._all_clients:
                _close_websocket_best_effort(client)
            self._all_clients.clear()
        self._websocket_client = None

    def _scheduler_for_env(self, env: gym.Env) -> LatencyActionScheduler:
        """Return the chunk scheduler, creating it for the env's batch size and control step on first use."""
        num_envs = env.unwrapped.num_envs
        if self._scheduler is None:
            latency_steps = inference_latency_steps(self.config.inference_latency_ms, env.unwrapped.step_dt)
            self._scheduler = LatencyActionScheduler(
                num_envs=num_envs,
                action_chunk_length=self._open_loop_horizon,
                action_horizon=self._open_loop_horizon,
                action_dim=self._adapter.action_dim,
                device=self.device,
                latency_steps=latency_steps,
                max_workers=NUM_INFERENCE_WORKERS,
            )
            if latency_steps > 0:
                print(
                    f"[{type(self).__name__}] Modeling an inference latency of {self.config.inference_latency_ms} ms"
                    f" = {latency_steps} control steps"
                )
        assert self._scheduler.num_envs == num_envs, (
            f"{type(self).__name__} num_envs changed from {self._scheduler.num_envs}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )
        return self._scheduler

    def _fetch_action_chunk(self, request: dict[str, Any]) -> torch.Tensor:
        """Send one env's request and return its ``(1, open_loop_horizon, action_dim)`` chunk.

        Runs on a worker thread when a latency is modeled, otherwise on the calling thread.
        """
        with Timer("policy_server_round_trip"):
            response = self._call_server_with_retry(request)
        return self._chunks_to_tensor([self._chunk_from_response(response)])

    def _fetch_action_chunks_batched(self, requests: list[dict[str, Any]]) -> torch.Tensor:
        """Send several envs' requests as one batched request and return their ``(len(requests), horizon, dim)`` chunks.

        Runs on a worker thread when a latency is modeled, otherwise on the calling thread.
        """
        with Timer("policy_server_round_trip"):
            response = self._call_server_with_retry({BATCH_REQUEST_KEY: requests})
        responses = response[BATCH_REQUEST_KEY]
        assert len(responses) == len(
            requests
        ), f"server returned {len(responses)} chunks for a batched request of {len(requests)} environments"
        return self._chunks_to_tensor([self._chunk_from_response(entry) for entry in responses])

    def _chunk_from_response(self, response: dict[str, Any]) -> np.ndarray:
        """Return the ``(open_loop_horizon, action_dim)`` float32 chunk of one env's server response."""
        chunk = np.asarray(response[self.server_response_actions_key])
        assert (
            chunk.ndim == 2 and chunk.shape[1] == self._adapter.action_dim
        ), f"Expected actions of shape (H, {self._adapter.action_dim}); got {chunk.shape}"
        assert (
            chunk.shape[0] >= self._open_loop_horizon
        ), f"Server returned horizon {chunk.shape[0]} < configured open_loop_horizon {self._open_loop_horizon}"
        return chunk[: self._open_loop_horizon].astype(np.float32, copy=True)

    def _chunks_to_tensor(self, chunks: list[np.ndarray]) -> torch.Tensor:
        """Stack per-env chunks into a ``(len(chunks), horizon, dim)`` tensor on the policy device."""
        return torch.from_numpy(np.stack(chunks)).to(device=self.device)

    def _call_server_with_retry(self, server_request: dict[str, Any]) -> dict[str, Any]:
        """Send the request on this thread's connection, reconnecting up to ``MAX_RECONNECT_ATTEMPTS`` times.

        On any reconnect every env's chunk is flushed on the next ``get_action`` so the caller
        re-queries with a fresh observation rather than replaying a potentially-stale chunk
        against the new server state.
        """
        with self._server_calls_lock:
            self._server_calls += 1
        for attempt_index in range(MAX_RECONNECT_ATTEMPTS):
            try:
                return self._client_for_current_thread().infer(server_request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as exc:
                is_last_attempt = (attempt_index + 1) >= MAX_RECONNECT_ATTEMPTS
                if is_last_attempt:
                    raise
                print(
                    f"[{type(self).__name__}] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                self._reconnect_current_thread()
                self._flush_requested = True
        raise RuntimeError("unreachable")

    def _client_for_current_thread(self) -> WebsocketClientPolicy:
        """Return the calling thread's connection, opening it on first use."""
        client = getattr(self._thread_clients, "client", None)
        if client is None:
            client = self._connect()
            self._thread_clients.client = client
        return client

    def _reconnect_current_thread(self) -> None:
        """Replace the calling thread's connection with a new one."""
        old_client = getattr(self._thread_clients, "client", None)
        if old_client is not None:
            _close_websocket_best_effort(old_client)
            with self._clients_lock:
                if old_client in self._all_clients:
                    self._all_clients.remove(old_client)
        self._thread_clients.client = self._connect()
        if threading.current_thread() is threading.main_thread():
            self._websocket_client = self._thread_clients.client

    def _connect(self) -> WebsocketClientPolicy:
        # Use our WebsocketClientPolicy override instead of openpi's, so we can set the
        # keepalive ping interval/timeout (upstream's client does not expose them).
        client = WebsocketClientPolicy(
            host=self._remote_host,
            port=self._remote_port,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        with self._clients_lock:
            self._all_clients.append(client)
        return client

    def _print_latency_summary(self) -> None:
        """Print how the chunk requests went out: server calls per chunk, and steps spent holding under latency."""
        scheduler = self._scheduler
        assert scheduler is not None
        name = type(self).__name__
        with self._server_calls_lock:
            server_calls = self._server_calls
        if self.server_batches_requests() and scheduler.requests_issued:
            print(
                f"[{name}] {scheduler.requests_issued} chunk requests sent in {server_calls} batched server calls"
                f" ({scheduler.requests_issued / max(server_calls, 1):.1f} environments per call)"
            )
        if scheduler.latency_steps == 0:
            return
        print(
            f"[{name}] Inference latency {scheduler.latency_steps} steps:"
            f" {scheduler.requests_issued} chunk requests,"
            f" {int(scheduler.hold_steps.sum().item())} env-steps holding"
            f" ({int(scheduler.stall_steps.sum().item())} of them stalled between chunks)"
        )


def _close_websocket_best_effort(client: websocket_client_policy.WebsocketClientPolicy | None) -> None:
    """Best-effort close of the websocket inside ``client``.

    Swallows the typical "peer already gone" errors so the teardown and reconnect paths can
    call this without crashing.
    """
    if client is None:
        return
    try:
        ws = getattr(client, "_ws", None)
        if ws is not None:
            ws.close()
    except (websockets.exceptions.ConnectionClosed, OSError):
        pass
