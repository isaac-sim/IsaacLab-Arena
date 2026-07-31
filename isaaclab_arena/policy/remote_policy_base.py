# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any

import websockets.exceptions
from openpi_client import websocket_client_policy

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.policy.websocket_client import WebsocketClientPolicy

MAX_RECONNECT_ATTEMPTS = 3


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
    refetching.

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
        self.device = config.policy_device

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port
        self._ping_interval = config.ping_interval
        self._ping_timeout = config.ping_timeout

        name = type(self).__name__
        print(f"[{name}] Connecting to server at {self._remote_host}:{self._remote_port} ...")
        # Use our WebsocketClientPolicy override instead of openpi's, so we can set the
        # keepalive ping interval/timeout (upstream's client does not expose them).
        self._websocket_client = WebsocketClientPolicy(
            host=self._remote_host,
            port=self._remote_port,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        print(f"[{name}] Connected.")

        # Per-env action cache. Lazy-allocated on the first get_action call when num_envs
        # is known. The wire format is one obs per request, so we keep one chunk + one step
        # counter per env and loop over them.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self.task_description: str | None = None

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            f"{type(self).__name__} requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        num_envs = env.unwrapped.num_envs
        self._maybe_init_per_env_state(num_envs)

        actions = []
        for env_id in range(num_envs):
            chunk_exhausted = (
                self._cached_action_chunks[env_id] is None or self._next_chunk_steps[env_id] >= self._open_loop_horizon
            )
            if chunk_exhausted:
                self._cached_action_chunks[env_id] = self._fetch_action_chunk(observation, env_id)
                self._next_chunk_steps[env_id] = 0
            actions.append(self._cached_action_chunks[env_id][self._next_chunk_steps[env_id]])
            self._next_chunk_steps[env_id] += 1

        batch = np.stack(actions)  # (num_envs, action_dim)
        return torch.from_numpy(batch).to(dtype=torch.float32, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._cached_action_chunks is None:
            return  # not yet initialized; nothing to clear
        ids = range(len(self._cached_action_chunks)) if env_ids is None else env_ids.reshape(-1).tolist()
        for env_id in ids:
            self._cached_action_chunks[env_id] = None
            self._next_chunk_steps[env_id] = 0

    def close(self) -> None:
        """Release the local websocket connection to the server.

        Does NOT stop the server process that runs in a separate container (or machine)
        and outlives this client.
        """
        _close_websocket_best_effort(self._websocket_client)
        self._websocket_client = None

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs, (
            f"{type(self).__name__} num_envs changed from {len(self._cached_action_chunks)}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )

    def _fetch_action_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        extracted = self._adapter.extract(observation, env_id)
        request = self._adapter.pack_request(extracted, self.task_description)
        response = self._call_server_with_retry(request)

        chunk = np.asarray(response[self.server_response_actions_key])
        assert (
            chunk.ndim == 2 and chunk.shape[1] == self._adapter.action_dim
        ), f"Expected actions of shape (H, {self._adapter.action_dim}); got {chunk.shape}"
        assert (
            chunk.shape[0] >= self._open_loop_horizon
        ), f"Server returned horizon {chunk.shape[0]} < configured open_loop_horizon {self._open_loop_horizon}"
        return chunk[: self._open_loop_horizon].astype(np.float32, copy=True)

    def _call_server_with_retry(self, server_request: dict[str, Any]) -> dict[str, Any]:
        """Send the request, reconnecting up to ``MAX_RECONNECT_ATTEMPTS`` times.

        On any reconnect the cached chunk is flushed so the caller's next ``get_action``
        re-queries with a fresh observation rather than replaying a potentially-stale chunk
        against the new server state.
        """
        for attempt_index in range(MAX_RECONNECT_ATTEMPTS):
            try:
                return self._websocket_client.infer(server_request)
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
                _close_websocket_best_effort(self._websocket_client)
                self._websocket_client = WebsocketClientPolicy(
                    host=self._remote_host,
                    port=self._remote_port,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                # Flush every env's cache: the reconnected server may have lost state, so we
                # force a fresh observation on the next get_action for each env rather than
                # replay cached actions.
                if self._cached_action_chunks is not None:
                    for i in range(len(self._cached_action_chunks)):
                        self._cached_action_chunks[i] = None
                        self._next_chunk_steps[i] = 0
        raise RuntimeError("unreachable")


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
