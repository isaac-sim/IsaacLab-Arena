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

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_cosmos.policy.cosmos_remote_config import MAX_RECONNECT_ATTEMPTS, CosmosRemotePolicyCfg
from isaaclab_arena_cosmos.policy.websocket_client import WebsocketClientPolicy


class CosmosEmbodimentAdapter(ABC):
    """Translates between Arena's gym observation dict and the Cosmos RoboLab wire
    format for a specific physical embodiment (DROID, ...).

    Subclasses declare the embodiment-specific action layout and observation keys.
    """

    action_dim: int

    @abstractmethod
    def extract(self, observation: dict[str, Any], env_id: int) -> Any:
        """Pull a single env's tensors out of the arena gym observation dict.

        ``env_id`` selects which slice of each per-env tensor to read. The Cosmos
        server takes one observation per request, so the policy loops over envs and
        calls this once per env to assemble the per-env requests.

        Concrete return type is adapter-defined (typically a frozen dataclass); the
        policy treats it as an opaque value to round-trip through pack_request.
        """

    @abstractmethod
    def pack_request(self, extracted: Any, language_instruction: str) -> dict[str, Any]:
        """Build the wire-format request payload the Cosmos RoboLab server expects."""


@register_policy
class CosmosRemotePolicy(PolicyBase[CosmosRemotePolicyCfg]):
    """Cosmos RoboLab remote closed-loop policy, parameterized by an embodiment adapter.

    The Cosmos ``action_policy_server_robolab`` server serves over openpi's
    ``WebsocketPolicyServer`` (msgpack+NumPy protocol), so the same websocket client
    speaks to it. Action handling is straight chunk replay: the policy fetches one
    ``(action_chunk_size, action_dim)`` chunk, keeps the first ``open_loop_horizon``
    rows, and yields them in order before refetching.
    """

    name = "cosmos_remote"

    def __init__(self, config: CosmosRemotePolicyCfg) -> None:
        super().__init__(config)
        self._cosmos_embodiment_adapter = _resolve_cosmos_embodiment_adapter(config.cosmos_embodiment_adapter)
        self._open_loop_horizon = config.open_loop_horizon
        self.device = config.policy_device

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port
        self._ping_interval = config.ping_interval
        self._ping_timeout = config.ping_timeout

        print(f"[CosmosRemotePolicy] Connecting to Cosmos server at {self._remote_host}:{self._remote_port} ...")
        # Use our WebsocketClientPolicy override instead of openpi's, so we can set the
        # keepalive ping interval/timeout (upstream's client does not expose them).
        self._websocket_client = WebsocketClientPolicy(
            host=self._remote_host,
            port=self._remote_port,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        print("[CosmosRemotePolicy] Connected.")

        # Per-env action cache. Lazy-allocated on the first get_action call when
        # num_envs is known. The server's wire format is one obs per request, so we
        # keep one chunk + one step counter per env and loop over them.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self.task_description: str | None = None

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "CosmosRemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        num_envs = env.unwrapped.num_envs
        self._maybe_init_per_env_state(num_envs)

        # The server takes one obs per request, so we iterate over envs and send one
        # inference per env that needs a fresh chunk.
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
        """Release the local websocket connection to the Cosmos server.

        Does NOT stop the Cosmos server process that runs in a separate container
        (or machine) and outlives this client.
        """
        _close_websocket_best_effort(self._websocket_client)
        self._websocket_client = None

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs, (
            f"CosmosRemotePolicy num_envs changed from {len(self._cached_action_chunks)}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )

    def _fetch_action_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        extracted = self._cosmos_embodiment_adapter.extract(observation, env_id)
        request = self._cosmos_embodiment_adapter.pack_request(extracted, self.task_description)
        response = self._call_server_with_retry(request)

        # The RoboLab server returns the action chunk under the singular "action" key.
        chunk = np.asarray(response["action"])
        assert (
            chunk.ndim == 2 and chunk.shape[1] == self._cosmos_embodiment_adapter.action_dim
        ), f"Expected actions of shape (H, {self._cosmos_embodiment_adapter.action_dim}); got {chunk.shape}"
        assert (
            chunk.shape[0] >= self._open_loop_horizon
        ), f"Server returned horizon {chunk.shape[0]} < configured open_loop_horizon {self._open_loop_horizon}"
        return chunk[: self._open_loop_horizon].astype(np.float32, copy=True)

    def _call_server_with_retry(self, server_request: dict[str, Any]) -> dict[str, Any]:
        """Send the request, reconnecting up to ``MAX_RECONNECT_ATTEMPTS`` times.

        On any reconnect the cached chunk is flushed so the caller's next
        ``get_action`` re-queries with a fresh observation rather than replaying a
        potentially-stale chunk against the new server state.
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
                    f"[CosmosRemotePolicy] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                _close_websocket_best_effort(self._websocket_client)
                self._websocket_client = WebsocketClientPolicy(
                    host=self._remote_host,
                    port=self._remote_port,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                # Flush every env's cache: the reconnected server may have lost state,
                # so we force a fresh observation on the next get_action for each env
                # rather than replay cached actions.
                if self._cached_action_chunks is not None:
                    for i in range(len(self._cached_action_chunks)):
                        self._cached_action_chunks[i] = None
                        self._next_chunk_steps[i] = 0
        raise RuntimeError("unreachable")


def _close_websocket_best_effort(client: websocket_client_policy.WebsocketClientPolicy | None) -> None:
    """Best-effort close of the websocket inside ``client``.

    Swallows the typical "peer already gone" errors so the teardown and reconnect
    paths can call this without crashing.
    """
    if client is None:
        return
    try:
        ws = getattr(client, "_ws", None)
        if ws is not None:
            ws.close()
    except (websockets.exceptions.ConnectionClosed, OSError):
        pass


def _resolve_cosmos_embodiment_adapter(key: str) -> CosmosEmbodimentAdapter:
    """Instantiate the adapter registered under ``key``.

    Imports are deferred to call time so adapter modules can import
    CosmosEmbodimentAdapter at their module top without a circular import.
    """
    if key == "droid":
        from isaaclab_arena_cosmos.policy.droid_adapter import CosmosDroidAdapter

        return CosmosDroidAdapter()
    raise ValueError(f"Unknown cosmos_embodiment_adapter {key!r}; expected 'droid'")
