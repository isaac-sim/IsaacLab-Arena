# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any

import websockets.exceptions
from openpi_client import websocket_client_policy

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_openpi.policy.pi0_remote_config import DEFAULT_VARIANT, MAX_RECONNECT_ATTEMPTS, Pi0RemotePolicyArgs
from isaaclab_arena_openpi.policy.websocket_client import WebsocketClientPolicy


class Pi0EmbodimentAdapter(ABC):
    """Translates between arena's gym observation dict and the openpi wire
    format for a specific physical embodiment (DROID, ALOHA, ...).

    Subclasses declare the embodiment-specific action layout and observation keys.
    """

    action_dim: int
    open_loop_horizon_by_variant: dict[str, int]

    @abstractmethod
    def extract(self, observation: dict[str, Any], env_id: int) -> Any:
        """Pull a single env's tensors out of the arena gym observation dict.

        ``env_id`` selects which slice of each per-env tensor to read. openpi's
        wire format takes one observation per request, so the policy loops over
        envs and calls this once per env to assemble the per-env requests.

        Concrete return type is adapter-defined (typically a frozen dataclass);
        the policy treats it as an opaque value to round-trip through
        :meth:`pack_request`.
        """

    @abstractmethod
    def pack_request(self, extracted: Any, language_instruction: str) -> dict[str, Any]:
        """Build the wire-format request payload openpi server expects."""


class Pi0RemotePolicy(PolicyBase):
    """openpi remote closed-loop policy, parameterized by an embodiment adapter.

    Action handling today is straight chunk replay: the policy fetches one
    ``(open_loop_horizon, action_dim)`` chunk from the server and yields
    rows in order. A future action scheduler (interpolation, smoothing,
    chunk-overlap blending) belongs as a pluggable component.
    """

    # TODO(cvolk, 2026-05-18): add an action_scheduler_cls so action_chunk
    # -> action is configurable (today it is a row-by-row replay).

    # TODO(cvolk, 2026-05-18): Add a RemotePolicy base class.

    name = "pi0_remote"
    config_class = Pi0RemotePolicyArgs

    def __init__(self, config: Pi0RemotePolicyArgs, openpi_embodiment_adapter: Pi0EmbodimentAdapter) -> None:
        super().__init__(config)
        assert config.policy_variant in openpi_embodiment_adapter.open_loop_horizon_by_variant, (
            f"Unknown policy_variant {config.policy_variant!r} for adapter"
            f" {type(openpi_embodiment_adapter).__name__};"
            f" known: {sorted(openpi_embodiment_adapter.open_loop_horizon_by_variant)}"
        )
        self._openpi_embodiment_adapter = openpi_embodiment_adapter
        self._open_loop_horizon = openpi_embodiment_adapter.open_loop_horizon_by_variant[config.policy_variant]
        self.device = config.policy_device

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port
        self._ping_interval = config.ping_interval
        self._ping_timeout = config.ping_timeout

        print(f"[Pi0RemotePolicy] Connecting to openpi server at {self._remote_host}:{self._remote_port} ...")
        # Use our WebsocketClientPolicy override instead of openpi's, so we can set the keepalive
        # ping interval/timeout (upstream's client does not expose them).
        self._websocket_client = WebsocketClientPolicy(
            host=self._remote_host,
            port=self._remote_port,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        print("[Pi0RemotePolicy] Connected.")

        # Per-env action cache. Lazy-allocated on the first get_action call when
        # num_envs is known. openpi's wire format is one obs per request, so we
        # keep one chunk + one step counter per env and loop over them.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self.task_description: str | None = None

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Pi0 Remote Policy",
            "Arguments for the openpi (pi0 / pi05) remote client.",
        )
        group.add_argument(
            "--openpi_embodiment_adapter",
            type=str,
            default="droid",
            choices=["droid"],
            help="Openpi-side embodiment adapter for obs / action wire format (default: droid).",
        )
        group.add_argument(
            "--policy_variant",
            type=str,
            default=DEFAULT_VARIANT,
            help=(
                f"openpi checkpoint variant (default: {DEFAULT_VARIANT})."
                " Valid values depend on the chosen --openpi_embodiment_adapter."
            ),
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Torch device for action tensors (default: cuda).",
        )
        group.add_argument("--remote_host", type=str, default="localhost", help="openpi server host.")
        group.add_argument("--remote_port", type=int, default=8000, help="openpi server port.")
        group.add_argument(
            "--ping_interval",
            type=float,
            default=20.0,
            help="Seconds between websocket keepalive pings.",
        )
        group.add_argument(
            "--ping_timeout",
            type=float,
            default=20.0,
            help="Seconds to wait for a keepalive pong before dropping the connection.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Pi0RemotePolicy:
        openpi_embodiment_adapter = _resolve_openpi_embodiment_adapter(args.openpi_embodiment_adapter)
        return Pi0RemotePolicy(
            Pi0RemotePolicyArgs(
                policy_variant=args.policy_variant,
                policy_device=args.policy_device,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                ping_interval=args.ping_interval,
                ping_timeout=args.ping_timeout,
            ),
            openpi_embodiment_adapter=openpi_embodiment_adapter,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Pi0RemotePolicy:
        """JSON-jobs-config path used by ``eval_runner``.

        Overrides ``PolicyBase.from_dict`` because our ``__init__`` takes an
        adapter alongside the config dataclass.
        # TODO(cvolk, 2026-05-18): add a RemotePolicy base class to unify this and other remote policies.
        """
        config_dict = dict(config_dict)
        adapter_key = config_dict.pop("openpi_embodiment_adapter", "droid")
        openpi_embodiment_adapter = _resolve_openpi_embodiment_adapter(adapter_key)
        return cls(Pi0RemotePolicyArgs(**config_dict), openpi_embodiment_adapter=openpi_embodiment_adapter)

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "Pi0RemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        num_envs = env.unwrapped.num_envs
        self._maybe_init_per_env_state(num_envs)

        # TODO(cvolk): openpi server takes one obs per request, so we iterate
        # over envs and send one inference per env that needs a fresh chunk.
        # This is N-times slower than a single batched call but is correct
        # for parallel envs; switch to a single batched call when openpi
        # grows batched-inference support upstream.
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
        """Release the local websocket connection to the openpi server.
        Does NOT stop the openpi server process that runs in a separate
        container (or machine) and outlives this client.
        """
        _close_websocket_best_effort(self._websocket_client)
        self._websocket_client = None

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs, (
            f"Pi0RemotePolicy num_envs changed from {len(self._cached_action_chunks)}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )

    def _fetch_action_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        extracted = self._openpi_embodiment_adapter.extract(observation, env_id)
        request = self._openpi_embodiment_adapter.pack_request(extracted, self.task_description)
        response = self._call_server_with_retry(request)

        chunk = np.asarray(response["actions"])
        assert (
            chunk.ndim == 2 and chunk.shape[1] == self._openpi_embodiment_adapter.action_dim
        ), f"Expected actions of shape (H, {self._openpi_embodiment_adapter.action_dim}); got {chunk.shape}"
        assert (
            chunk.shape[0] >= self._open_loop_horizon
        ), f"Server returned horizon {chunk.shape[0]} < configured open_loop_horizon {self._open_loop_horizon}"
        return chunk[: self._open_loop_horizon].astype(np.float32, copy=True)

    def _call_server_with_retry(self, server_request: dict[str, Any]) -> dict[str, Any]:
        """Send the request, reconnecting up to ``MAX_RECONNECT_ATTEMPTS`` times.

        On any reconnect the cached chunk is flushed so the caller's next
        ``get_action`` re-queries with a fresh observation rather than
        replaying a potentially-stale chunk against the new server state.
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
                    f"[Pi0RemotePolicy] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                _close_websocket_best_effort(self._websocket_client)
                self._websocket_client = WebsocketClientPolicy(
                    host=self._remote_host,
                    port=self._remote_port,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                # Flush every env's cache: the reconnected server may have lost
                # state, so we force a fresh observation on the next get_action
                # for each env rather than replay cached actions.
                if self._cached_action_chunks is not None:
                    for i in range(len(self._cached_action_chunks)):
                        self._cached_action_chunks[i] = None
                        self._next_chunk_steps[i] = 0
        raise RuntimeError("unreachable")


def _close_websocket_best_effort(client: websocket_client_policy.WebsocketClientPolicy | None) -> None:
    """Best-effort close of the websocket inside ``client``.

    Swallows the typical "peer already gone" errors so the teardown and
    reconnect paths can call this without crashing.
    """
    if client is None:
        return
    try:
        ws = getattr(client, "_ws", None)
        if ws is not None:
            ws.close()
    except (websockets.exceptions.ConnectionClosed, OSError):
        pass


def _resolve_openpi_embodiment_adapter(key: str) -> Pi0EmbodimentAdapter:
    """Instantiate the adapter registered under ``key``.

    Imports are deferred to call time so adapter modules can ``from
    pi0_remote_policy import Pi0EmbodimentAdapter`` at module top without
    creating a circular import.
    """
    if key == "droid":
        from isaaclab_arena_openpi.policy.droid_adapter import Pi0DroidAdapter

        return Pi0DroidAdapter()
    raise ValueError(f"Unknown openpi_embodiment_adapter {key!r}; expected 'droid'")
