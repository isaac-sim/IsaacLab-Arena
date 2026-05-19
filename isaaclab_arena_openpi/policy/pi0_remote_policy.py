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


class Pi0EmbodimentAdapter(ABC):
    """Translates between arena's gym observation dict and the openpi wire
    format for a specific physical embodiment (DROID, ALOHA, ...).

    Subclasses declare the embodiment-specific action layout and observation keys.
    """

    action_dim: int
    open_loop_horizon_by_variant: dict[str, int]

    @abstractmethod
    def extract(self, observation: dict[str, Any]) -> Any:
        """Pull env tensors out of the arena gym observation dict.

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

        print(f"[Pi0RemotePolicy] Connecting to openpi server at {self._remote_host}:{self._remote_port} ...")
        self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._remote_host, port=self._remote_port
        )
        print("[Pi0RemotePolicy] Connected.")

        self._cached_action_chunk: np.ndarray | None = None
        self._next_chunk_step: int = 0
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
        # TODO(cvolk, 2026-05-18): extend to parallel envs once the openpi
        # server supports batched observations; today it accepts one obs
        # per request.
        assert (
            env.unwrapped.num_envs == 1
        ), f"Pi0RemotePolicy only supports num_envs=1 (openpi server limitation), got num_envs={env.unwrapped.num_envs}"
        assert self.task_description, (
            "Pi0RemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        chunk_exhausted = self._cached_action_chunk is None or self._next_chunk_step >= self._open_loop_horizon
        if chunk_exhausted:
            self._cached_action_chunk = self._fetch_action_chunk(observation)
            self._next_chunk_step = 0

        next_action_np = self._cached_action_chunk[self._next_chunk_step]
        self._next_chunk_step += 1
        return torch.from_numpy(next_action_np).to(dtype=torch.float32, device=self.device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # TODO(cvolk, 2026-05-18): honor env_ids when parallel envs are
        # supported; today the cache is single-env and we always reset the
        # whole thing.
        self._cached_action_chunk = None
        self._next_chunk_step = 0

    def close(self) -> None:
        """Release the local websocket connection to the openpi server.
        Does NOT stop the openpi server process that runs in a separate
        container (or machine) and outlives this client.
        """
        _close_websocket_best_effort(self._websocket_client)
        self._websocket_client = None

    def _fetch_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        extracted = self._openpi_embodiment_adapter.extract(observation)
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
                self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
                    host=self._remote_host, port=self._remote_port
                )
                self._cached_action_chunk = None
                self._next_chunk_step = 0
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
