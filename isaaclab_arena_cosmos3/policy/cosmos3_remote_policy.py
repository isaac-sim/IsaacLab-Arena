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

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_cosmos3.policy.cosmos3_remote_config import MAX_RECONNECT_ATTEMPTS, Cosmos3RemotePolicyArgs


class Cosmos3EmbodimentAdapter(ABC):
    """Translates between arena's gym observation dict and the cosmos3 wire
    format for a specific physical embodiment (DROID).

    Subclasses declare the embodiment-specific action layout and observation keys.
    """

    action_dim: int
    open_loop_horizon: int

    @abstractmethod
    def extract(self, observation: dict[str, Any], env_id: int) -> Any:
        """Pull a single env's tensors out of the arena gym observation dict.

        ``env_id`` selects which slice of each per-env tensor to read. Cosmos3's
        wire format takes one observation per request, so the policy loops over
        envs and calls this once per env to assemble the per-env requests.

        Concrete return type is adapter-defined (typically a frozen dataclass);
        the policy treats it as an opaque value to round-trip through
        :meth:`pack_request`.
        """

    @abstractmethod
    def pack_request(self, extracted: Any, language_instruction: str) -> dict[str, Any]:
        """Build the wire-format request payload the cosmos3 server expects."""


@register_policy
class Cosmos3RemotePolicy(PolicyBase):
    """Cosmos3 remote closed-loop policy, parameterized by an embodiment adapter.

    Cosmos3 is NVIDIA's World-Action Model (WAM) post-trained on the DROID
    dataset.  It jointly processes language, images, and action sequences to
    predict chunks of future robot actions.

    Action handling is straight chunk replay: the policy fetches one
    ``(open_loop_horizon, action_dim)`` chunk from the server and yields
    rows in order.
    """

    name = "cosmos3_remote"
    config_class = Cosmos3RemotePolicyArgs

    def __init__(self, config: Cosmos3RemotePolicyArgs, cosmos3_embodiment_adapter: Cosmos3EmbodimentAdapter) -> None:
        super().__init__(config)
        self._cosmos3_embodiment_adapter = cosmos3_embodiment_adapter
        self._open_loop_horizon = cosmos3_embodiment_adapter.open_loop_horizon

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port
        self.device = config.policy_device

        print(f"[Cosmos3RemotePolicy] Connecting to cosmos3 server at {self._remote_host}:{self._remote_port} ...")
        self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._remote_host, port=self._remote_port
        )
        print("[Cosmos3RemotePolicy] Connected.")

        # Per-env action cache. Lazy-allocated on the first get_action call when
        # num_envs is known. Cosmos3's wire format is one obs per request, so we
        # keep one chunk + one step counter per env and loop over them.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self.task_description: str | None = None

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Cosmos3 Remote Policy",
            "Arguments for the cosmos3 remote client.",
        )
        group.add_argument(
            "--cosmos3_embodiment_adapter",
            type=str,
            default="droid",
            choices=["droid"],
            help="Cosmos3-side embodiment adapter for obs / action wire format (default: droid).",
        )
        group.add_argument("--remote_host", type=str, default="localhost", help="Cosmos3 server host.")
        group.add_argument("--remote_port", type=int, default=8000, help="Cosmos3 server port.")
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Torch device for action tensors (e.g. 'cuda', 'cuda:0', 'cpu').",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Cosmos3RemotePolicy:
        cosmos3_embodiment_adapter = _resolve_cosmos3_embodiment_adapter(args.cosmos3_embodiment_adapter)
        return Cosmos3RemotePolicy(
            Cosmos3RemotePolicyArgs(
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                policy_device=getattr(args, "policy_device", "cuda"),
            ),
            cosmos3_embodiment_adapter=cosmos3_embodiment_adapter,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Cosmos3RemotePolicy:
        """JSON-jobs-config path used by ``eval_runner``.

        Overrides ``PolicyBase.from_dict`` because our ``__init__`` takes an
        adapter alongside the config dataclass.
        # TODO(cvolk, 2026-05-18): add a RemotePolicy base class to unify this and other remote policies.
        """
        config_dict = dict(config_dict)
        adapter_key = config_dict.pop("cosmos3_embodiment_adapter", "droid")
        cosmos3_embodiment_adapter = _resolve_cosmos3_embodiment_adapter(adapter_key)
        return cls(Cosmos3RemotePolicyArgs(**config_dict), cosmos3_embodiment_adapter=cosmos3_embodiment_adapter)

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "Cosmos3RemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        num_envs = env.unwrapped.num_envs
        self._maybe_init_per_env_state(num_envs)

        # Cosmos3 server takes one obs per request, so we iterate over envs
        # and send one inference per env that needs a fresh chunk.
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
        """Release the local websocket connection to the cosmos3 server.

        Does NOT stop the cosmos3 server process that runs in a separate
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
            f"Cosmos3RemotePolicy num_envs changed from {len(self._cached_action_chunks)}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )

    def _fetch_action_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        extracted = self._cosmos3_embodiment_adapter.extract(observation, env_id)
        request = self._cosmos3_embodiment_adapter.pack_request(extracted, self.task_description)
        response = self._call_server_with_retry(request)

        chunk = np.asarray(response["action"])
        assert (
            chunk.ndim == 2 and chunk.shape[1] == self._cosmos3_embodiment_adapter.action_dim
        ), f"Expected action of shape (H, {self._cosmos3_embodiment_adapter.action_dim}); got {chunk.shape}"
        assert (
            chunk.shape[0] >= self._open_loop_horizon
        ), f"Server returned horizon {chunk.shape[0]} < configured open_loop_horizon {self._open_loop_horizon}"

        chunk = chunk[: self._open_loop_horizon].astype(np.float32, copy=True)
        chunk = self._postprocess_chunk(chunk)
        return chunk

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Apply post-processing to the action chunk.

        Binarizes the last dimension (gripper): values > 0.5 → 1.0, others → 0.0.
        This matches the cosmos3 RoboLab client's ``_postprocess_chunk`` behavior.
        """
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _call_server_with_retry(self, server_request: dict[str, Any]) -> dict[str, Any]:
        """Send the request, reconnecting up to ``MAX_RECONNECT_ATTEMPTS`` times.

        On any reconnect the cached chunks are flushed so the caller's next
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
                    f"[Cosmos3RemotePolicy] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                _close_websocket_best_effort(self._websocket_client)
                self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
                    host=self._remote_host, port=self._remote_port
                )
                # Flush every env's cache: the reconnected server may have lost
                # state, so we force a fresh observation on the next get_action
                # for each env rather than replay cached actions.
                if self._cached_action_chunks is not None:
                    for i in range(len(self._cached_action_chunks)):
                        self._cached_action_chunks[i] = None
                        self._next_chunk_steps[i] = 0
        raise RuntimeError("unreachable")

    @property
    def is_remote(self) -> bool:
        return True


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


def _resolve_cosmos3_embodiment_adapter(key: str) -> Cosmos3EmbodimentAdapter:
    """Instantiate the adapter registered under ``key``.

    Imports are deferred to call time so adapter modules can ``from
    cosmos3_remote_policy import Cosmos3EmbodimentAdapter`` at module top
    without creating a circular import.
    """
    if key == "droid":
        from isaaclab_arena_cosmos3.policy.droid_adapter import Cosmos3DroidAdapter

        return Cosmos3DroidAdapter()
    raise ValueError(f"Unknown cosmos3_embodiment_adapter {key!r}; expected 'droid'")
