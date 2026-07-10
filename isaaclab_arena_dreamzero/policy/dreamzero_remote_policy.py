# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import dataclasses
import gymnasium as gym
import inspect
import numpy as np
import time
import torch
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import msgpack
import websockets.exceptions
import websockets.sync.client as ws_sync

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_dreamzero.policy.dreamzero_remote_config import MAX_RECONNECT_ATTEMPTS, DreamZeroRemotePolicyConfig

RECONNECT_WAIT_S = 120
"""Per-attempt reconnect budget (seconds) after a connection drop mid-rollout, long enough
for a supervised tunnel or restarted server to come back before the attempt is charged."""


def _msgpack_encode(obj):
    """Encode numpy arrays to the DreamZero server wire format."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _msgpack_decode(obj):
    """Decode DreamZero server wire format back to numpy arrays."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _pack(data: dict) -> bytes:
    return msgpack.packb(data, default=_msgpack_encode)


def _unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, object_hook=_msgpack_decode, strict_map_key=False)


class DreamZeroEmbodimentAdapter(ABC):
    """Translates between Arena's gym observation dict and DreamZero's wire
    format for a specific physical embodiment (DROID, ...).

    Subclasses declare the embodiment-specific action layout and observation keys.
    Adding support for a new embodiment means writing a new adapter and adding one
    entry to ``_EMBODIMENT_ADAPTER_LOADERS`` — no branching inside
    DreamZeroRemotePolicy. This keeps the client embodiment agnostic and makes each
    embodiment's joint order / camera mapping explicit in one place.
    """

    action_dim: int

    @classmethod
    def config_field_names(cls) -> tuple[str, ...]:
        """Eval-jobs config keys that configure this adapter (routed by DreamZeroRemotePolicy.from_dict).

        Derived from ``__init__`` so it cannot drift from the constructor signature and
        stays consistent with the default ``from_config_dict`` (``cls(**config)``). Override
        only if the adapter takes constructor args that are not eval-jobs config fields.
        """
        params = inspect.signature(cls.__init__).parameters.values()
        return tuple(p.name for p in params if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))

    @abstractmethod
    def extract(self, observation: dict[str, Any], env_id: int) -> Any:
        """Pull a single env's tensors out of the arena gym observation dict.

        ``env_id`` selects which slice of each per-env tensor to read. DreamZero's
        wire format takes one observation per request, so the policy loops over
        envs and calls this once per env to assemble the per-env requests.

        Concrete return type is adapter-defined (typically a frozen dataclass);
        the policy treats it as an opaque value to round-trip through the
        pack_request method.
        """

    @abstractmethod
    def pack_request(self, extracted: Any) -> dict[str, Any]:
        """Build the observation/* portion of the DreamZero wire-format request."""

    @abstractmethod
    def parse_actions(self, response: dict[str, Any] | np.ndarray) -> np.ndarray:
        """Decode a server response into a float32 (num_steps, action_dim) action chunk.

        Owns the embodiment-specific parts of response decoding — envelope
        unwrapping, dimensional padding, and joint ordering / conversion to Arena
        actions. The policy handles only the replay horizon on top of this.
        """

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> DreamZeroEmbodimentAdapter:
        """Build the adapter from its slice of an eval-jobs config dict. Default: kwargs."""
        return cls(**config)


def _load_droid_adapter() -> type[DreamZeroEmbodimentAdapter]:
    from isaaclab_arena_dreamzero.policy.droid_adapter import DroidAdapter

    return DroidAdapter


_EMBODIMENT_ADAPTER_LOADERS: dict[str, Callable[[], type[DreamZeroEmbodimentAdapter]]] = {
    "droid": _load_droid_adapter,
}
"""Registry mapping embodiment key -> a loader for its adapter class. Loaders defer
the import to call time so adapter modules can import DreamZeroEmbodimentAdapter at
their module top without a circular import. Register a new embodiment by adding an entry."""


def _resolve_adapter_class(key: str) -> type[DreamZeroEmbodimentAdapter]:
    """Return the adapter class registered under ``key``."""
    loader = _EMBODIMENT_ADAPTER_LOADERS.get(key)
    assert (
        loader is not None
    ), f"Unknown dreamzero_embodiment_adapter {key!r}; expected one of {sorted(_EMBODIMENT_ADAPTER_LOADERS)}"
    return loader()


@register_policy
class DreamZeroRemotePolicy(PolicyBase[DreamZeroRemotePolicyConfig]):
    """Remote closed-loop policy that communicates with a DreamZero inference server.

    Observations are formatted into DreamZero's flat wire format and sent over a
    synchronous WebSocket connection using MessagePack serialization. The server
    returns action chunks that are replayed step-by-step, querying for new chunks
    only when the current one is exhausted.

    The server is stateful: it maintains a temporal observation history per session
    UUID. Each parallel environment gets its own UUID so their histories do not mix.
    """

    name = "dreamzero_remote"
    config_class = DreamZeroRemotePolicyConfig

    def __init__(
        self,
        config: DreamZeroRemotePolicyConfig,
        dreamzero_embodiment_adapter: DreamZeroEmbodimentAdapter | None = None,
    ) -> None:
        super().__init__(config)
        if dreamzero_embodiment_adapter is None:
            dreamzero_embodiment_adapter = _resolve_adapter_class(config.dreamzero_embodiment_adapter)()
        self._dreamzero_embodiment_adapter = dreamzero_embodiment_adapter
        self.task_description: str | None = None
        self.device = config.policy_device

        # Per-env state; lists are None until first get_action call, when num_envs is known.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self._session_ids: list[str | None] | None = None

        self._ws: ws_sync.ClientConnection | None = None
        uri = f"ws://{config.remote_host}:{config.remote_port}"
        print(f"[DreamZeroRemotePolicy] Connecting to DreamZero server at {uri} ...")
        self._ws = self._connect_with_wait(uri, deadline_s=config.initial_connect_wait_s)
        print("[DreamZeroRemotePolicy] Connected.")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> DreamZeroRemotePolicy:
        """Build the policy from a plain config dict (JSON-config frontends and tests).

        An optional ``dreamzero_embodiment_adapter`` key selects the adapter
        (default 'droid'), mirroring the config field of the same name. Keys listed in
        the chosen adapter's ``config_field_names`` are routed to the adapter;
        everything else goes to ``DreamZeroRemotePolicyConfig``. Any key that
        matches neither raises a clear error instead of an opaque ``TypeError``
        from the dataclass constructor.
        """
        config_dict = dict(config_dict)
        adapter_key = config_dict.get("dreamzero_embodiment_adapter", "droid")
        adapter_cls = _resolve_adapter_class(adapter_key)
        adapter_field_names = set(adapter_cls.config_field_names())

        policy_field_names = {f.name for f in dataclasses.fields(DreamZeroRemotePolicyConfig)}
        unknown_keys = set(config_dict) - policy_field_names - adapter_field_names
        assert not unknown_keys, (
            f"Unknown DreamZeroRemotePolicy config keys: {sorted(unknown_keys)};"
            f" expected one of {sorted(policy_field_names | adapter_field_names)}."
        )

        adapter_kwargs = {key: config_dict.pop(key) for key in list(config_dict) if key in adapter_field_names}
        adapter = adapter_cls.from_config_dict(adapter_kwargs)
        return cls(DreamZeroRemotePolicyConfig(**config_dict), dreamzero_embodiment_adapter=adapter)

    # TODO(tstuyck, 2026-07-01): add a RemotePolicy base class
    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return the next scheduled action for every parallel environment.

        Fetches a new action chunk from the server only for environments whose
        current chunk is exhausted. Non-stale environments replay from cache.

        Args:
            env: Gymnasium-wrapped Isaac Lab environment.
            observation: Arena observation dict with 'camera_obs' and 'policy' sub-dicts.

        Returns:
            Float32 tensor of shape (num_envs, action_dim) on self.device.
        """
        assert (
            self.task_description
        ), "DreamZeroRemotePolicy requires a language instruction. Call set_task_description() before get_action()."
        if self._ws is None:
            raise RuntimeError(
                "DreamZeroRemotePolicy WebSocket is closed. Recreate the policy before calling get_action()."
            )
        num_envs = env.unwrapped.num_envs
        self._maybe_init_per_env_state(num_envs)

        actions = []
        for env_id in range(num_envs):
            chunk_exhausted = (
                self._cached_action_chunks[env_id] is None
                or self._next_chunk_steps[env_id] >= self.config.open_loop_horizon
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
            return
        ids = range(len(self._cached_action_chunks)) if env_ids is None else env_ids.reshape(-1).tolist()
        uuids_to_reset = []
        for env_id in ids:
            self._cached_action_chunks[env_id] = None
            self._next_chunk_steps[env_id] = 0
            if self._session_ids[env_id] is not None:
                uuids_to_reset.append(self._session_ids[env_id])
                self._session_ids[env_id] = None
        if uuids_to_reset:
            self._send_reset(uuids_to_reset)

    def close(self) -> None:
        """Send a best-effort server reset then close the WebSocket."""
        with contextlib.suppress(Exception):
            self._send_reset(None)
        with contextlib.suppress(Exception):
            if self._ws is not None:
                self._ws.close()
        self._ws = None

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            self._session_ids = [None] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs, (
            f"DreamZeroRemotePolicy num_envs changed from {len(self._cached_action_chunks)}"
            f" to {num_envs} mid-rollout; recreate the policy for the new num_envs."
        )

    def _get_or_create_session_id(self, env_id: int) -> str:
        """Return the existing session UUID for env_id or mint a new one."""
        if self._session_ids[env_id] is None:
            self._session_ids[env_id] = str(uuid.uuid4())
        return self._session_ids[env_id]

    def _fetch_action_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        """Fetch the next action chunk from the server for one environment.

        Args:
            observation: Full batched Arena observation.
            env_id: Index into the batch dimension.

        Returns:
            float32 ndarray of shape (open_loop_horizon, action_dim).
        """
        request = self._build_request(observation, env_id)
        response = self._call_server_with_retry(request, env_id=env_id)
        return self._parse_action_chunk(response)

    def _build_request(self, observation: dict[str, Any], env_id: int) -> dict[str, Any]:
        """Assemble the flat wire-format request dict for one environment.

        Args:
            observation: Full batched Arena observation.
            env_id: Index into the batch dimension.

        Returns:
            Dict with flat slash-delimited keys ready for MessagePack serialization.
        """
        extracted = self._dreamzero_embodiment_adapter.extract(observation, env_id)
        request = self._dreamzero_embodiment_adapter.pack_request(extracted)
        request["prompt"] = self.task_description
        request["session_id"] = self._get_or_create_session_id(env_id)
        request["endpoint"] = "infer"
        return request

    def _parse_action_chunk(self, response: dict[str, Any] | np.ndarray) -> np.ndarray:
        """Decode a server response and truncate it to the replay horizon.

        Response decoding (envelope, padding, ordering) is delegated to the
        embodiment adapter; this only enforces and applies open_loop_horizon.

        Args:
            response: Decoded MessagePack payload — either a dict with an 'actions'
                key or a bare ndarray.

        Returns:
            float32 ndarray of shape (open_loop_horizon, action_dim).
        """
        chunk = self._dreamzero_embodiment_adapter.parse_actions(response)
        assert (
            chunk.shape[0] >= self.config.open_loop_horizon
        ), f"Server returned {chunk.shape[0]} steps but open_loop_horizon={self.config.open_loop_horizon}"
        return chunk[: self.config.open_loop_horizon].copy()

    def _call_server_with_retry(
        self, request: dict[str, Any], env_id: int | None = None
    ) -> dict[str, Any] | np.ndarray:
        """Send request and return decoded response, reconnecting on transient errors.

        On any reconnect, all per-env session IDs are invalidated and the session_id
        embedded in *request* is refreshed so the retry reaches the server with a
        valid, tracked session rather than a stale UUID.

        Args:
            request: Wire-format request dict (contains session_id; mutated on reconnect).
            env_id: Batch index of the environment that owns this request. Used to
                refresh request['session_id'] after a reconnect invalidates all IDs.

        Returns:
            Decoded server response.
        """
        for attempt_index in range(MAX_RECONNECT_ATTEMPTS):
            try:
                if self._ws is None:
                    raise OSError("WebSocket not connected")
                payload = _pack(request)
                self._ws.send(payload)
                raw = self._ws.recv()
                # Drain any stale reset acknowledgement strings (e.g. "reset successful")
                # that arrived late after a prior _send_reset timed out on recv.
                _MAX_DRAIN = 3
                for _ in range(_MAX_DRAIN):
                    if not isinstance(raw, str):
                        break
                    print(f"[DreamZeroRemotePolicy] Draining stale server message: {raw!r}")
                    raw = self._ws.recv()
                if isinstance(raw, str):
                    raise RuntimeError(f"DreamZero server returned error: {raw}")
                return _unpack(raw)
            except (websockets.exceptions.ConnectionClosed, OSError) as exc:
                is_last_attempt = (attempt_index + 1) >= MAX_RECONNECT_ATTEMPTS
                if is_last_attempt:
                    raise
                print(
                    f"[DreamZeroRemotePolicy] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                _close_ws_best_effort(self._ws)
                self._ws = None
                uri = f"ws://{self.config.remote_host}:{self.config.remote_port}"
                # Reconnect with a paced wait, not a single attempt: when the server sits
                # behind a supervised tunnel, the tunnel accepts TCP again (or refuses
                # instantly) well before the server is reachable through it.
                with contextlib.suppress(OSError, TimeoutError, websockets.exceptions.InvalidMessage):
                    self._ws = self._connect_with_wait(uri, deadline_s=RECONNECT_WAIT_S)
                # If the reconnect deadline expired, self._ws remains None; next iteration's
                # None guard raises OSError which the except clause retries or re-raises.
                # Invalidate all sessions: after reconnect the server has no history.
                if self._session_ids is not None:
                    for i in range(len(self._session_ids)):
                        self._session_ids[i] = None
                    # Mint a fresh session for this request so the retry is consistent.
                    if env_id is not None:
                        request["session_id"] = self._get_or_create_session_id(env_id)
                if self._cached_action_chunks is not None:
                    for i in range(len(self._cached_action_chunks)):
                        self._cached_action_chunks[i] = None
                        self._next_chunk_steps[i] = 0
        raise RuntimeError("unreachable")

    def _send_reset(self, session_uuids: list[str] | None) -> None:
        """Notify the server to discard the given session histories (best-effort, fire-and-forget).

        Local state is always cleared by the caller before this is invoked, so a
        failed send is non-fatal: the server will simply time out the old sessions.

        Args:
            session_uuids: UUIDs to reset, or None to reset all server sessions.
        """
        if self._ws is None:
            return
        with contextlib.suppress(websockets.exceptions.ConnectionClosed, OSError, TimeoutError):
            payload = _pack({"endpoint": "reset", "session_ids": session_uuids})
            self._ws.send(payload)
            self._ws.recv(timeout=60.0)

    @classmethod
    def _connect_with_wait(cls, uri: str, deadline_s: int) -> ws_sync.ClientConnection:
        """Connect, retrying every 15s until ``deadline_s`` elapses.

        The server binds its port only after loading its checkpoint, which can outlast
        client startup — especially when both are cold-started together (e.g. co-scheduled
        OSMO tasks) or when a tunnel accepts TCP connections before the server is up.

        Args:
            uri: WebSocket URI, e.g. ws://localhost:8000.
            deadline_s: Total time budget in seconds; the last failure is re-raised.

        Returns:
            An open ClientConnection.
        """
        retry_interval_s = 15.0
        deadline = time.monotonic() + deadline_s
        while True:
            try:
                return cls._connect(uri)
            except (OSError, TimeoutError, websockets.exceptions.InvalidMessage) as exc:
                if time.monotonic() + retry_interval_s > deadline:
                    raise
                print(f"[DreamZeroRemotePolicy] Server not ready ({exc}); retrying in {retry_interval_s:.0f}s ...")
                time.sleep(retry_interval_s)

    @staticmethod
    def _connect(uri: str) -> ws_sync.ClientConnection:
        """Open a synchronous WebSocket connection.

        The DreamZero server sends a metadata greeting immediately after the
        handshake. Consume it here so the first recv() in _call_server_with_retry
        returns an inference response, not the greeting.

        Args:
            uri: WebSocket URI, e.g. ws://localhost:5000.

        Returns:
            An open ClientConnection.
        """
        ws = ws_sync.connect(uri, open_timeout=60, ping_interval=60, ping_timeout=300)
        try:
            greeting_raw = ws.recv(timeout=30.0)
            greeting = _unpack(greeting_raw)
            print(f"[DreamZeroRemotePolicy] Server greeting: {greeting}")
        except Exception as exc:
            print(f"[DreamZeroRemotePolicy] No server greeting or failed to decode: {exc}")
        return ws


def _close_ws_best_effort(ws: ws_sync.ClientConnection | None) -> None:
    """Close ws, swallowing any errors that indicate the peer is already gone."""
    if ws is None:
        return
    with contextlib.suppress(websockets.exceptions.ConnectionClosed, OSError):
        ws.close()
