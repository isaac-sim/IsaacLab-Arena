# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import gymnasium as gym
import numpy as np
import torch
import uuid
from typing import Any, get_args

import msgpack
import websockets.exceptions
import websockets.sync.client as ws_sync

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_dreamzero.policy.dreamzero_remote_config import (
    MAX_RECONNECT_ATTEMPTS,
    Cam2Source,
    DreamZeroRemotePolicyConfig,
)
from isaaclab_arena_dreamzero.policy.image_utils import TARGET_H, TARGET_W, resize_with_pad

# DreamZero server action layout: 7 arm joints + 1 gripper.
_ACTION_DIM = 8


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


@register_policy
class DreamZeroRemotePolicy(PolicyBase):
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

    def __init__(self, config: DreamZeroRemotePolicyConfig) -> None:
        super().__init__(config)
        self.task_description: str | None = None
        self.device = config.policy_device

        # Per-env state; lists are None until first get_action call, when num_envs is known.
        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None
        self._session_ids: list[str | None] | None = None

        self._ws: ws_sync.ClientConnection | None = None
        uri = f"ws://{config.remote_host}:{config.remote_port}"
        print(f"[DreamZeroRemotePolicy] Connecting to DreamZero server at {uri} ...")
        self._ws = self._connect(uri)
        print("[DreamZeroRemotePolicy] Connected.")

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add DreamZero CLI arguments to parser."""
        group = parser.add_argument_group(
            "DreamZero Remote Policy",
            "Arguments for the DreamZero remote inference client.",
        )
        group.add_argument("--dreamzero_host", type=str, default="localhost", help="DreamZero server hostname.")
        group.add_argument("--dreamzero_port", type=int, default=5000, help="DreamZero server port.")
        group.add_argument(
            "--dreamzero_open_loop_horizon",
            type=int,
            default=24,
            help="Action steps to replay per server inference call.",
        )
        group.add_argument(
            "--dreamzero_num_arm_joints",
            type=int,
            default=7,
            help="Number of arm DOF in robot_joint_pos (remainder is gripper).",
        )
        group.add_argument(
            "--dreamzero_embodiment",
            type=str,
            default="droid",
            choices=["droid"],
            help="Embodiment the DreamZero checkpoint was trained on. Only 'droid' is currently supported.",
        )
        group.add_argument(
            "--dreamzero_cam_exterior_left",
            type=str,
            default="external_camera_rgb",
            help="Arena camera key for the primary exterior (left shoulder) camera.",
        )
        group.add_argument(
            "--dreamzero_cam2_source",
            type=str,
            default="black",
            choices=list(get_args(Cam2Source)),
            help="Source for the second exterior camera slot.",
        )
        group.add_argument(
            "--dreamzero_cam_exterior_right",
            type=str,
            default="external_camera_2_rgb",
            help="Arena camera key for the right shoulder camera (used when cam2_source='right').",
        )
        group.add_argument(
            "--dreamzero_cam_head",
            type=str,
            default="head_camera",
            help="Arena camera key for the head camera (used when cam2_source='head').",
        )
        group.add_argument(
            "--dreamzero_cam_wrist",
            type=str,
            default="wrist_camera_rgb",
            help="Arena camera key for the wrist camera.",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Torch device for action tensors.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> DreamZeroRemotePolicy:
        """Construct policy from parsed CLI arguments."""
        return DreamZeroRemotePolicy(DreamZeroRemotePolicyConfig.from_cli_args(args))

    # TODO(tstuyck, 2026-07-01): add a RemotePolicy base class
    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return the next scheduled action for every parallel environment.

        Fetches a new action chunk from the server only for environments whose
        current chunk is exhausted. Non-stale environments replay from cache.

        Args:
            env: Gymnasium-wrapped Isaac Lab environment.
            observation: Arena observation dict with 'camera_obs' and 'policy' sub-dicts.

        Returns:
            Float32 tensor of shape (num_envs, _ACTION_DIM) on self.device.
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

        batch = np.stack(actions)  # (num_envs, _ACTION_DIM)
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
            float32 ndarray of shape (open_loop_horizon, _ACTION_DIM).
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
        cam_obs = observation["camera_obs"]
        joint_pos_full = observation["policy"]["robot_joint_pos"][env_id].detach().cpu().numpy().astype(np.float32)

        img0 = self._extract_image(cam_obs, self.config.cam_exterior_left, env_id)
        img1 = self._resolve_cam2(cam_obs, env_id, img0)
        img_wrist = self._extract_image(cam_obs, self.config.cam_wrist, env_id)

        n = self.config.num_arm_joints
        return {
            "observation/exterior_image_0_left": img0,
            "observation/exterior_image_1_left": img1,
            "observation/wrist_image_left": img_wrist,
            "observation/joint_position": joint_pos_full[:n],
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": joint_pos_full[n : n + 1],
            "prompt": self.task_description,
            "session_id": self._get_or_create_session_id(env_id),
            "endpoint": "infer",
        }

    def _extract_image(self, cam_obs: dict[str, Any], cam_name: str, env_id: int) -> np.ndarray:
        """Slice one camera frame, move to CPU, and resize to (TARGET_H, TARGET_W, 3).

        Args:
            cam_obs: Arena camera_obs sub-dict.
            cam_name: Key into cam_obs.
            env_id: Batch index.

        Returns:
            uint8 ndarray of shape (TARGET_H, TARGET_W, 3).
        """
        assert (
            cam_name in cam_obs
        ), f"Camera {cam_name!r} not found in observation. Available cameras: {sorted(cam_obs.keys())}"
        tensor = cam_obs[cam_name][env_id].detach().cpu()
        assert tensor.dtype == torch.uint8, (
            f"Camera {cam_name!r} returned {tensor.dtype} tensor; expected uint8."
            " If Isaac Sim is configured with float images, convert to uint8 in your scene config."
        )
        return resize_with_pad(tensor.numpy(), TARGET_H, TARGET_W)

    def _resolve_cam2(self, cam_obs: dict[str, Any], env_id: int, img0: np.ndarray) -> np.ndarray:
        """Return the image for the second exterior camera slot based on cam2_source.

        Args:
            cam_obs: Arena camera_obs sub-dict.
            env_id: Batch index.
            img0: Already-extracted exterior_left image (used for 'duplicate' mode).

        Returns:
            uint8 ndarray of shape (TARGET_H, TARGET_W, 3).
        """
        src = self.config.cam2_source
        if src == "black":
            return np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        if src == "duplicate":
            return img0.copy()
        if src == "right":
            return self._extract_image(cam_obs, self.config.cam_exterior_right, env_id)
        if src == "head":
            return self._extract_image(cam_obs, self.config.cam_head, env_id)
        raise ValueError(f"Unreachable: unknown cam2_source {src!r}")

    def _parse_action_chunk(self, response: dict[str, Any] | np.ndarray) -> np.ndarray:
        """Normalize server response to a float32 (open_loop_horizon, _ACTION_DIM) array.

        Args:
            response: Decoded MessagePack payload — either a dict with an 'actions' key
                or a bare ndarray. Server may return 7-D actions (no gripper); a zero
                column is appended in that case.

        Returns:
            float32 ndarray of shape (open_loop_horizon, _ACTION_DIM).
        """
        raw = response.get("actions", response) if isinstance(response, dict) else response
        chunk = np.asarray(raw, dtype=np.float32)
        assert chunk.ndim == 2, f"Expected 2-D action chunk from server, got shape {chunk.shape}"
        assert chunk.shape[1] in (7, 8), f"Expected 7 or 8 action dims, got {chunk.shape[1]}"
        if chunk.shape[1] == 7:
            chunk = np.concatenate([chunk, np.zeros((len(chunk), 1), dtype=np.float32)], axis=1)
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
                with contextlib.suppress(OSError):
                    self._ws = self._connect(uri)
                # If _connect raised, self._ws remains None; next iteration's None guard
                # raises OSError which the except clause retries or re-raises.
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
