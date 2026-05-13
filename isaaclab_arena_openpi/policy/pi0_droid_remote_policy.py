# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np
import torch
from typing import Any

import websockets.exceptions
from openpi_client import image_tools, websocket_client_policy

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_openpi.policy.pi0_droid_config import (
    ACTION_DIM,
    ARENA_EXTERNAL_CAMERA_KEY,
    ARENA_WRIST_CAMERA_KEY,
    DEFAULT_VARIANT,
    MAX_RECONNECT_ATTEMPTS,
    OPEN_LOOP_HORIZON_BY_VARIANT,
    TARGET_IMAGE_SIZE,
    Pi0DroidRemotePolicyArgs,
)


class Pi0DroidRemotePolicy(PolicyBase):
    """openpi remote closed-loop policy for DROID, single-env only."""

    name = "pi0_droid_remote"
    config_class = Pi0DroidRemotePolicyArgs

    def __init__(self, config: Pi0DroidRemotePolicyArgs) -> None:
        super().__init__(config)
        assert (
            config.policy_variant in OPEN_LOOP_HORIZON_BY_VARIANT
        ), f"Unknown policy_variant {config.policy_variant!r}; known: {sorted(OPEN_LOOP_HORIZON_BY_VARIANT)}"
        self.open_loop_horizon = OPEN_LOOP_HORIZON_BY_VARIANT[config.policy_variant]
        self.device = config.policy_device

        self._remote_host = config.remote_host
        self._remote_port = config.remote_port

        print(f"[Pi0DroidRemotePolicy] Connecting to openpi server at {self._remote_host}:{self._remote_port} ...")
        self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._remote_host, port=self._remote_port
        )
        print("[Pi0DroidRemotePolicy] Connected.")

        self._cached_action_chunk: np.ndarray | None = None
        self._next_chunk_step: int = 0
        self.task_description: str | None = None

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Pi0 DROID Remote Policy",
            "Arguments for the openpi (pi0 / pi0_fast / pi05) remote client.",
        )
        group.add_argument(
            "--policy_variant",
            type=str,
            default=DEFAULT_VARIANT,
            choices=sorted(OPEN_LOOP_HORIZON_BY_VARIANT),
            help=f"openpi droid checkpoint variant (default: {DEFAULT_VARIANT}).",
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
    def from_args(args: argparse.Namespace) -> Pi0DroidRemotePolicy:
        return Pi0DroidRemotePolicy(
            Pi0DroidRemotePolicyArgs(
                policy_variant=args.policy_variant,
                policy_device=args.policy_device,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
            )
        )

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert env.unwrapped.num_envs == 1, (
            "Pi0DroidRemotePolicy only supports num_envs=1 (openpi server limitation),"
            f" got num_envs={env.unwrapped.num_envs}"
        )
        assert self.task_description, (
            "Pi0DroidRemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        chunk_exhausted = self._cached_action_chunk is None or self._next_chunk_step >= self.open_loop_horizon
        if chunk_exhausted:
            self._cached_action_chunk = self._fetch_action_chunk(observation)
            self._next_chunk_step = 0

        next_action_np = self._cached_action_chunk[self._next_chunk_step]
        self._next_chunk_step += 1
        return torch.from_numpy(next_action_np).to(dtype=torch.float32, device=self.device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._cached_action_chunk = None
        self._next_chunk_step = 0

    def _fetch_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        droid_obs = self._extract_droid_observation(observation)
        server_request = self._pack_pi0_request(droid_obs, self.task_description)
        server_response = self._call_server_with_retry(server_request)

        action_chunk = np.asarray(server_response["actions"])
        assert (
            action_chunk.ndim == 2 and action_chunk.shape[1] == ACTION_DIM
        ), f"Expected actions of shape (H, {ACTION_DIM}); got {action_chunk.shape}"
        assert (
            action_chunk.shape[0] >= self.open_loop_horizon
        ), f"Server returned horizon {action_chunk.shape[0]} < configured open_loop_horizon {self.open_loop_horizon}"
        return action_chunk[: self.open_loop_horizon].astype(np.float32, copy=True)

    def _extract_droid_observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        cam = observation["camera_obs"]
        proprio = observation["policy"]
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in {
                "exterior_image": cam[ARENA_EXTERNAL_CAMERA_KEY][0],
                "wrist_image": cam[ARENA_WRIST_CAMERA_KEY][0],
                "joint_position": proprio["joint_pos"][0],
                "gripper_position": proprio["gripper_pos"][0],
            }.items()
        }

    def _pack_pi0_request(self, droid_obs: dict[str, np.ndarray], language_instruction: str) -> dict[str, Any]:
        target_height, target_width = TARGET_IMAGE_SIZE
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                droid_obs["exterior_image"], target_height, target_width
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                droid_obs["wrist_image"], target_height, target_width
            ),
            "observation/joint_position": droid_obs["joint_position"],
            "observation/gripper_position": droid_obs["gripper_position"],
            "prompt": language_instruction,
        }

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
                    f"[Pi0DroidRemotePolicy] Connection lost ({exc}); reconnecting"
                    f" (attempt {attempt_index + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ..."
                )
                self._websocket_client = websocket_client_policy.WebsocketClientPolicy(
                    host=self._remote_host, port=self._remote_port
                )
                self._cached_action_chunk = None
                self._next_chunk_step = 0
        raise RuntimeError("unreachable")
