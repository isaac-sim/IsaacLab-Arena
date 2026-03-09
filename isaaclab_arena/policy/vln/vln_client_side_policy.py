# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLN client-side policy: queries the remote VLM server for navigation commands.

Current limitation (single-env only):
    Designed for ``num_envs=1``.  See VLN_BENCHMARK_GUIDE.md for multi-env
    limitations and proposed solutions.
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.client_side_policy import ClientSidePolicy
from isaaclab_arena.remote_policy.action_protocol import VlnVelocityActionProtocol
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class VlnClientSidePolicy(ClientSidePolicy):
    """Client-side VLN policy that queries a remote VLM server.

    NaVILA-style scheduling: the VLM is queried only when the previous
    command's duration expires.  Between queries, the last velocity
    command is replayed.
    """

    def __init__(
        self,
        remote_config: RemotePolicyConfig,
        device: str = "cuda",
    ) -> None:
        # ClientSidePolicy requires a config object; we pass None since
        # VLN client has no extra config beyond remote connection settings.
        super().__init__(
            config=None,
            remote_config=remote_config,
            protocol_cls=VlnVelocityActionProtocol,
        )
        self._device = device

        # Scheduling state
        self._step_count: int = 0
        self._target_step: int = 0

        # action_dim and default_duration come from the server via protocol
        # handshake, not hardcoded.
        self._last_cmd = np.zeros(self.action_dim, dtype=np.float32)

        # env dt is computed lazily from the env cfg
        self._env_dt: float | None = None

    # ------------------------------------------------------------------ #
    # PolicyBase interface                                                #
    # ------------------------------------------------------------------ #

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Compute or replay a velocity command.

        Returns:
            A ``[1, action_dim]`` tensor (e.g. ``[1, 3]`` for ``[vx, vy, yaw_rate]``).
        """
        if self._env_dt is None:
            self._env_dt = self._compute_env_dt(env)

        if self._step_count >= self._target_step:
            packed_obs = self.pack_observation_for_server(observation)
            resp = self.remote_client.get_action(observation=packed_obs)

            vel_cmd = np.asarray(
                resp.get("action", np.zeros(self.action_dim)),
                dtype=np.float32,
            )
            # duration is returned by the server per query; it varies based
            # on the VLM output (e.g. "turn left 45" → 1.5s, "move forward 25" → 0.5s).
            duration = float(resp.get("duration", self.protocol.default_duration))

            self._last_cmd = vel_cmd

            if self._env_dt > 0.0 and duration > 0.0:
                steps_to_hold = max(1, int(duration / self._env_dt))
            else:
                steps_to_hold = 1
            self._target_step = self._step_count + steps_to_hold

            # STOP detection: zero velocity + zero duration = VLM says stop.
            # set_stop_called is defined on VLNEnvWrapper, not on the Isaac
            # Lab base env.  We check with hasattr so this policy also works
            # without the wrapper (e.g. in unit tests).
            if np.allclose(vel_cmd, 0.0) and duration <= 0.0:
                if hasattr(env, "set_stop_called"):
                    env.set_stop_called(True)

        self._step_count += 1
        return torch.tensor(self._last_cmd, device=self._device, dtype=torch.float32).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset scheduling state and notify the server."""
        super().reset(env_ids)
        self._step_count = 0
        self._target_step = 0
        self._last_cmd = np.zeros(self.action_dim, dtype=np.float32)

    def set_task_description(self, task_description: str | None) -> str:
        """Set the task description on both client-side and remote policy.

        For VLN, this is called:
          - Once at startup by policy_runner with a generic description.
          - Per episode by run_vln_benchmark.py with the actual instruction
            read from env.extras["current_instruction"].
        """
        self.task_description = task_description
        if task_description is not None:
            self.remote_client.call_endpoint(
                "set_task_description",
                data={"task_description": task_description},
                requires_input=True,
            )
        return self.task_description or ""

    @property
    def is_remote(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_env_dt(env) -> float:
        """Compute wall-clock time per env step from the sim config."""
        try:
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            return float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
        except Exception:
            return 0.02  # fallback: 50 Hz

    # ------------------------------------------------------------------ #
    # CLI helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = ClientSidePolicy.add_remote_args_to_parser(parser)
        parser.add_argument(
            "--policy_device", type=str, default="cuda",
            help="Device for the client-side policy tensors (default: cuda).",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> VlnClientSidePolicy:
        remote_config = ClientSidePolicy.build_remote_config_from_args(args)
        device = getattr(args, "policy_device", "cuda")
        return VlnClientSidePolicy(remote_config=remote_config, device=device)
