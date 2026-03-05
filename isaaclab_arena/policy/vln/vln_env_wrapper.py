# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLN environment wrapper that bridges the low-level locomotion policy.

.. note::

    For standard evaluation, prefer :class:`VlnPolicy` which integrates
    the VLM client + low-level policy into a single ``PolicyBase`` and
    works directly with Arena's ``policy_runner.py``.  This wrapper is
    kept as a lower-level building block for custom evaluation scripts.

Supports ``num_envs >= 1``.

Architecture overview::

    ┌──────────────┐      velocity cmd [N,3]  ┌───────────────────┐
    │  High-level  │  ──────────────────────> │  VLNEnvWrapper    │
    │  VLM policy  │                          │                   │
    │  (remote)    │  <────────────────────── │  - update_command │
    └──────────────┘      camera RGB [N,H,W,C]│  - low-level step │
                                              │  - stuck detect   │
                                              └───────────────────┘
                                                       │
                                               joint actions [N, D]
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │  Isaac Sim env    │
                                              │  (ManagerBased)   │
                                              └───────────────────┘
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class VLNEnvWrapper:
    """Wraps an Isaac Lab ManagerBasedRLEnv for VLN evaluation.

    Args:
        env: The RSL-RL-wrapped Isaac Lab environment.
        low_level_policy: Callable that maps ``obs_tensor -> action_tensor``.
        task_name: String identifier used to select warmup duration.
        max_length: Maximum number of *low-level* steps per episode.
        high_level_obs_key: Key in ``info["observations"]`` for the camera obs.
        use_history_wrapper: Whether ``env`` uses ``RslRlVecEnvHistoryWrapper``.
    """

    def __init__(
        self,
        env,
        low_level_policy,
        task_name: str = "h1",
        max_length: int = 10_000,
        high_level_obs_key: str = "camera_obs",
        use_history_wrapper: bool = True,
    ):
        self.env = env
        self.low_level_policy = low_level_policy
        self.task_name = task_name
        self.max_length = max_length
        self.high_level_obs_key = high_level_obs_key
        self.use_history_wrapper = use_history_wrapper

        # Internal state (initialized in reset)
        self.low_level_obs: torch.Tensor | None = None
        self.low_level_action: torch.Tensor | None = None
        self.env_step: int = 0
        self._num_envs: int | None = None

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def device(self):
        return self.unwrapped.device

    @property
    def num_envs(self) -> int:
        return self._num_envs or getattr(self.unwrapped, "num_envs", 1)

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self) -> tuple[Any, dict]:
        """Reset the environment and warm up the low-level policy."""
        low_level_obs, infos = self.env.reset()
        self.low_level_obs = low_level_obs

        # Detect num_envs from the observation shape
        self._num_envs = low_level_obs.shape[0] if low_level_obs.ndim >= 2 else 1

        zero_cmd = torch.zeros(3, device=self.device)

        # Warmup: stabilize the robot with zero velocity
        warmup_steps = self._get_warmup_steps()
        for i in range(warmup_steps):
            if i % 100 == 0 or i == warmup_steps - 1:
                print(f"[VLNEnvWrapper] Warmup step {i}/{warmup_steps}")
            self.update_command(zero_cmd)
            actions = self.low_level_policy(self.low_level_obs)
            low_level_obs, _, _, infos = self._step_low_level(actions)
            self.low_level_obs = low_level_obs
            self.low_level_action = actions

        self.env_step = 0

        obs = infos["observations"][self.high_level_obs_key]
        return obs, infos

    # ------------------------------------------------------------------ #
    # Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, dict]:
        """Execute one high-level step.

        Args:
            action: Velocity command, shape ``[3]`` or ``[N, 3]``.

        Returns:
            obs: Camera observation for the high-level policy.
            reward: Reward tensor ``[N]``.
            done: Per-env done flags ``[N]`` (bool tensor).
            info: Info dict.
        """
        self.update_command(action)

        low_level_action = self.low_level_policy(self.low_level_obs)
        self.low_level_action = low_level_action

        low_level_obs, reward, done, info = self._step_low_level(low_level_action)
        self.low_level_obs = low_level_obs
        self.env_step += 1

        obs = info["observations"][self.high_level_obs_key]

        if not isinstance(done, torch.Tensor):
            done = torch.tensor([done], device=self.device).expand(self.num_envs)
        if done.ndim == 0:
            done = done.unsqueeze(0).expand(self.num_envs)

        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # Command injection                                                   #
    # ------------------------------------------------------------------ #

    def update_command(self, command: torch.Tensor | list | np.ndarray) -> None:
        """Inject velocity command into the observation buffer.

        Args:
            command: Shape ``[3]`` (broadcast to all envs) or ``[N, 3]``.
        """
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.device, dtype=torch.float32)

        # Broadcast [3] -> [N, 3] if needed
        if command.ndim == 1 and self.num_envs > 1:
            command = command.unsqueeze(0).expand(self.num_envs, -1)

        if self.use_history_wrapper:
            self.low_level_obs[:, 9:12] = command
            if hasattr(self.env, "proprio_obs_buf"):
                self.env.proprio_obs_buf[:, -1, 9:12] = command
        else:
            self.low_level_obs[:, 9:12] = command

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _step_low_level(self, actions: torch.Tensor):
        """Step the underlying environment (handles RSL-RL wrapper API)."""
        result = self.env.step(actions)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
            return obs, reward, done, info
        else:
            return result

    def _get_warmup_steps(self) -> int:
        name = self.task_name.lower()
        if "go2" in name:
            return 100
        elif "h1" in name or "g1" in name:
            return 200
        return 50

    def close(self) -> None:
        self.env.close()
