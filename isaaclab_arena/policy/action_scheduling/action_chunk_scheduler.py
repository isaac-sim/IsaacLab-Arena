# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ActionChunkScheduler: buffer an action chunk and step through it sequentially."""

from __future__ import annotations

import torch
from collections.abc import Callable

from isaaclab_arena.policy.action_scheduling.action_scheduler import ActionScheduler


class ActionChunkScheduler(ActionScheduler):
    """Buffers one action chunk and replays it one step at a time.

    Fetches a new action tensor from the policy only when the current one is exhausted.
    Per-env tracking allows environments to refetch independently.
    """

    def __init__(
        self,
        num_envs: int,
        action_chunk_length: int,
        action_horizon: int,
        action_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create a scheduler that replays each fetched action chunk per environment.

        Args:
            num_envs: Number of environments.
            action_chunk_length: Number of actions replayed from a chunk before fetching a new one.
            action_horizon: Number of actions stored per environment.
            action_dim: Dimension of a single action.
            device: Device of the stored chunks.
            dtype: Data type of the stored chunks.
        """
        self.num_envs = num_envs
        self.action_chunk_length = action_chunk_length
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = device

        self.current_action_chunk = torch.zeros(
            (num_envs, action_horizon, action_dim),
            dtype=dtype,
            device=device,
        )
        # Use a bool list to indicate that the action chunk is not yet computed for each env
        # True means the action chunk is not yet computed, False means the action
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.env_requires_new_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Fetch-efficiency tracking: how many times each env triggered a chunk fetch,
        # and how many envs actually needed the fetch vs. total (wasted compute detection).
        self._n_fetch_calls: int = 0
        self._total_envs_needed: int = 0
        self._per_env_fetch_count = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def get_action(
        self,
        fetch_action_tensor_fn: Callable[[], torch.Tensor],
        hold_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one action per env, refilling the chunk when needed.

        fetch_action_tensor_fn() must return a tensor of shape (num_envs, horizon, action_dim)
        with horizon >= action_horizon.

        ``hold_action`` is part of the base ``ActionScheduler`` API for schedulers that
        need a fallback for waiting envs; this scheduler doesn't have a waiting state
        so the argument is accepted and ignored.

        Args:
            fetch_action_tensor_fn: Function that returns a new chunk of actions.
            hold_action: Ignored.

        Returns:
            Actions shaped (num_envs, action_dim).
        """
        del hold_action
        if self.env_requires_new_chunk.any():
            new_chunk = fetch_action_tensor_fn()
            mask = self.env_requires_new_chunk
            self.current_action_chunk[mask] = new_chunk[mask]
            self.current_action_index[mask] = 0
            self.env_requires_new_chunk[mask] = False

            self._n_fetch_calls += 1
            n_needed = int(mask.sum().item())
            self._total_envs_needed += n_needed
            self._per_env_fetch_count[mask] += 1

        assert self.current_action_index.min() >= 0, "At least one env's action index is less than 0"
        assert (
            self.current_action_index.max() < self.action_chunk_length
        ), "At least one env's action index is greater than the action chunk length"

        # Take one action per env at the current index (before incrementing)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.current_action_chunk[batch_idx, self.current_action_index]
        assert action.shape == (
            self.num_envs,
            self.action_dim,
        ), f"{action.shape=} != ({self.num_envs}, {self.action_dim})"

        self.current_action_index += 1
        reset_env_ids = self.current_action_index == self.action_chunk_length
        self.current_action_chunk[reset_env_ids] = 0.0
        self.env_requires_new_chunk = self.current_action_index >= self.action_chunk_length
        self.current_action_index[reset_env_ids] = -1

        return action

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset chunking state for the given envs (all if None).

        Args:
            env_ids: Environments to reset, or None for all environments.
        """
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_chunk[env_ids] = True
