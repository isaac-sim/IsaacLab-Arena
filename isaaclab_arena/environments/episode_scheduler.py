# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assign episodes to parallel environments and track their completion."""

from __future__ import annotations

import torch
from collections.abc import Sequence


class EpisodeScheduler:
    """Assign episode indices to available environments until the episode limit is reached.

    The environment requests new episodes and reports when they finish. This
    scheduler tracks each environment's current episode and how many have finished.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        episode_limit: int | None = None,
    ) -> None:
        assert num_envs > 0, "num_envs must be positive"
        self._num_envs = num_envs
        self._device = device
        self._episode_limit: int | None = None
        self._next_global_episode_index = 0
        """Zero-based index for the next episode across all environments; also the number started so far."""
        self._active_global_episode_index_by_env: dict[int, int] = {}
        """Map each active environment ID to its current global episode index; inactive environments are absent."""
        self._num_episodes_started_by_env = [0] * num_envs
        """Episodes started in each environment, used to calculate its local episode index."""
        self.set_episode_limit(episode_limit)

    @property
    def episode_limit(self) -> int | None:
        """Maximum number of episodes to start, or None for no limit."""
        return self._episode_limit

    def set_episode_limit(self, episode_limit: int | None) -> None:
        """Set the episode limit before any episode starts."""
        assert self.num_episodes_started == 0, "The episode limit cannot change after episodes have started"
        assert episode_limit is None or episode_limit >= 0, "The episode limit cannot be negative"
        assert self._episode_limit is None or self._episode_limit == episode_limit, "Episode limits disagree"
        self._episode_limit = episode_limit

    @property
    def num_episodes_started(self) -> int:
        """Number of episodes started across all environments."""
        return self._next_global_episode_index

    @property
    def num_episodes_completed(self) -> int:
        """Number of assigned episodes that have finished."""
        return self.num_episodes_started - len(self._active_global_episode_index_by_env)

    @property
    def is_complete(self) -> bool:
        """Whether a finite episode limit is set and all requested episodes have finished."""
        return self._episode_limit is not None and self.num_episodes_completed == self._episode_limit

    @property
    def active_episode_mask(self) -> torch.Tensor:
        """Return a mask identifying environments with an assigned episode."""
        active_episode_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        if self._active_global_episode_index_by_env:
            active_episode_mask[list(self._active_global_episode_index_by_env)] = True
        return active_episode_mask

    def get_global_episode_index(self, env_id: int) -> int | None:
        """Return the environment's current global episode index, or None when inactive."""
        return self._active_global_episode_index_by_env.get(int(env_id))

    def get_episode_index_in_env(self, env_id: int) -> int:
        """Return the most recently started episode's index within this environment, initially zero."""
        return max(self._num_episodes_started_by_env[int(env_id)] - 1, 0)

    def start_episodes(self, available_env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Assign episodes in environment ID order and return the IDs that receive an episode."""
        selected_env_ids = self._validate_and_sort_env_ids(available_env_ids)
        assert all(
            env_id not in self._active_global_episode_index_by_env for env_id in selected_env_ids
        ), "Cannot start another episode in an active environment"
        if self._episode_limit is not None:
            num_remaining_episodes = self._episode_limit - self.num_episodes_started
            selected_env_ids = selected_env_ids[:num_remaining_episodes]
        for env_id in selected_env_ids:
            self._active_global_episode_index_by_env[env_id] = self._next_global_episode_index
            self._next_global_episode_index += 1
            self._num_episodes_started_by_env[env_id] += 1
        return torch.tensor(selected_env_ids, dtype=torch.long, device=self._device)

    def finish_episodes(self, completed_env_ids: Sequence[int] | torch.Tensor) -> None:
        """Complete the episodes assigned to the supplied active environments."""
        completed_env_ids = self._validate_and_sort_env_ids(completed_env_ids)
        assert all(
            env_id in self._active_global_episode_index_by_env for env_id in completed_env_ids
        ), "Cannot finish an episode in an inactive environment"
        for env_id in completed_env_ids:
            del self._active_global_episode_index_by_env[env_id]

    def _validate_and_sort_env_ids(self, env_ids: Sequence[int] | torch.Tensor) -> list[int]:
        supplied_env_ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        ordered_env_ids = sorted(int(env_id) for env_id in supplied_env_ids)
        assert len(ordered_env_ids) == len(set(ordered_env_ids)), "Environment ids must be unique"
        assert all(0 <= env_id < self._num_envs for env_id in ordered_env_ids), "Environment id is out of range"
        return ordered_env_ids
