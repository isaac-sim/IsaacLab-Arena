# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Global round-robin scheduler for replaying episode conditions across parallel env slots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from isaaclab_arena.variations.episode_conditions import EpisodeCondition, RebuildConditions


class ConditionScheduler:
    """Assign source conditions globally in file order, cycling when needed."""

    def __init__(self, conditions: RebuildConditions) -> None:
        assert conditions.num_conditions > 0, "Rebuild conditions must list at least one episode"
        self._conditions = conditions
        self._next_occurrence_index = 0
        self._source_index_by_env: dict[int, int] = {}

    @property
    def num_assignments_started(self) -> int:
        """Number of assignment occurrences started so far."""
        return self._next_occurrence_index

    @property
    def num_assignments_completed(self) -> int:
        """Number of assignment occurrences completed so far."""
        return self._next_occurrence_index - len(self._source_index_by_env)

    def assign_new_episodes(self, env_ids: Sequence[int]) -> None:
        """Assign conditions to envs that do not already have an active assignment."""
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            if env_id in self._source_index_by_env:
                continue
            self._source_index_by_env[env_id] = self._next_occurrence_index % self._conditions.num_conditions
            self._next_occurrence_index += 1

    def complete_episodes(self, env_ids: Sequence[int]) -> None:
        """Mark and remove the active assignment for each finishing env."""
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            self._source_index_by_env.pop(env_id)

    def condition_for_env(self, env_id: int) -> EpisodeCondition:
        """Return the source condition currently assigned to ``env_id``."""
        return self._conditions.episodes[self._source_index_by_env[int(env_id)]]

    def runtime_sample_for(self, variation_key: str, env_ids: Sequence[int]) -> list[Any]:
        """Return one recorded sample row per env in ``env_ids``."""
        rows: list[Any] = []
        for raw_env_id in env_ids:
            condition = self.condition_for_env(int(raw_env_id))
            assert (
                variation_key in condition.runtime_variations
            ), f"Condition {condition.condition_id!r} has no runtime variation {variation_key!r}"
            rows.append(condition.runtime_variations[variation_key])
        return rows
