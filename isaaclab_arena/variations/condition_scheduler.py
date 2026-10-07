# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Global round-robin scheduler for replaying episode conditions across parallel env slots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.variations.recorded_variation_samples import EpisodeVariationRecord, RebuildVariationRecord


@dataclass(frozen=True)
class ConditionAssignment:
    """One occurrence of a source condition assigned to an environment."""

    occurrence_index: int
    source_record_index: int


class ConditionScheduler:
    """Assign source conditions globally in file order, cycling when needed."""

    def __init__(self, variation_record: RebuildVariationRecord) -> None:
        assert variation_record.num_recorded_episodes > 0, "Rebuild variation record must list at least one episode"
        self._variation_record = variation_record
        self._next_occurrence_index = 0
        self._env_assignments: dict[int, ConditionAssignment] = {}
        self._num_assignments_completed = 0

    @property
    def num_sample_records(self) -> int:
        """Number of source episode records available for replay."""
        return self._variation_record.num_recorded_episodes

    @property
    def num_assignments_started(self) -> int:
        """Number of assignment occurrences started so far."""
        return self._next_occurrence_index

    @property
    def num_assignments_completed(self) -> int:
        """Number of assignment occurrences completed so far."""
        return self._num_assignments_completed

    def assign_new_episodes(self, env_ids: Sequence[int]) -> None:
        """Assign conditions to envs that do not already have an active assignment."""
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            if env_id in self._env_assignments:
                continue
            occurrence_index = self._next_occurrence_index
            self._env_assignments[env_id] = ConditionAssignment(
                occurrence_index=occurrence_index,
                source_record_index=occurrence_index % self.num_sample_records,
            )
            self._next_occurrence_index += 1

    def complete_episodes(self, env_ids: Sequence[int]) -> None:
        """Mark and remove the active assignment for each finishing env."""
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            self._env_assignments.pop(env_id)
            self._num_assignments_completed += 1

    def assignment_for_env(self, env_id: int) -> ConditionAssignment:
        """Return the active assignment for ``env_id``."""
        return self._env_assignments[int(env_id)]

    def source_record_index_for_env(self, env_id: int) -> int:
        """Return the source episode-record index currently assigned to ``env_id``."""
        return self.assignment_for_env(env_id).source_record_index

    def record_for_env(self, env_id: int) -> EpisodeVariationRecord:
        """Return the source episode record currently assigned to ``env_id``."""
        return self._variation_record.episode_records[self.source_record_index_for_env(env_id)]

    def runtime_sample_for(self, variation_key: str, env_ids: Sequence[int]) -> list[Any]:
        """Return one recorded sample row per env in ``env_ids``."""
        rows: list[Any] = []
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            record = self.record_for_env(env_id)
            assert (
                variation_key in record.runtime_samples
            ), f"Source record {self.source_record_index_for_env(env_id)} has no run-time sample {variation_key!r}"
            rows.append(record.runtime_samples[variation_key])
        return rows
