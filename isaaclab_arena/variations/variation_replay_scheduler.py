# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Global round-robin scheduler for replaying variation records across parallel env slots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from isaaclab_arena.variations.recorded_variation_samples import EpisodeVariationRecord, RebuildVariationRecord


class VariationReplayScheduler:
    """Assign recorded episode variation records to parallel environment ids during rollouts.

    The scheduler tracks the next variation-record index and advances it whenever a newly started episode
    receives an assignment. Each environment ID retains its assigned index until that episode completes;
    assignment follows recorded JSONL order globally across environment ids and loops after the final record.
    """

    def __init__(self, variation_record: RebuildVariationRecord) -> None:
        assert variation_record.num_recorded_episodes > 0, "Rebuild variation record must list at least one episode"
        self._variation_record = variation_record
        self._next_occurrence_index = 0
        self._source_record_index_by_env: dict[int, int] = {}

    @property
    def num_assignments_started(self) -> int:
        """Number of assignment occurrences started so far."""
        return self._next_occurrence_index

    @property
    def num_assignments_completed(self) -> int:
        """Number of assignment occurrences completed so far."""
        return self._next_occurrence_index - len(self._source_record_index_by_env)

    def assign_new_episodes(self, env_ids: Sequence[int]) -> None:
        """Assign source records to envs that do not already have an active assignment."""
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            assert (
                env_id not in self._source_record_index_by_env
            ), f"Environment {env_id} already has an active variation-record assignment."
            self._source_record_index_by_env[env_id] = (
                self._next_occurrence_index % self._variation_record.num_recorded_episodes
            )
            self._next_occurrence_index += 1

    def complete_episodes(self, env_ids: Sequence[int]) -> None:
        """Mark and remove the active assignment for each finishing env."""
        for raw_env_id in env_ids:
            self._source_record_index_by_env.pop(int(raw_env_id))

    def source_record_index_for_env(self, env_id: int) -> int:
        """Return the source episode-record index currently assigned to ``env_id``."""
        return self._source_record_index_by_env[int(env_id)]

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
