# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Global FIFO scheduler for replaying episode conditions across parallel env slots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from isaaclab_arena.variations.episode_conditions import EpisodeCondition, EpisodeConditionsOverlay


class ConditionScheduler:
    """Assign recorded conditions to env slots in file order."""

    def __init__(self, overlay: EpisodeConditionsOverlay) -> None:
        self._overlay = overlay
        self._next_condition_index = 0
        self._env_to_condition_index: dict[int, int] = {}
        self._completed_condition_indices: set[int] = set()
        self._parked_env_ids: set[int] = set()

    @property
    def num_conditions(self) -> int:
        return self._overlay.num_conditions

    @property
    def num_completed(self) -> int:
        return len(self._completed_condition_indices)

    def all_conditions_complete(self) -> bool:
        return self.num_completed >= self.num_conditions

    def condition_for_env(self, env_id: int) -> EpisodeCondition:
        """Return the condition currently bound to ``env_id``."""
        index = self._env_to_condition_index[int(env_id)]
        return self._overlay.episodes[index]

    def condition_id_for_env(self, env_id: int) -> str | None:
        """Return the active replay condition id for ``env_id``, if assigned."""
        env_id = int(env_id)
        if env_id in self._parked_env_ids or env_id not in self._env_to_condition_index:
            return None
        return self.condition_for_env(env_id).condition_id

    def on_pre_reset(self, env_ids: Sequence[int], *, is_initial_reset: bool) -> None:
        """Assign the next condition(s) before reset-mode variation events run."""
        if is_initial_reset:
            for env_id in env_ids:
                self._assign_next_condition(int(env_id))
            return
        for env_id in env_ids:
            env_id = int(env_id)
            if env_id in self._env_to_condition_index:
                self._mark_condition_complete(self._env_to_condition_index[env_id])
            if self._next_condition_index >= self.num_conditions:
                self._parked_env_ids.add(env_id)
                self._env_to_condition_index.pop(env_id, None)
                continue
            self._assign_next_condition(env_id)

    def runtime_sample_for(self, variation_key: str, env_ids: Sequence[int]) -> list[Any]:
        """Return one recorded sample row per env in ``env_ids``."""
        rows: list[Any] = []
        for env_id in env_ids:
            env_id = int(env_id)
            assert env_id not in self._parked_env_ids, f"Env {env_id} is parked but requested {variation_key!r}"
            condition = self.condition_for_env(env_id)
            assert (
                variation_key in condition.runtime_variations
            ), f"Condition {condition.condition_id!r} has no runtime variation {variation_key!r}"
            rows.append(condition.runtime_variations[variation_key])
        return rows

    def _assign_next_condition(self, env_id: int) -> None:
        assert self._next_condition_index < self.num_conditions, "No remaining conditions to assign"
        self._env_to_condition_index[env_id] = self._next_condition_index
        self._next_condition_index += 1

    def _mark_condition_complete(self, condition_index: int) -> None:
        self._completed_condition_indices.add(condition_index)
