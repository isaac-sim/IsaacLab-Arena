# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from isaaclab_arena.progress_tracking.progress_tracking_utils import (
    PredicateGroups,
    _format_predicate_groups,
    _normalize_scores,
)


class ProgressObjectiveCompletionMode(str, Enum):
    """How completed groups combine to determine whether a ProgressObjective is complete."""

    ALL = "all"
    """Complete when every group is complete."""

    ANY = "any"
    """Complete when at least one group is complete."""

    CHOOSE = "choose"
    """Complete when at least K groups are complete (K is set on the ProgressObjective)."""


@dataclass
class ProgressObjective:
    """Configuration object that defines a scored predicate sequence to track progress within a task.

    A ProgressObjective specifies what the progress tracker (ProgressTracker) should track.
    Each ProgressObjective holds one or more sequential predicate chains (groups).
    Within a group, predicates run in order. Across groups, predicates run in parallel.

    A group is complete once every predicate in its chain has been satisfied. The ProgressObjective
    is complete when enough of its groups, as set by logical, are complete. ALL requires every
    group, ANY requires at least one, and CHOOSE requires at least K.

    Args:
        name: Identifies the ProgressObjective within the TaskBase.
        predicate_groups: The sequential predicate chains that define the ProgressObjective.
        score: Weight of the ProgressObjective in the TaskBase-level overall_score.
        logical: How completed groups combine to determine if the ProgressObjective is complete.
            A ProgressObjectiveCompletionMode (ALL, ANY, or CHOOSE); a matching string value is also accepted.
        K: Required when logical == "choose". Specifies the number of groups that must be completed
            to consider the ProgressObjective complete.
        description: An optional description of the ProgressObjective.
    """

    name: str
    predicate_groups: PredicateGroups
    score: float = 1.0
    logical: ProgressObjectiveCompletionMode = ProgressObjectiveCompletionMode.ALL
    K: int | None = None
    description: str | None = None

    canonical_predicate_groups: dict[str, list[tuple[Callable, float]]] = field(init=False, repr=False)

    # Index of the parent TaskBase this progress objective belongs to. Set automatically by
    # CompositeTaskBase.get_progress_objectives() when used with composite tasks.
    parent_subtask_idx: int | None = None

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0, f"ProgressObjective '{self.name}': score must be in [0, 1], got {self.score}"
        # Accept either a ProgressObjectiveCompletionMode or its string value; normalize to the enum (raises on invalid).
        self.logical = ProgressObjectiveCompletionMode(self.logical)

        # Format the predicate groups into the canonical form and normalize the scores.
        formatted = _format_predicate_groups(self.predicate_groups)
        normalized = _normalize_scores(formatted)
        self.canonical_predicate_groups = normalized

        # Validate the logical and K parameters.
        num_groups = len(self.canonical_predicate_groups)
        if self.logical == ProgressObjectiveCompletionMode.CHOOSE:
            assert self.K is not None, f"ProgressObjective '{self.name}': K is required when logical='choose'"
            assert (
                1 <= self.K <= num_groups
            ), f"ProgressObjective '{self.name}': K={self.K} but must be in [1, {num_groups}]"

    @property
    def group_names(self) -> list[str]:
        """Returns the names of the groups in the ProgressObjective."""
        return list(self.canonical_predicate_groups.keys())

    def get_chain(self, group_name: str) -> list[tuple[Callable, float]]:
        """Returns the chain of predicates for a given group."""
        return self.canonical_predicate_groups[group_name]
