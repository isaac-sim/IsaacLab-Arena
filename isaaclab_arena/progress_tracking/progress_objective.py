# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from isaaclab_arena.progress_tracking.progress_tracking_utils import (
    DEFAULT_GROUP_NAME,
    Predicate,
    PredicateSequence,
    PredicateSequences,
    _format_predicate_sequences,
    _normalize_scores,
)


class ProgressObjectiveCompletionMode(str, Enum):
    """How completed predicate sequences determine whether a ProgressObjective is complete."""

    ALL = "all"
    """Complete when every sequence is complete."""

    ANY = "any"
    """Complete when at least one sequence is complete."""

    CHOOSE = "choose"
    """Complete when at least K sequences are complete (K is set on the ProgressObjective)."""


@dataclass
class ProgressObjective:
    """Define task progress using one predicate sequence or named independent sequences.

    Provide exactly one of predicate_sequence or predicate_sequences. Predicates within
    each sequence must hold in order. The logical setting determines how many sequences
    must complete.

    Args:
        name: Identifies the ProgressObjective within the TaskBase.
        predicate_sequence: One ordered list of predicates, optionally paired with scores.
        predicate_sequences: Named independent lists of predicates, optionally paired with scores.
        score: Weight of the ProgressObjective in the TaskBase-level overall_score.
        logical: How completed sequences combine to determine if the ProgressObjective is complete.
            A ProgressObjectiveCompletionMode (ALL, ANY, or CHOOSE); a matching string value is also accepted.
        K: Required when logical == "choose". Specifies the number of sequences that must be completed
            to consider the ProgressObjective complete.
        description: An optional description of the ProgressObjective.
    """

    name: str
    predicate_sequence: PredicateSequence | None = None
    """One ordered sequence of predicates."""

    predicate_sequences: PredicateSequences | None = None
    """Named predicate sequences that progress independently."""

    score: float = 1.0
    logical: ProgressObjectiveCompletionMode = ProgressObjectiveCompletionMode.ALL
    K: int | None = None
    description: str | None = None

    canonical_predicate_sequences: dict[str, list[tuple[Predicate, float]]] = field(init=False, repr=False)

    parent_subtask_idx: int | None = None
    """Subtask index assigned by CompositeTaskBase; None for standalone task objectives."""

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0, f"ProgressObjective '{self.name}': score must be in [0, 1], got {self.score}"
        # Accept either a ProgressObjectiveCompletionMode or its string value; normalize to the enum (raises on invalid).
        self.logical = ProgressObjectiveCompletionMode(self.logical)

        assert self.parent_subtask_idx is None or (
            isinstance(self.parent_subtask_idx, int) and self.parent_subtask_idx >= 0
        ), "parent_subtask_idx must be a non-negative integer or None."

        has_single_sequence = self.predicate_sequence is not None
        has_named_sequences = self.predicate_sequences is not None
        assert (
            has_single_sequence != has_named_sequences
        ), "Provide exactly one of predicate_sequence or predicate_sequences."

        if self.predicate_sequence is not None:
            named_sequences = {DEFAULT_GROUP_NAME: self.predicate_sequence}
        else:
            assert isinstance(
                self.predicate_sequences, dict
            ), "predicate_sequences must map names to predicate sequences."
            named_sequences = self.predicate_sequences

        formatted_sequences = _format_predicate_sequences(named_sequences)
        self.canonical_predicate_sequences = _normalize_scores(formatted_sequences)

        # Validate the logical and K parameters.
        num_sequences = len(self.canonical_predicate_sequences)
        if self.logical == ProgressObjectiveCompletionMode.CHOOSE:
            assert self.K is not None, f"ProgressObjective '{self.name}': K is required when logical='choose'"
            assert (
                1 <= self.K <= num_sequences
            ), f"ProgressObjective '{self.name}': K={self.K} but must be in [1, {num_sequences}]"

    @property
    def group_names(self) -> list[str]:
        """Return the sequence names used as group identifiers in progress reports."""
        return list(self.canonical_predicate_sequences.keys())

    def get_chain(self, group_name: str) -> list[tuple[Predicate, float]]:
        """Return the weighted predicate sequence for a progress-report group."""
        return self.canonical_predicate_sequences[group_name]
