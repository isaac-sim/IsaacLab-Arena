# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from isaaclab_arena.progress_tracking.progress_tracking_utils import (
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
    """Define task progress using predicate sequences or composed child objectives.

    Supply exactly one of predicate_sequences or children. A list defines one sequence;
    a dictionary defines named independent sequences. Predicates within each sequence must
    hold in order. The logical setting determines how many sequences must complete.
    A composed objective requires every child to complete, optionally in order.

    A leaf's current outcome uses its final predicates. A composed child's current
    outcome requires its completion and any explicit desired-child-state constraints.

    Args:
        name: Identifies the ProgressObjective within the TaskBase.
        predicate_sequences: One ordered list of predicates or a dictionary of named lists.
            Each list contains callables or (callable, score) pairs.
        score: Weight of the ProgressObjective in the TaskBase-level overall_score.
        logical: How completed sequences combine to determine if the ProgressObjective is complete.
            A ProgressObjectiveCompletionMode (ALL, ANY, or CHOOSE); a matching string value is also accepted.
        K: Required when logical == "choose". Specifies the number of sequences that must be completed
            to consider the ProgressObjective complete.
        description: An optional description of the ProgressObjective.
        children: Child objectives to compose instead of predicate sequences.
        sequential: Whether each child waits for its predecessor to complete.
        desired_child_states: Current final-condition requirements after all child histories complete.
            None entries omit only the current requirement, not the child's history.
    """

    name: str
    predicate_sequences: PredicateSequence | PredicateSequences | None = None
    score: float = 1.0
    logical: ProgressObjectiveCompletionMode = ProgressObjectiveCompletionMode.ALL
    K: int | None = None
    description: str | None = None

    canonical_predicate_sequences: dict[str, list[tuple[Callable, float]]] = field(init=False, repr=False)

    children: list[ProgressObjective] | None = None
    """Child objectives to compose instead of predicate sequences."""

    sequential: bool = False
    """Whether children must complete in order, with at most one child advancing per step."""

    desired_child_states: list[bool | None] | None = None
    """Required current child outcomes after all children complete; None leaves the current outcome unconstrained."""

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0, f"ProgressObjective '{self.name}': score must be in [0, 1], got {self.score}"
        # Accept either a ProgressObjectiveCompletionMode or its string value; normalize to the enum (raises on invalid).
        self.logical = ProgressObjectiveCompletionMode(self.logical)

        assert (self.predicate_sequences is None) != (
            self.children is None
        ), "Provide exactly one of predicate_sequences or children for a progress objective."
        if self.children is not None:
            assert self.children, "A composed progress objective requires at least one child."
            assert self.logical == ProgressObjectiveCompletionMode.ALL, "Composed objectives require every child."
            assert self.K is None, "K only applies to predicate sequences."
            if self.desired_child_states is not None:
                assert len(self.desired_child_states) == len(
                    self.children
                ), "Desired child states must have one entry per child."
                assert all(
                    value is None or isinstance(value, bool) for value in self.desired_child_states
                ), "Desired child states must be True, False, or None."
            self.canonical_predicate_sequences = {}
            return

        assert not self.sequential, "Sequential composition requires children."
        assert self.desired_child_states is None, "Desired child states require children."

        formatted_sequences = _format_predicate_sequences(self.predicate_sequences)
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

    def get_chain(self, group_name: str) -> list[tuple[Callable, float]]:
        """Return the weighted predicate sequence for a progress-report group."""
        return self.canonical_predicate_sequences[group_name]
