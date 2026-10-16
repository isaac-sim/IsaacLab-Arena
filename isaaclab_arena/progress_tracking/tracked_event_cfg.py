# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Definitions for recurring observations independent of task success."""

from dataclasses import dataclass

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_SEQUENCE_NAME, Predicate


@dataclass
class TrackedEventCfg:
    """Record recurring completions, requiring the observed condition to clear between occurrences."""

    name: str
    """Unique name within the task, used in recorded event occurrences."""

    predicates: list[Predicate] | dict[str, list[Predicate]]
    """An ordered list, or named independent ordered sequences, all required.

    Use one predicate per dictionary entry for any-order tracking.
    """

    desired_predicate_state: list[bool | None] | None = None
    """Optional current states in addition to history; None entries ignore that member.

    A False entry requires the predicate to complete first, then become false.
    Entries follow sequence insertion order, then predicate order within each sequence.
    Ignored members still gate later members in their sequence.
    Omitting the list uses completion history alone.
    """

    def __post_init__(self):
        assert isinstance(self.name, str) and self.name.strip(), "Tracked events require a nonempty name."
        criteria = self._to_completion_criteria()
        num_predicates = 0
        for sequence in criteria.canonical_predicate_sequences.values():
            num_predicates += len(sequence)
            assert not any(
                isinstance(predicate, type) for predicate, _ in sequence
            ), "Configure predicate classes with TerminationTermCfg(func=MyPredicate)."
        if self.desired_predicate_state is not None:
            assert len(self.desired_predicate_state) == num_predicates, "Provide one desired state per predicate."
            assert all(
                state is None or isinstance(state, bool) for state in self.desired_predicate_state
            ), "Desired predicate states must be True, False, or None."
            assert any(state is not None for state in self.desired_predicate_state), "At least one member is required."

    def _to_completion_criteria(self) -> CompletionCriteria:
        """Build completion criteria using the configured sequence ordering."""
        assert isinstance(self.predicates, (list, dict)), "predicates must be a list or a dictionary of named lists."
        sequences = self.predicates
        if isinstance(sequences, list):
            sequences = {DEFAULT_SEQUENCE_NAME: sequences}
        return CompletionCriteria(name=self.name, predicate_sequences=sequences)


@dataclass(frozen=True)
class TrackedEventOccurrence:
    """One completed occurrence, retained until the environment starts a new episode."""

    env_idx: int
    """Index of the parallel environment."""

    name: str
    """Name of the event that occurred."""

    step: int
    """Episode control step when the event occurred."""
