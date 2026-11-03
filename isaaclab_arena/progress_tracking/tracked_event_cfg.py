# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Definitions for recurring observations independent of task success."""

from dataclasses import dataclass

from isaaclab_arena.progress_tracking.progress_tracking_utils import Predicate, _is_predicate


@dataclass
class TrackedEventCfg:
    """Record an ordered predicate sequence each time it completes during an episode."""

    name: str
    """Unique name within the task, used in recorded event occurrences."""

    predicate_sequence: list[Predicate]
    """Nonempty ordered predicates; use a one-element list for a single condition.

    Earlier predicates need not remain true when the sequence completes. For a
    recurring event, the final predicate must become false before the event resets.
    """

    latch: bool = False
    """Keep the first completion until episode reset instead of re-arming the event."""

    def __post_init__(self):
        assert isinstance(self.latch, bool), "latch must be a boolean."
        assert isinstance(self.name, str) and self.name.strip(), "Tracked events require a nonempty name."
        assert (
            isinstance(self.predicate_sequence, list) and self.predicate_sequence
        ), "predicate_sequence must be a nonempty list."
        for predicate in self.predicate_sequence:
            assert _is_predicate(
                predicate
            ), "Each entry must be a predicate; (predicate, score) pairs are not supported."
            assert not isinstance(
                predicate, type
            ), "Configure predicate classes with TerminationTermCfg(func=MyPredicate)."


@dataclass(frozen=True)
class TrackedEventOccurrence:
    """One completed occurrence, retained until the environment starts a new episode."""

    env_idx: int
    """Index of the parallel environment."""

    name: str
    """Name of the event that occurred."""

    step: int
    """Episode control step when the event occurred."""
