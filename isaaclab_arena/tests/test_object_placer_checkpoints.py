# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement candidates accumulated across solver checkpoints."""

from isaaclab_arena.relations.object_placer import PlacementCandidate, _CheckpointCandidateAccumulator
from isaaclab_arena.relations.placement_validation import PlacementValidationResults


def _candidate(loss: float, valid: bool, x: float) -> PlacementCandidate:
    """Build a candidate with an inspectable single-object position."""
    return PlacementCandidate(
        loss=loss,
        positions={"object": (x, 0.0, 0.0)},
        validation_results=PlacementValidationResults({"required": valid}),
    )


def _candidates(valid_indices: set[int]) -> list[PlacementCandidate]:
    """Build two candidates for each of two environments."""
    return [_candidate(loss=float(index), valid=index in valid_indices, x=float(index)) for index in range(4)]


def test_accumulator_keeps_first_strict_snapshot():
    """A later strict checkpoint cannot replace the first strict candidate identity."""
    accumulator = _CheckpointCandidateAccumulator(num_envs=1, candidates_per_env=2, results_per_env=1)
    first = [_candidate(loss=0.4, valid=True, x=1.0), _candidate(loss=2.0, valid=False, x=2.0)]
    later = [_candidate(loss=0.1, valid=True, x=9.0), _candidate(loss=1.0, valid=False, x=3.0)]

    assert accumulator.record(first) is True
    assert accumulator.record(later) is True
    assert accumulator.finalize(later)[0][0].positions["object"][0] == 1.0


def test_accumulator_requires_quota_for_every_environment():
    """Readiness requires the strict result quota independently in every environment."""
    accumulator = _CheckpointCandidateAccumulator(num_envs=2, candidates_per_env=2, results_per_env=1)

    assert accumulator.record(_candidates(valid_indices={0})) is False
    assert accumulator.record(_candidates(valid_indices={0, 2})) is True


def test_accumulator_ranks_strict_snapshots_before_invalid_fallbacks_without_duplicates():
    """Finalization prefers ranked strict snapshots and excludes their later identities from fallback."""
    accumulator = _CheckpointCandidateAccumulator(num_envs=1, candidates_per_env=3, results_per_env=3)
    first = [
        _candidate(loss=0.4, valid=True, x=1.0),
        _candidate(loss=4.0, valid=False, x=2.0),
        _candidate(loss=3.0, valid=False, x=3.0),
    ]
    latest = [
        _candidate(loss=0.0, valid=True, x=9.0),
        _candidate(loss=0.1, valid=True, x=4.0),
        _candidate(loss=2.0, valid=False, x=5.0),
    ]

    assert accumulator.record(first) is False
    assert accumulator.record(latest) is False

    result = accumulator.finalize(latest)[0]

    assert [candidate.positions["object"][0] for candidate in result] == [4.0, 1.0, 5.0]


def test_accumulator_copies_strict_snapshot_data():
    """Mutating a recorded candidate later cannot mutate its first strict snapshot."""
    accumulator = _CheckpointCandidateAccumulator(num_envs=1, candidates_per_env=2, results_per_env=1)
    candidate = _candidate(loss=0.4, valid=True, x=1.0)
    invalid = _candidate(loss=1.0, valid=False, x=2.0)

    assert accumulator.record([candidate, invalid]) is True
    candidate.positions["object"] = (9.0, 0.0, 0.0)
    candidate.validation_results.validation_results["required"] = False

    assert accumulator.finalize([candidate, invalid])[0][0].positions["object"][0] == 1.0
