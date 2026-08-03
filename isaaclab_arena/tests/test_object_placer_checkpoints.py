# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement candidates accumulated across solver checkpoints."""

import torch

from isaaclab_arena.relations.object_placer import ObjectPlacer, PlacementCandidate, _CheckpointCandidateAccumulator
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_validation import PlacementValidationResults
from isaaclab_arena.relations.relation_solver import RelationSolver, RelationSolverCheckpoint
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.tests.test_object_placer_reproducibility import _create_test_objects


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
        _candidate(loss=0.0, valid=False, x=9.0),
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


class _PositiveXValidator:
    """Test validator that accepts candidates whose inspected object has positive X."""

    check = "positive_x"
    run_after_inexpensive_checks = False

    def __init__(self, inspected_object):
        self._inspected_object = inspected_object

    def validate_batch(self, positions, orientations, bboxes, collision_objects):
        return [candidate[self._inspected_object][0] > 0.0 for candidate in positions]


def test_place_stops_when_one_strict_layout_exists():
    """A real placement stops once a checkpoint contains one strictly valid layout."""
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=1,
            max_placement_attempts=10,
            solver_params=RelationSolverParams(
                max_iters=600,
                checkpoint_iters=(25, 50, 100, 600),
                verbose=False,
            ),
        )
    )

    result = placer.place(list(_create_test_objects()))[0]

    assert result.success
    assert placer.last_iterations_run <= 100


def test_ranked_placement_waits_for_requested_strict_quota(monkeypatch):
    """Ranked placement continues until every requested result has passed validation."""
    objects = list(_create_test_objects())
    inspected_object = objects[1]
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=1,
            max_placement_attempts=2,
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(max_iters=100, checkpoint_iters=(25, 50, 100), verbose=False),
        )
    )
    placer._validators = [_PositiveXValidator(inspected_object)]

    def solve_with_checkpoint_stream(
        solver,
        solver_objects,
        initial_positions,
        *,
        checkpoint_callback=None,
        **kwargs,
    ):
        assert checkpoint_callback is not None
        for iteration, num_valid in ((25, 1), (50, 3), (100, len(initial_positions))):
            positions = [dict(candidate) for candidate in initial_positions]
            for candidate_idx, candidate in enumerate(positions):
                candidate[inspected_object] = (1.0 if candidate_idx < num_valid else 0.0, 0.0, 0.0)
            losses = tuple(float(candidate_idx) for candidate_idx in range(len(positions)))
            should_stop = checkpoint_callback(
                RelationSolverCheckpoint(
                    iteration=iteration,
                    positions=positions,
                    losses=losses,
                    elapsed_ms=float(iteration),
                )
            )
            solver._last_iterations_run = iteration
            solver._last_loss_per_env = torch.tensor(losses)
            if should_stop:
                return positions
        return positions

    monkeypatch.setattr(RelationSolver, "solve", solve_with_checkpoint_stream)

    results = placer.place_ranked_per_env(objects, num_envs=1, results_per_env=3)

    assert len([result for result in results[0] if result.success]) == 3
    assert placer.last_iterations_run == 50
