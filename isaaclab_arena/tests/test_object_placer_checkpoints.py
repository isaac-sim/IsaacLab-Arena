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
from isaaclab_arena.relations.relations import AtPosition, IsAnchor
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.tests.test_object_placer_reproducibility import _create_test_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


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


def test_unvalidated_relation_does_not_make_an_early_checkpoint_strict():
    """A validator-clean checkpoint cannot stop before an uncovered solver relation is satisfied."""
    point_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.0, 0.0, 0.0))
    anchor = DummyObject(name="anchor", bounding_box=point_bbox)
    anchor.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    anchor.add_relation(IsAnchor())
    target_z = 0.85
    subject = DummyObject(name="subject", bounding_box=point_bbox)
    subject.add_relation(AtPosition(z=target_z))
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=1,
            max_placement_attempts=1,
            apply_positions_to_objects=False,
            enabled_checks=set(),
            solver_params=RelationSolverParams(
                max_iters=100,
                checkpoint_iters=(25, 100),
                clearance_m=0.0,
                verbose=False,
            ),
        )
    )

    result = placer.place([anchor, subject])[0]

    assert placer.last_iterations_run == 100
    assert abs(result.positions[subject][2] - target_z) < 0.05


def test_profile_records_checkpoint_and_strict_counts():
    """Profile mode records the solver checkpoints and strict result quota."""
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=1,
            max_placement_attempts=1,
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(
                max_iters=100,
                checkpoint_iters=(25, 50, 100),
                verbose=False,
                profile=True,
            ),
        )
    )

    placer.place(list(_create_test_objects()))

    profile = placer.last_profile
    assert profile is not None
    assert profile.device == "cpu"
    assert profile.candidate_count == 1
    assert profile.cumulative_iterations == placer.last_iterations_run
    assert profile.strict_layouts_per_env == (1,)
    assert profile.checkpoints[-1].iteration == placer.last_iterations_run
    assert profile.validation_counts
    assert set(profile.validation_counts) == set(profile.validation_times_ms)
    assert all(elapsed_ms >= 0.0 for elapsed_ms in profile.validation_times_ms.values())


def test_zero_iteration_profile_records_final_checkpoint():
    """A zero-iteration solve still profiles its final validation inspection."""
    placer = ObjectPlacer(
        ObjectPlacerParams(
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(max_iters=0, verbose=False, profile=True),
        )
    )

    placer.place(list(_create_test_objects()))

    profile = placer.last_profile
    assert profile is not None
    assert profile.cumulative_iterations == 0
    assert profile.checkpoints[-1].iteration == 0


def test_ranked_placement_waits_for_requested_strict_quota(monkeypatch):
    """The checkpoint callback stops only when production validators reach the requested strict quota."""
    objects = list(_create_test_objects())
    desk, box1, box2 = objects
    placer = ObjectPlacer(
        ObjectPlacerParams(
            placement_seed=1,
            max_placement_attempts=2,
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(max_iters=100, checkpoint_iters=(25, 50, 100), verbose=False),
        )
    )
    callback_stop_iterations = []

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
                candidate.update({
                    desk: (0.0, 0.0, 0.0),
                    box1: (0.2, 0.2, 0.11),
                    # Valid: box2 starts 5 cm past box1's +X face. Invalid candidates
                    # stay on the desk without overlap but violate that NextTo distance.
                    box2: ((0.45 if candidate_idx < num_valid else 0.7), 0.225, 0.11),
                })
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
                callback_stop_iterations.append(iteration)
                return positions
        return positions

    monkeypatch.setattr(RelationSolver, "solve", solve_with_checkpoint_stream)

    results = placer.place_ranked_per_env(objects, num_envs=1, results_per_env=3)

    assert len([result for result in results[0] if result.success]) == 3
    assert callback_stop_iterations == [50]
    assert placer.last_iterations_run == 50
