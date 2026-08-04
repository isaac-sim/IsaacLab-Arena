# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RelationSolver checkpoints."""

import torch
from dataclasses import FrozenInstanceError

import pytest

from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.tests.test_object_placer_reproducibility import _create_test_objects


def _initial_positions(objects):
    """Return the fixed reproducibility-test positions for the test objects."""
    desk, box1, box2 = objects
    return {desk: (0.0, 0.0, 0.0), box1: (0.5, 0.5, 0.5), box2: (0.3, 0.7, 0.3)}


def test_callback_stops_at_first_requested_checkpoint():
    """A callback can successfully stop solving at its first checkpoint."""
    objects = list(_create_test_objects())
    initial = _initial_positions(objects)
    seen = []

    def stop(checkpoint):
        seen.append(checkpoint.iteration)
        return True

    solver = RelationSolver(RelationSolverParams(max_iters=100, checkpoint_iters=(25, 50, 100), verbose=False))
    solver.solve(objects, [initial], checkpoint_callback=stop)

    assert seen == [25]
    assert solver.last_iterations_run == 25


def test_default_checkpoints_are_capped_by_max_iters():
    """Default checkpoints include the maximum iteration cap."""

    params = RelationSolverParams(max_iters=120)

    assert params.get_checkpoints() == (25, 50, 100, 120)


def test_checkpoint_configuration_must_be_strictly_increasing():
    """Checkpoint iterations must form a strictly increasing sequence."""

    with pytest.raises(AssertionError, match="strictly increasing"):
        RelationSolverParams(checkpoint_iters=(25, 25, 100))


def test_position_history_defaults_off():
    """Position history capture is disabled by default."""

    assert RelationSolverParams().save_position_history is False


@pytest.mark.parametrize(("mesh_collision_enabled", "expected"), [(False, "cpu"), (True, "cuda")])
def test_solver_device_policy(monkeypatch, mesh_collision_enabled, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert RelationSolver._select_device(mesh_collision_enabled) == torch.device(expected)


def test_callback_runs_at_cumulative_checkpoints_until_stop():
    """A continuing callback observes every configured cumulative checkpoint."""
    objects = list(_create_test_objects())
    seen = []
    solver = RelationSolver(RelationSolverParams(max_iters=80, checkpoint_iters=(10, 40), verbose=False))

    solver.solve(
        objects,
        [_initial_positions(objects)],
        checkpoint_callback=lambda checkpoint: seen.append(checkpoint.iteration) or False,
    )

    assert seen == [10, 40, 80]
    assert solver.last_iterations_run == 80
    assert tuple(iteration for iteration, _ in solver.last_checkpoint_profiles) == (10, 40, 80)


def test_disabled_position_history_keeps_only_loss_history():
    """Default history settings keep scalar losses without position snapshots."""
    objects = list(_create_test_objects())
    solver = RelationSolver(RelationSolverParams(max_iters=25, save_position_history=False, verbose=False))

    solver.solve(objects, [_initial_positions(objects)])

    assert solver.last_position_history == []
    assert len(solver.last_loss_history) == 25


def test_checkpoint_positions_are_not_mutated_by_later_optimizer_steps():
    """Checkpoint position snapshots stay stable as subsequent steps run."""
    objects = list(_create_test_objects())
    snapshots = []
    expected_first_positions = []

    def capture(checkpoint):
        snapshots.append(checkpoint)
        expected_first_positions.append([dict(positions) for positions in checkpoint.positions])
        return False

    solver = RelationSolver(RelationSolverParams(max_iters=20, checkpoint_iters=(10,), verbose=False))
    solver.solve(objects, [_initial_positions(objects)], checkpoint_callback=capture)

    assert snapshots[0].positions == expected_first_positions[0]
    assert len(snapshots) == 2
    assert solver.last_iterations_run == 20

    with pytest.raises(FrozenInstanceError):
        snapshots[0].iteration = 0
