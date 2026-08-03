# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RelationSolver checkpoint configuration and benchmark reporting."""

import importlib
import json
import torch
from dataclasses import FrozenInstanceError

import pytest

from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _create_test_objects() -> tuple[DummyObject, DummyObject, DummyObject]:
    """Create a deterministic anchor and two related placement objects."""
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.add_relation(IsAnchor())

    box1 = DummyObject(
        name="box1",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box2 = DummyObject(
        name="box2",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
    )
    box1.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(NextTo(box1, side=Side.POSITIVE_X, distance_m=0.05))
    return desk, box1, box2


def _initial_positions(
    objects: list[DummyObject],
) -> dict[DummyObject, tuple[float, float, float]]:
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


def test_bbox_device_is_cpu_even_when_cuda_is_available(monkeypatch):
    """BBOX solving remains on CPU even when CUDA is available."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert RelationSolver._select_device(mesh_collision_enabled=False) == torch.device("cpu")


def test_mesh_device_uses_cuda_when_available(monkeypatch):
    """MESH solving uses CUDA when it is available."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert RelationSolver._select_device(mesh_collision_enabled=True) == torch.device("cuda")


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


def _benchmark_module():
    """Load the optional simulator-free placement benchmark CLI."""
    return importlib.import_module("isaaclab_arena.scripts.benchmark_relation_placement")


def test_benchmark_parse_args_accepts_bbox_cpu_case_matrix(tmp_path):
    """The benchmark CLI preserves each requested BBOX-only matrix input."""
    module = _benchmark_module()

    args = module.parse_args([
        "--device-policy",
        "bbox-cpu",
        "--candidates",
        "1",
        "10",
        "50",
        "--iterations",
        "25",
        "50",
        "100",
        "600",
        "--json-output",
        str(tmp_path / "benchmark.json"),
    ])

    assert args.device_policy == "bbox-cpu"
    assert args.candidates == [1, 10, 50]
    assert args.iterations == [25, 50, 100, 600]
    assert args.json_output == tmp_path / "benchmark.json"


def test_benchmark_case_serializes_stable_profile_schema_from_dummy_objects():
    """A BBOX benchmark case emits real strict/profile counters without volatile identity fields."""
    module = _benchmark_module()
    objects = list(_create_test_objects())
    objects[0].set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

    result = module.run_benchmark_case(
        objects=objects,
        seed=1,
        candidate_count=1,
        requested_layouts=1,
        max_iterations=100,
        device_policy="bbox-cpu",
    )
    serialized = json.loads(json.dumps(result, sort_keys=True))

    assert {
        "revision",
        "seed",
        "collision_mode",
        "candidate_count",
        "requested_layouts",
        "device",
        "iterations",
        "strict_count",
        "generation_ms",
        "validation_ms",
        "optimizer_ms",
        "used_best_loss_fallback",
        "resolver_ms",
        "layouts",
    } <= serialized.keys()
    assert serialized["identity"] == {
        "candidate_count": 1,
        "collision_mode": "bbox",
        "max_iterations": 100,
        "requested_layouts": 1,
        "revision": serialized["revision"],
        "seed": 1,
    }
    assert serialized["device"] == "cpu"
    assert serialized["strict_count"] == 1
    assert len(serialized["layouts"]) == 1
    assert serialized["layouts"][0]["success"] is True
    assert serialized["used_best_loss_fallback"] is False
    assert all("time" not in field for field in serialized["identity"])


def test_benchmark_repeats_fixed_seed_with_strict_validator_valid_layouts(tmp_path):
    """Repeated BBOX CLI cases preserve validator-valid strict layouts and work counters."""
    module = _benchmark_module()
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    args = [
        "--device-policy",
        "bbox-cpu",
        "--candidates",
        "1",
        "--iterations",
        "100",
        "--seed",
        "1",
    ]

    assert module.main([*args, "--json-output", str(first_output)]) == 0
    assert module.main([*args, "--json-output", str(second_output)]) == 0

    first_payload = json.loads(first_output.read_text())
    second_payload = json.loads(second_output.read_text())
    assert set(first_payload) == {"revision", "configuration", "warmup_ms", "cases"}
    assert first_payload["configuration"] == {
        "device_policy": "bbox-cpu",
        "seed": 1,
        "candidates": [1],
        "iterations": [100],
        "requested_layouts": 1,
    }
    assert len(first_payload["cases"]) == 1
    first_case = first_payload["cases"][0]
    second_case = second_payload["cases"][0]
    assert {
        "resolver_ms",
        "layouts",
        "strict_count",
        "used_best_loss_fallback",
    } <= first_case.keys()
    assert first_case["strict_count"] == first_case["requested_layouts"]
    assert first_case["used_best_loss_fallback"] is False
    assert all(layout["success"] for layout in first_case["layouts"])

    work_and_layout_fields = (
        "identity",
        "device",
        "iterations",
        "strict_count",
        "used_best_loss_fallback",
        "layouts",
    )
    assert {field: first_case[field] for field in work_and_layout_fields} == {
        field: second_case[field] for field in work_and_layout_fields
    }
