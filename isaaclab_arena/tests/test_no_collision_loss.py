# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for NoCollisionLossStrategy and RelationSolver built-in no-overlap behavior."""

import torch

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _create_box(name: str = "box", size: float = 0.2) -> DummyObject:
    """Create a small box (local bbox [0,0,0] to [size, size, size])."""
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(size, size, size)),
    )


def _create_table() -> DummyObject:
    """Create a table-like object at origin."""
    return DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )


def _create_no_collision_scene() -> tuple[DummyObject, DummyObject, DummyObject]:
    """Create table + two boxes with On(table). No-overlap is handled by the solver automatically."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    box_a.add_relation(On(table, clearance_m=0.01))
    box_b.add_relation(On(table, clearance_m=0.01))
    return table, box_a, box_b


# =============================================================================
# NoCollisionLossStrategy tests
# =============================================================================


def test_no_collision_zero_loss_when_fully_separated():
    """Test that NoCollision loss is zero when AABBs do not overlap on any axis."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    strategy = NoCollisionLossStrategy(slope=10.0)

    # Child at origin -> world X [0, 0.2]. Parent at x=1 -> world X [1, 1.2]. No X overlap => volume 0.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((1.0, 0.0, 0.0))

    loss = strategy.compute_loss(
        clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_no_collision_zero_loss_when_separated_on_one_axis_only():
    """Test that NoCollision loss is zero when separated on one axis (overlap_x=0 => volume=0)."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    strategy = NoCollisionLossStrategy(slope=10.0)

    # Child X [0, 0.2], parent X [0.5, 0.7] -> no X overlap. Y and Z overlapping.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.5, 0.0, 0.0))

    loss = strategy.compute_loss(
        clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_no_collision_zero_loss_when_just_touching():
    """Test that NoCollision loss is zero when intervals just touch (clearance_m=0)."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    strategy = NoCollisionLossStrategy(slope=10.0)

    # Child X [0, 0.2], parent X [0.2, 0.4]. Just touching.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.2, 0.0, 0.0))

    loss = strategy.compute_loss(
        clearance_m=0.0, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_no_collision_positive_loss_when_just_touching():
    """Test that NoCollision loss is positive when just touching with default clearance."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    strategy = NoCollisionLossStrategy(slope=10.0)

    # Child X [0, 0.2], parent X [0.2, 0.4]. Just touching; default clearance expands parent so overlap > 0.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.2, 0.0, 0.0))

    loss = strategy.compute_loss(
        clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert loss > 0.0


def test_no_collision_positive_loss_when_3d_overlap():
    """Test that NoCollision loss is positive when AABBs overlap in all three axes."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    strategy = NoCollisionLossStrategy(slope=10.0)

    # Child at (0.1, 0.1, 0), parent at (0.05, 0.05, 0) -> overlap in X, Y, Z.
    child_pos = torch.tensor([0.1, 0.1, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.05, 0.05, 0.0))

    loss = strategy.compute_loss(
        clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert loss > 0.0


def test_no_collision_loss_scales_with_slope():
    """Test that NoCollision loss scales with slope (loss = slope * overlap_volume)."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    strategy_slope_10 = NoCollisionLossStrategy(slope=10.0)
    strategy_slope_20 = NoCollisionLossStrategy(slope=20.0)

    child_pos = torch.tensor([0.1, 0.1, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.05, 0.05, 0.0))

    loss_10 = strategy_slope_10.compute_loss(
        clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    loss_20 = strategy_slope_20.compute_loss(
        clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss_20, 2.0 * loss_10, rtol=1e-5)


def test_no_collision_loss_volume_formula():
    """Test that NoCollision loss equals slope * overlap volume for known overlap (clearance_m=0)."""
    box_a = _create_box("box_a", size=0.2)
    box_b = _create_box("box_b", size=0.2)
    strategy = NoCollisionLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.1, 0.1, 0.1])
    parent_world_bbox = box_b.get_bounding_box().translated((0.15, 0.15, 0.15))
    # Overlap [0.15, 0.3]^3, volume 0.15^3. Expected loss = 10 * 0.15^3.
    expected_loss = 10.0 * (0.15**3)

    loss = strategy.compute_loss(
        clearance_m=0.0, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(expected_loss), rtol=1e-4)


# =============================================================================
# RelationSolver with built-in no-overlap tests
# =============================================================================


def test_relation_solver_no_collision_produces_separated_positions():
    """Test that RelationSolver with On(table) places objects so they do not overlap (built-in no-overlap)."""
    table, box_a, box_b = _create_no_collision_scene()
    objects = [table, box_a, box_b]
    initial_positions = [{
        table: (0.0, 0.0, 0.0),
        box_a: (0.2, 0.2, 0.11),
        box_b: (0.25, 0.25, 0.11),
    }]

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    solver = RelationSolver(params=solver_params)
    result = solver.solve(objects=objects, initial_positions=initial_positions)[0]

    pos_a = result[box_a]
    pos_b = result[box_b]
    bbox_a = box_a.get_bounding_box().translated(pos_a)
    bbox_b = box_b.get_bounding_box().translated(pos_b)

    assert not bbox_a.overlaps(bbox_b).item(), f"Solver should separate boxes; box_a at {pos_a}, box_b at {pos_b}"


def test_solver_respects_clearance_m():
    """With clearance_m=0.05, solved boxes should be at least 5 cm apart."""
    table, box_a, box_b = _create_no_collision_scene()
    objects = [table, box_a, box_b]
    initial_positions = [{
        table: (0.0, 0.0, 0.0),
        box_a: (0.2, 0.2, 0.11),
        box_b: (0.25, 0.25, 0.11),
    }]

    solver_params = RelationSolverParams(max_iters=800, convergence_threshold=1e-6, clearance_m=0.05, verbose=False)
    solver = RelationSolver(params=solver_params)
    result = solver.solve(objects=objects, initial_positions=initial_positions)[0]

    pos_a = result[box_a]
    pos_b = result[box_b]
    bbox_a = box_a.get_bounding_box().translated(pos_a)
    bbox_b = box_b.get_bounding_box().translated(pos_b)

    assert not bbox_a.overlaps(
        bbox_b, margin=0.05
    ).item(), f"Boxes should be at least 5 cm apart; box_a at {pos_a}, box_b at {pos_b}"


def test_negative_clearance_m_raises():
    """Negative clearance_m should be rejected."""
    import pytest

    with pytest.raises(ValueError):
        RelationSolverParams(clearance_m=-0.01)


def test_validation_accepts_on_parent_overlap():
    """Non-anchor sitting On(anchor) should pass validation (On pairs are skipped)."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box = _create_box("box")
    box.add_relation(On(table, clearance_m=0.01))

    # Box at z=0.11 sits on top of table (bbox 0..0.1). With clearance_m=0.01
    # the expanded table extends to 0.11, so they just touch. The On pair
    # should be skipped by validation.
    positions = {table: (0.0, 0.0, 0.0), box: (0.4, 0.4, 0.11)}

    placer = ObjectPlacer(ObjectPlacerParams())
    assert placer._validate_no_overlap(positions)


def test_validation_rejects_non_anchor_overlap():
    """Two overlapping non-anchor boxes should fail validation."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    box_a.add_relation(On(table, clearance_m=0.01))
    box_b.add_relation(On(table, clearance_m=0.01))

    # Both boxes at nearly the same position -- they overlap
    positions = {table: (0.0, 0.0, 0.0), box_a: (0.3, 0.3, 0.11), box_b: (0.35, 0.35, 0.11)}

    placer = ObjectPlacer(ObjectPlacerParams())
    assert not placer._validate_no_overlap(positions)


def test_relation_solver_no_collision_same_inputs_reproducible():
    """Test that RelationSolver with same initial positions yields identical positions."""
    table1, box_a1, box_b1 = _create_no_collision_scene()
    initial = (0.0, 0.0, 0.0), (0.3, 0.3, 0.11), (0.6, 0.6, 0.11)
    initial_positions1 = [{table1: initial[0], box_a1: initial[1], box_b1: initial[2]}]

    solver_params = RelationSolverParams(max_iters=50)
    solver1 = RelationSolver(params=solver_params)
    result1 = solver1.solve(objects=[table1, box_a1, box_b1], initial_positions=initial_positions1)[0]

    table2, box_a2, box_b2 = _create_no_collision_scene()
    initial_positions2 = [{table2: initial[0], box_a2: initial[1], box_b2: initial[2]}]
    solver2 = RelationSolver(params=solver_params)
    result2 = solver2.solve(objects=[table2, box_a2, box_b2], initial_positions=initial_positions2)[0]

    assert result1[box_a1] == result2[box_a2], "box_a positions should match"
    assert result1[box_b1] == result2[box_b2], "box_b positions should match"


def test_no_collision_loss_multi_env_shape_and_values():
    """Test that NoCollision with batched (N,3) input returns (N,) loss with correct per-env values."""
    box_a = _create_box("box_a")
    strategy = NoCollisionLossStrategy(slope=10.0)

    child_pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
    parent_world_bbox = AxisAlignedBoundingBox(
        min_point=torch.tensor([[1.0, 0.0, 0.0], [0.05, 0.05, 0.0]]),
        max_point=torch.tensor([[1.2, 0.2, 0.2], [0.25, 0.25, 0.2]]),
    )

    loss = strategy.compute_loss(
        clearance_m=0.0, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert loss.shape == (2,)
    assert torch.isclose(loss[0], torch.tensor(0.0), atol=1e-5)
    assert loss[1] > 0.0


def test_relation_solver_multi_env_returns_list_of_dicts():
    """Test that solver returns list[dict] when given list[dict] input."""
    table, box_a, box_b = _create_no_collision_scene()
    objects = [table, box_a, box_b]
    initial_positions = [
        {table: (0.0, 0.0, 0.0), box_a: (0.2, 0.2, 0.11), box_b: (0.25, 0.25, 0.11)},
        {table: (0.0, 0.0, 0.0), box_a: (0.3, 0.3, 0.11), box_b: (0.6, 0.6, 0.11)},
    ]

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    solver = RelationSolver(params=solver_params)
    result = solver.solve(objects=objects, initial_positions=initial_positions)

    assert isinstance(result, list)
    assert len(result) == 2
    for d in result:
        assert isinstance(d, dict)
        assert box_a in d
        assert box_b in d
