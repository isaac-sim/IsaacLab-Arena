# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for relation loss strategies (OnLossStrategy, NextToLossStrategy, NotNextToLossStrategy)."""

import torch

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy, NotNextToLossStrategy, OnLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, NotNextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _create_table():
    """Create a table-like object at origin."""

    return DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )


def _create_box():
    """Create a small box object."""

    return DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )


# =============================================================================
# OnLossStrategy tests
# =============================================================================


def test_on_loss_strategy_zero_loss_when_perfectly_placed():
    """Test that On loss is zero when child is perfectly placed on parent."""

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_on_loss_strategy_penalizes_child_outside_x_bounds():
    """Test that On loss penalizes child outside parent's X bounds."""

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([2.0, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_penalizes_child_outside_y_bounds():
    """Test that On loss penalizes child outside parent's Y bounds."""

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 2.0, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_penalizes_wrong_z_height():
    """Test that On loss penalizes incorrect Z height."""

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 0.4, 0.5])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_respects_clearance():
    """Test that On loss accounts for clearance parameter."""

    table = _create_table()
    box = _create_box()
    clearance = 0.05
    relation = On(table, clearance_m=clearance)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 0.4, 0.15])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_on_loss_strategy_respects_relation_weight():
    """Test that On loss is scaled by relation_loss_weight."""

    table = _create_table()
    box = _create_box()
    relation_normal = On(table, clearance_m=0.01, relation_loss_weight=1.0)
    relation_double = On(table, clearance_m=0.01, relation_loss_weight=2.0)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([2.0, 0.4, 0.11])

    loss_normal = strategy.compute_loss(relation_normal, child_pos, box.bounding_box, table.bounding_box)
    loss_double = strategy.compute_loss(relation_double, child_pos, box.bounding_box, table.bounding_box)

    assert torch.isclose(loss_double, 2.0 * loss_normal, rtol=1e-5)


def test_on_loss_strategy_constrains_entire_footprint():
    """Test that On loss constrains entire child footprint within parent."""

    table = _create_table()
    box = _create_box()  # 0.2m wide
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.9, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0, "Loss should penalize child footprint extending beyond parent"


# =============================================================================
# NextToLossStrategy tests
# =============================================================================


def test_next_to_loss_strategy_zero_loss_when_perfectly_placed():
    """Test that NextTo loss is zero when child is perfectly placed."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.05, 0.4, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_next_to_loss_strategy_penalizes_wrong_side():
    """Test that NextTo loss penalizes child on wrong side of parent."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([-0.5, 0.5, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_penalizes_outside_y_band():
    """Test that NextTo loss penalizes child outside parent's Y extent."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.05, 2.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_penalizes_wrong_distance():
    """Test that NextTo loss penalizes incorrect distance from parent."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.5, 0.5, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_respects_relation_weight():
    """Test that NextTo loss is scaled by relation_loss_weight."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation_normal = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05, relation_loss_weight=1.0)
    relation_double = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05, relation_loss_weight=2.0)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.5, 0.5, 0.0])

    loss_normal = strategy.compute_loss(relation_normal, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    loss_double = strategy.compute_loss(relation_double, child_pos, child_obj.bounding_box, parent_obj.bounding_box)

    assert torch.isclose(loss_double, 2.0 * loss_normal, rtol=1e-5)


def test_next_to_zero_distance_raises():
    """Test that NextTo raises assertion for zero distance (touching not allowed)."""

    parent_obj = _create_table()

    with pytest.raises(AssertionError, match="Distance must be positive"):
        NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.0)


def test_on_loss_strategy_multi_env_shape_and_values():
    """Test that On with batched (N,3) input returns (N,) loss with correct per-env values."""
    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([[0.4, 0.4, 0.11], [0.4, 0.4, 0.5]])
    parent_world_bbox = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        max_point=torch.tensor([[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]),
    )

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, parent_world_bbox)
    assert loss.shape == (2,)
    assert torch.isclose(loss[0], torch.tensor(0.0), atol=1e-4)
    assert loss[1] > 0.0


def test_next_to_loss_strategy_multi_env_shape_and_values():
    """Test that NextTo with batched (N,3) input returns (N,) loss with correct per-env values."""
    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    # Env 0: perfectly placed. Env 1: wrong side.
    child_pos = torch.tensor([[1.05, 0.4, 0.0], [-0.5, 0.5, 0.0]])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss.shape == (2,)
    assert torch.isclose(loss[0], torch.tensor(0.0), atol=1e-4)
    assert loss[1] > 0.0


# =============================================================================
# NotNextToLossStrategy tests
# =============================================================================


def test_not_next_to_loss_strategy_positive_at_forbidden_spot():
    """Test that NotNextTo loss is positive at the NextTo target spot (no escape made)."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NotNextTo(parent_obj, side=Side.POSITIVE_X)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    child_pos = torch.tensor([1.05, 0.4, 0.5])

    # +X edge at 1.0, safe line at 0.95: remaining_side = 1.05 - 0.95 = 0.10 (the nearest
    # exit). remaining_cross = 0.45 (centred in the Y band). loss = slope * min(0.10, 0.45) = 1.0.
    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-4)


def test_not_next_to_loss_strategy_flat_along_blocked_axis():
    """Test that NotNextTo loss is positive and distance-independent along the blocked axis."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NotNextTo(parent_obj, side=Side.POSITIVE_X)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    # Both are far past the +X edge but within the Y footprint, so the cross-band tent
    # (capped at slope * half-band) sets the loss: backing off further in +X is no escape,
    # so the two positions must share the same positive loss.
    loss_near = strategy.compute_loss(
        relation, torch.tensor([2.0, 0.4, 0.5]), child_obj.bounding_box, parent_obj.bounding_box
    )
    loss_far = strategy.compute_loss(
        relation, torch.tensor([5.0, 0.4, 0.5]), child_obj.bounding_box, parent_obj.bounding_box
    )
    assert loss_near > 0.0
    assert torch.isclose(loss_near, loss_far, atol=1e-4)


def test_not_next_to_loss_strategy_zero_when_crossed_to_opposite_side():
    """Test that NotNextTo loss is zero when the child crosses to the opposite side of the edge."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NotNextTo(parent_obj, side=Side.POSITIVE_X)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    child_pos = torch.tensor([0.3, 0.4, 0.5])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_not_next_to_loss_strategy_zero_when_outside_cross_band():
    """Test that NotNextTo loss is zero when the child is outside the parent's cross band."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NotNextTo(parent_obj, side=Side.POSITIVE_X)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    child_pos = torch.tensor([1.05, 5.0, 0.5])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


@pytest.mark.parametrize(
    "side,target",
    [
        (Side.POSITIVE_X, (1.05, 0.4, 0.5)),
        (Side.NEGATIVE_X, (-0.25, 0.4, 0.5)),
        (Side.POSITIVE_Y, (0.4, 1.05, 0.5)),
        (Side.NEGATIVE_Y, (0.4, -0.25, 0.5)),
    ],
)
def test_not_next_to_loss_strategy_positive_at_each_side_target(side, target):
    """Test that all four Side variants give a positive loss at their respective NextTo target."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NotNextTo(parent_obj, side=side)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    loss = strategy.compute_loss(relation, torch.tensor(target), child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_not_next_to_loss_strategy_respects_relation_weight():
    """Test that NotNextTo loss is scaled by relation_loss_weight."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation_normal = NotNextTo(parent_obj, side=Side.POSITIVE_X, relation_loss_weight=1.0)
    relation_double = NotNextTo(parent_obj, side=Side.POSITIVE_X, relation_loss_weight=2.0)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    child_pos = torch.tensor([1.05, 0.4, 0.5])

    loss_normal = strategy.compute_loss(relation_normal, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    loss_double = strategy.compute_loss(relation_double, child_pos, child_obj.bounding_box, parent_obj.bounding_box)

    assert torch.isclose(loss_double, 2.0 * loss_normal, rtol=1e-5)


def test_not_next_to_loss_strategy_multi_env_shape_and_values():
    """Test that NotNextTo with batched (N,3) input returns (N,) loss with correct per-env values."""

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NotNextTo(parent_obj, side=Side.POSITIVE_X)
    strategy = NotNextToLossStrategy(slope=10.0, margin_m=0.05)

    # Env 0: sitting in the forbidden zone. Env 1: escaped across to the opposite side.
    child_pos = torch.tensor([[1.05, 0.4, 0.5], [0.3, 0.4, 0.5]])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss.shape == (2,)
    assert loss[0] > 0.0
    assert torch.isclose(loss[1], torch.tensor(0.0), atol=1e-6)


# =============================================================================
# RelationSolver integration (NotNextTo) tests
# =============================================================================


def test_solver_drives_box_out_of_forbidden_next_to_zone():
    """Test that the solver drives a box out of the NextTo zone on the table's +X side."""

    table = _create_table()
    box = _create_box()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box.add_relation(NotNextTo(table, side=Side.POSITIVE_X))

    # Box starts inside the forbidden zone, near the +Y end of the table's footprint so
    # the cross-band escape is the closest route (the distance escape no longer exists).
    initial = {table: (0.0, 0.0, 0.0), box: (1.05, 0.82, 0.13)}
    solver = RelationSolver(params=RelationSolverParams(verbose=False, save_position_history=False, max_iters=400))
    final = solver.solve(objects=[table, box], initial_positions=[initial])[0]

    # The solved placement should no longer be penalized by NotNextTo.
    loss = NotNextToLossStrategy(slope=10.0, margin_m=0.05).compute_loss(
        NotNextTo(table, side=Side.POSITIVE_X),
        torch.tensor(final[box]),
        box.bounding_box,
        table.bounding_box,
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4), f"Box should escape the forbidden zone; got {final[box]}"


def test_solver_escapes_box_buried_deep_in_keep_out_zone():
    """Test that the solver frees a box started deep inside the zone (no flat-plateau dead spot)."""

    table = _create_table()
    box = _create_box()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box.add_relation(NotNextTo(table, side=Side.POSITIVE_X))

    # Box starts far past the +X edge and well inside the Y band — squarely in the zone's
    # interior, where a plateau penalty would leave zero gradient. The distance-to-exit
    # loss keeps a downhill toward the nearest footprint edge, so the box still escapes.
    initial = {table: (0.0, 0.0, 0.0), box: (1.4, 0.5, 0.13)}
    solver = RelationSolver(params=RelationSolverParams(verbose=False, save_position_history=False, max_iters=600))
    final = solver.solve(objects=[table, box], initial_positions=[initial])[0]

    loss = NotNextToLossStrategy(slope=10.0, margin_m=0.1).compute_loss(
        NotNextTo(table, side=Side.POSITIVE_X),
        torch.tensor(final[box]),
        box.bounding_box,
        table.bounding_box,
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4), f"Buried box should escape the zone; got {final[box]}"
