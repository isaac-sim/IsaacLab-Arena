# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for relation loss strategies (OnLossStrategy, NextToLossStrategy)."""

import torch

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy, OnLossStrategy
from isaaclab_arena.relations.relations import NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _create_table() -> DummyObject:
    """Create a table-like object at origin."""
    return DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )


def _create_box() -> DummyObject:
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

    # Child centered on table, at correct Z height
    # Table top = 0.1, clearance = 0.01, box bottom = 0.0 (relative)
    # So child_z should be 0.11 for box bottom to be at 0.11
    child_pos = torch.tensor([0.4, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_on_loss_strategy_penalizes_child_outside_x_bounds():
    """Test that On loss penalizes child outside parent's X bounds."""
    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    # Child way to the right (outside table)
    child_pos = torch.tensor([2.0, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_penalizes_child_outside_y_bounds():
    """Test that On loss penalizes child outside parent's Y bounds."""
    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    # Child way to the back (outside table)
    child_pos = torch.tensor([0.4, 2.0, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_penalizes_wrong_z_height():
    """Test that On loss penalizes incorrect Z height."""
    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    # Child at wrong Z (floating above table)
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

    # Table top = 0.1, with 5cm clearance, box bottom should be at 0.15
    # Box min_point[2] = 0.0, so child_z should be 0.15
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

    child_pos = torch.tensor([2.0, 0.4, 0.11])  # Outside bounds

    loss_normal = strategy.compute_loss(relation_normal, child_pos, box.bounding_box, table.bounding_box)
    loss_double = strategy.compute_loss(relation_double, child_pos, box.bounding_box, table.bounding_box)

    assert torch.isclose(loss_double, 2.0 * loss_normal, rtol=1e-5)


def test_on_loss_strategy_constrains_entire_footprint():
    """Test that On loss constrains entire child footprint within parent."""
    table = _create_table()
    box = _create_box()  # 0.2m wide
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    # Child positioned so its right edge just exits the table
    # Table X range: [0, 1], box width: 0.2
    # If child_pos_x = 0.9, box right edge = 0.9 + 0.2 = 1.1 (outside!)
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

    # Parent right edge = 0 + 1.0 = 1.0
    # Child left edge should be at 1.0 + 0.05 = 1.05
    # Child min_point[0] = 0.0, so child_pos[0] should be 1.05
    # Y: cross_position_ratio=0.0 (centered). Valid child Y range is [0.0, 0.8]
    # (parent [0,1] minus child extent 0.2), so centered target = 0.4
    child_pos = torch.tensor([1.05, 0.4, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_next_to_loss_strategy_penalizes_wrong_side():
    """Test that NextTo loss penalizes child on wrong side of parent."""
    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    # Child on the LEFT of parent (wrong side)
    child_pos = torch.tensor([-0.5, 0.5, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_penalizes_outside_y_band():
    """Test that NextTo loss penalizes child outside parent's Y extent."""
    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    # Child at correct X but outside Y range
    child_pos = torch.tensor([1.05, 2.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_penalizes_wrong_distance():
    """Test that NextTo loss penalizes incorrect distance from parent."""
    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    # Child too far from parent (0.5m instead of 0.05m)
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

    child_pos = torch.tensor([1.5, 0.5, 0.0])  # 0.5m gap instead of required 0.05m

    loss_normal = strategy.compute_loss(relation_normal, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    loss_double = strategy.compute_loss(relation_double, child_pos, child_obj.bounding_box, parent_obj.bounding_box)

    assert torch.isclose(loss_double, 2.0 * loss_normal, rtol=1e-5)


def test_next_to_zero_distance_raises():
    """Test that NextTo raises assertion for zero distance (touching not allowed)."""
    parent_obj = _create_table()

    with pytest.raises(AssertionError, match="Distance must be positive"):
        NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.0)
