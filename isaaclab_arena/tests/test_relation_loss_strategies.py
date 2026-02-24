# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for relation loss strategies (OnLossStrategy, NextToLossStrategy)."""

import torch

import pytest


def _create_table():
    """Create a table-like object at origin."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    return DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )


def _create_box():
    """Create a small box object."""
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    return DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )


# =============================================================================
# OnLossStrategy tests
# =============================================================================


def test_on_loss_strategy_zero_loss_when_perfectly_placed():
    """Test that On loss is zero when child is perfectly placed on parent."""
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_on_loss_strategy_penalizes_child_outside_x_bounds():
    """Test that On loss penalizes child outside parent's X bounds."""
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([2.0, 0.4, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_penalizes_child_outside_y_bounds():
    """Test that On loss penalizes child outside parent's Y bounds."""
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 2.0, 0.11])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_penalizes_wrong_z_height():
    """Test that On loss penalizes incorrect Z height."""
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

    table = _create_table()
    box = _create_box()
    relation = On(table, clearance_m=0.01)
    strategy = OnLossStrategy(slope=10.0)

    child_pos = torch.tensor([0.4, 0.4, 0.5])

    loss = strategy.compute_loss(relation, child_pos, box.bounding_box, table.bounding_box)
    assert loss > 0.0


def test_on_loss_strategy_respects_clearance():
    """Test that On loss accounts for clearance parameter."""
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

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
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

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
    from isaaclab_arena.relations.relation_loss_strategies import OnLossStrategy
    from isaaclab_arena.relations.relations import On

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
    from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy
    from isaaclab_arena.relations.relations import NextTo, Side

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.05, 0.4, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_next_to_loss_strategy_penalizes_wrong_side():
    """Test that NextTo loss penalizes child on wrong side of parent."""
    from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy
    from isaaclab_arena.relations.relations import NextTo, Side

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([-0.5, 0.5, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_penalizes_outside_y_band():
    """Test that NextTo loss penalizes child outside parent's Y extent."""
    from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy
    from isaaclab_arena.relations.relations import NextTo, Side

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.05, 2.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_penalizes_wrong_distance():
    """Test that NextTo loss penalizes incorrect distance from parent."""
    from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy
    from isaaclab_arena.relations.relations import NextTo, Side

    parent_obj = _create_table()
    child_obj = _create_box()
    relation = NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.05)
    strategy = NextToLossStrategy(slope=10.0)

    child_pos = torch.tensor([1.5, 0.5, 0.0])

    loss = strategy.compute_loss(relation, child_pos, child_obj.bounding_box, parent_obj.bounding_box)
    assert loss > 0.0


def test_next_to_loss_strategy_respects_relation_weight():
    """Test that NextTo loss is scaled by relation_loss_weight."""
    from isaaclab_arena.relations.relation_loss_strategies import NextToLossStrategy
    from isaaclab_arena.relations.relations import NextTo, Side

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
    from isaaclab_arena.relations.relations import NextTo, Side

    parent_obj = _create_table()

    with pytest.raises(AssertionError, match="Distance must be positive"):
        NextTo(parent_obj, side=Side.POSITIVE_X, distance_m=0.0)
