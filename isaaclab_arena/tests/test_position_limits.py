# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PositionLimits relation and PositionLimitsLossStrategy."""

import torch

import pytest

from isaaclab_arena.relations.relation_loss_strategies import PositionLimitsLossStrategy
from isaaclab_arena.relations.relations import PositionLimits
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

# Dummy bounding box used for all strategy tests (child object is a 0.1m cube at origin)
_DUMMY_BBOX = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))


# =============================================================================
# PositionLimits construction / validation tests
# =============================================================================


def test_position_limits_requires_at_least_one_bound():
    """PositionLimits() with no bounds should raise AssertionError."""

    with pytest.raises(AssertionError):
        PositionLimits()


def test_position_limits_rejects_min_greater_than_max():
    """PositionLimits with x_min > x_max should raise AssertionError."""

    with pytest.raises(AssertionError):
        PositionLimits(x_min=0.5, x_max=-0.5)


def test_position_limits_allows_single_bound():
    """PositionLimits with only x_min should construct without error."""

    relation = PositionLimits(x_min=-0.3)
    assert relation.x_min == -0.3
    assert relation.x_max is None


# =============================================================================
# PositionLimitsLossStrategy tests
# =============================================================================


def test_position_limits_zero_loss_when_inside():
    """Loss is approximately zero when position is inside the box."""

    relation = PositionLimits(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0)
    strategy = PositionLimitsLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.0, 0.0, 1.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_position_limits_positive_loss_when_outside_x():
    """Loss is positive when position exceeds x_max."""

    relation = PositionLimits(x_min=-1.0, x_max=1.0)
    strategy = PositionLimitsLossStrategy(slope=10.0)
    child_pos = torch.tensor([2.0, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert loss > 0.0


def test_position_limits_positive_loss_when_below_min():
    """Loss is positive when position is below y_min."""

    relation = PositionLimits(y_min=0.5, y_max=1.5)
    strategy = PositionLimitsLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.0, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert loss > 0.0


def test_position_limits_single_bound_min_only():
    """With only x_min set: zero loss above bound, positive loss below."""

    relation = PositionLimits(x_min=0.5)
    strategy = PositionLimitsLossStrategy(slope=10.0)

    pos_above = torch.tensor([1.0, 0.0, 0.0])
    loss_above = strategy.compute_loss(relation, pos_above, _DUMMY_BBOX)
    assert torch.isclose(loss_above, torch.tensor(0.0), atol=1e-6)

    pos_below = torch.tensor([0.0, 0.0, 0.0])
    loss_below = strategy.compute_loss(relation, pos_below, _DUMMY_BBOX)
    assert loss_below > 0.0


def test_position_limits_single_bound_max_only():
    """With only x_max set: zero loss below bound, positive loss above."""

    relation = PositionLimits(x_max=0.5)
    strategy = PositionLimitsLossStrategy(slope=10.0)

    pos_below = torch.tensor([0.0, 0.0, 0.0])
    loss_below = strategy.compute_loss(relation, pos_below, _DUMMY_BBOX)
    assert torch.isclose(loss_below, torch.tensor(0.0), atol=1e-6)

    pos_above = torch.tensor([1.0, 0.0, 0.0])
    loss_above = strategy.compute_loss(relation, pos_above, _DUMMY_BBOX)
    assert loss_above > 0.0


def test_position_limits_loss_scales_with_weight():
    """weight=2.0 gives exactly 2x the loss of weight=1.0."""

    relation_1x = PositionLimits(x_min=-1.0, x_max=1.0, relation_loss_weight=1.0)
    relation_2x = PositionLimits(x_min=-1.0, x_max=1.0, relation_loss_weight=2.0)
    strategy = PositionLimitsLossStrategy(slope=10.0)
    child_pos = torch.tensor([3.0, 0.0, 0.0])

    loss_1x = strategy.compute_loss(relation_1x, child_pos, _DUMMY_BBOX)
    loss_2x = strategy.compute_loss(relation_2x, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss_2x, 2.0 * loss_1x, rtol=1e-5)


def test_position_limits_z_constraint():
    """Z bounds are enforced like X and Y bounds."""

    relation = PositionLimits(z_min=0.0, z_max=1.0)
    strategy = PositionLimitsLossStrategy(slope=10.0)

    pos_inside = torch.tensor([0.0, 0.0, 0.5])
    loss_inside = strategy.compute_loss(relation, pos_inside, _DUMMY_BBOX)
    assert torch.isclose(loss_inside, torch.tensor(0.0), atol=1e-6)

    pos_outside = torch.tensor([0.0, 0.0, 2.0])
    loss_outside = strategy.compute_loss(relation, pos_outside, _DUMMY_BBOX)
    assert loss_outside > 0.0


def test_position_limits_unconstrained_axes_ignored():
    """Only X constrained: extreme Y/Z values produce no loss."""

    relation = PositionLimits(x_min=-1.0, x_max=1.0)
    strategy = PositionLimitsLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.0, 1e6, -1e6])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


# =============================================================================
# Solver integration test
# =============================================================================


from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.pose import Pose


def test_solver_respects_position_limits():
    """Solver moves an object inside the PositionLimits region."""
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 2.0, 0.1)),
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1)),
    )
    box.add_relation(On(table, clearance_m=0.01))
    box.add_relation(PositionLimits(x_min=0.2, x_max=0.5, y_min=0.2, y_max=0.5))

    initial_positions = {table: (0.0, 0.0, 0.0), box: (1.5, 1.5, 0.11)}

    solver = RelationSolver(params=RelationSolverParams(max_iters=300, convergence_threshold=1e-4, verbose=False))
    result = solver.solve(objects=[table, box], initial_positions=initial_positions)

    pos = result[box]
    assert 0.2 <= pos[0] <= 0.5, f"x={pos[0]} should be within [0.2, 0.5]"
    assert 0.2 <= pos[1] <= 0.5, f"y={pos[1]} should be within [0.2, 0.5]"
