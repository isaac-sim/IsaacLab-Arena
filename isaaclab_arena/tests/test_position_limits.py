# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for box and cylindrical position-limit relations and loss strategies."""

import torch

import pytest

from isaaclab_arena.relations.relation_loss_strategies import (
    PositionLimitsBoxLossStrategy,
    PositionLimitsCylindricalLossStrategy,
)
from isaaclab_arena.relations.relations import PositionLimitsBox, PositionLimitsCylindrical
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

# Dummy bounding box used for all strategy tests (child object is a 0.1m cube at origin)
_DUMMY_BBOX = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))


# =============================================================================
# PositionLimitsBox construction / validation tests
# =============================================================================


def test_position_limits_requires_at_least_one_bound():
    """PositionLimitsBox() with no bounds should raise AssertionError."""

    with pytest.raises(AssertionError):
        PositionLimitsBox()


def test_position_limits_rejects_min_greater_than_max():
    """PositionLimitsBox with x_min > x_max should raise AssertionError."""

    with pytest.raises(AssertionError):
        PositionLimitsBox(x_min=0.5, x_max=-0.5)


def test_position_limits_allows_single_bound():
    """PositionLimitsBox with only x_min should construct without error."""

    relation = PositionLimitsBox(x_min=-0.3)
    assert relation.x_min == -0.3
    assert relation.x_max is None


def test_position_limits_preserves_original_positional_weight_argument():
    """The seventh positional argument remains the relation loss weight."""

    relation = PositionLimitsBox(-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, 2.0)
    assert relation.relation_loss_weight == 2.0


def test_position_limits_cylindrical_requires_at_least_one_bound():
    """PositionLimitsCylindrical requires at least one radial bound."""

    with pytest.raises(AssertionError):
        PositionLimitsCylindrical(center_x=0.0, center_y=0.0)


def test_position_limits_cylindrical_rejects_min_greater_than_max():
    """PositionLimitsCylindrical rejects radius_min > radius_max."""

    with pytest.raises(AssertionError):
        PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_min=0.5, radius_max=0.2)


def test_position_limits_cylindrical_allows_single_bound():
    """PositionLimitsCylindrical constructs with only radius_min."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_min=0.3)
    assert relation.radius_min == 0.3
    assert relation.radius_max is None


# =============================================================================
# PositionLimitsBoxLossStrategy tests
# =============================================================================


def test_position_limits_zero_loss_when_inside():
    """Loss is approximately zero when position is inside the box."""

    relation = PositionLimitsBox(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.0, 0.0, 1.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_position_limits_positive_loss_when_outside_x():
    """Loss is positive when position exceeds x_max."""

    relation = PositionLimitsBox(x_min=-1.0, x_max=1.0)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)
    child_pos = torch.tensor([2.0, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert loss > 0.0


def test_position_limits_positive_loss_when_below_min():
    """Loss is positive when position is below y_min."""

    relation = PositionLimitsBox(y_min=0.5, y_max=1.5)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.0, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert loss > 0.0


def test_position_limits_single_bound_min_only():
    """With only x_min set: zero loss above bound, positive loss below."""

    relation = PositionLimitsBox(x_min=0.5)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)

    pos_above = torch.tensor([1.0, 0.0, 0.0])
    loss_above = strategy.compute_loss(relation, pos_above, _DUMMY_BBOX)
    assert torch.isclose(loss_above, torch.tensor(0.0), atol=1e-6)

    pos_below = torch.tensor([0.0, 0.0, 0.0])
    loss_below = strategy.compute_loss(relation, pos_below, _DUMMY_BBOX)
    assert loss_below > 0.0


def test_position_limits_single_bound_max_only():
    """With only x_max set: zero loss below bound, positive loss above."""

    relation = PositionLimitsBox(x_max=0.5)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)

    pos_below = torch.tensor([0.0, 0.0, 0.0])
    loss_below = strategy.compute_loss(relation, pos_below, _DUMMY_BBOX)
    assert torch.isclose(loss_below, torch.tensor(0.0), atol=1e-6)

    pos_above = torch.tensor([1.0, 0.0, 0.0])
    loss_above = strategy.compute_loss(relation, pos_above, _DUMMY_BBOX)
    assert loss_above > 0.0


def test_position_limits_loss_scales_with_weight():
    """weight=2.0 gives exactly 2x the loss of weight=1.0."""

    relation_1x = PositionLimitsBox(x_min=-1.0, x_max=1.0, relation_loss_weight=1.0)
    relation_2x = PositionLimitsBox(x_min=-1.0, x_max=1.0, relation_loss_weight=2.0)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)
    child_pos = torch.tensor([3.0, 0.0, 0.0])

    loss_1x = strategy.compute_loss(relation_1x, child_pos, _DUMMY_BBOX)
    loss_2x = strategy.compute_loss(relation_2x, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss_2x, 2.0 * loss_1x, rtol=1e-5)


def test_position_limits_z_constraint():
    """Z bounds are enforced like X and Y bounds."""

    relation = PositionLimitsBox(z_min=0.0, z_max=1.0)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)

    pos_inside = torch.tensor([0.0, 0.0, 0.5])
    loss_inside = strategy.compute_loss(relation, pos_inside, _DUMMY_BBOX)
    assert torch.isclose(loss_inside, torch.tensor(0.0), atol=1e-6)

    pos_outside = torch.tensor([0.0, 0.0, 2.0])
    loss_outside = strategy.compute_loss(relation, pos_outside, _DUMMY_BBOX)
    assert loss_outside > 0.0


def test_position_limits_unconstrained_axes_ignored():
    """Only X constrained: extreme Y/Z values produce no loss."""

    relation = PositionLimitsBox(x_min=-1.0, x_max=1.0)
    strategy = PositionLimitsBoxLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.0, 1e6, -1e6])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_position_limits_cylindrical_zero_loss_when_inside():
    """Loss is zero when the position is inside the radial bounds."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_min=0.5, radius_max=1.0)
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.75, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_position_limits_cylindrical_positive_loss_when_above_max():
    """Loss is positive when the position exceeds radius_max."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_min=0.5, radius_max=1.0)
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)
    child_pos = torch.tensor([1.25, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert loss > 0.0


def test_position_limits_cylindrical_positive_loss_when_below_min():
    """Loss is positive when the position is below radius_min."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_min=0.5, radius_max=1.0)
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)
    child_pos = torch.tensor([0.25, 0.0, 0.0])

    loss = strategy.compute_loss(relation, child_pos, _DUMMY_BBOX)
    assert loss > 0.0


def test_position_limits_cylindrical_single_bound_min_only():
    """With only radius_min set: zero loss above the bound, positive loss below."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_min=0.5)
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)

    pos_above = torch.tensor([0.75, 0.0, 0.0])
    loss_above = strategy.compute_loss(relation, pos_above, _DUMMY_BBOX)
    assert torch.isclose(loss_above, torch.tensor(0.0), atol=1e-6)

    pos_below = torch.tensor([0.25, 0.0, 0.0])
    loss_below = strategy.compute_loss(relation, pos_below, _DUMMY_BBOX)
    assert loss_below > 0.0


def test_position_limits_cylindrical_single_bound_max_only():
    """With only radius_max set: zero loss below the bound, positive loss above."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_max=0.5)
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)

    pos_below = torch.tensor([0.25, 0.0, 0.0])
    loss_below = strategy.compute_loss(relation, pos_below, _DUMMY_BBOX)
    assert torch.isclose(loss_below, torch.tensor(0.0), atol=1e-6)

    pos_above = torch.tensor([0.75, 0.0, 0.0])
    loss_above = strategy.compute_loss(relation, pos_above, _DUMMY_BBOX)
    assert loss_above > 0.0


def test_position_limits_cylindrical_loss_scales_with_weight():
    """weight=2.0 gives exactly 2x the loss of weight=1.0."""

    relation_1x = PositionLimitsCylindrical(
        center_x=0.0, center_y=0.0, radius_min=0.5, radius_max=1.0, relation_loss_weight=1.0
    )
    relation_2x = PositionLimitsCylindrical(
        center_x=0.0, center_y=0.0, radius_min=0.5, radius_max=1.0, relation_loss_weight=2.0
    )
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)
    child_pos = torch.tensor([1.25, 0.0, 0.0])

    loss_1x = strategy.compute_loss(relation_1x, child_pos, _DUMMY_BBOX)
    loss_2x = strategy.compute_loss(relation_2x, child_pos, _DUMMY_BBOX)
    assert torch.isclose(loss_2x, 2.0 * loss_1x, rtol=1e-5)


def test_position_limits_cylindrical_unconstrained_z_ignored():
    """Extreme Z values do not affect cylindrical position-limit loss."""

    relation = PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_max=1.0)
    strategy = PositionLimitsCylindricalLossStrategy(slope=10.0)

    loss_near_z = strategy.compute_loss(relation, torch.tensor([0.5, 0.0, 0.0]), _DUMMY_BBOX)
    loss_far_z = strategy.compute_loss(relation, torch.tensor([0.5, 0.0, 1e6]), _DUMMY_BBOX)
    assert torch.isclose(loss_near_z, loss_far_z, atol=1e-6)


# =============================================================================
# Solver integration test
# =============================================================================


from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.pose import Pose


def test_solver_respects_position_limits():
    """Solver moves an object inside the PositionLimitsBox region."""
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
    box.add_relation(PositionLimitsBox(x_min=0.2, x_max=0.5, y_min=0.2, y_max=0.5))

    initial_positions = [{table: (0.0, 0.0, 0.0), box: (1.5, 1.5, 0.11)}]

    solver = RelationSolver(params=RelationSolverParams(max_iters=300, convergence_threshold=1e-4, verbose=False))
    result = solver.solve(objects=[table, box], initial_positions=initial_positions)

    pos = result[0][box]
    assert 0.2 <= pos[0] <= 0.5, f"x={pos[0]} should be within [0.2, 0.5]"
    assert 0.2 <= pos[1] <= 0.5, f"y={pos[1]} should be within [0.2, 0.5]"


def test_solver_respects_cylindrical_position_limits():
    """Solver moves an object from an annulus center into its allowed radial band."""
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 2.0, 0.1)),
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box = DummyObject(name="box", bounding_box=_DUMMY_BBOX)
    box.add_relation(On(table, clearance_m=0.01))
    box.add_relation(PositionLimitsCylindrical(center_x=0.5, center_y=0.5, radius_min=0.25, radius_max=0.5))

    result = RelationSolver(params=RelationSolverParams(max_iters=300, convergence_threshold=1e-4)).solve(
        objects=[table, box], initial_positions=[{table: (0.0, 0.0, 0.0), box: (0.5, 0.5, 0.11)}]
    )
    x, y, _ = result[0][box]
    radius = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
    assert 0.249 <= radius <= 0.501
