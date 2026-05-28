# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for NotOn / NotNextTo relations and their loss strategies."""

import torch

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NotNextToLossStrategy, NotOnLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NotNextTo, NotOn, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _table():
    """1m x 1m x 0.1m table at the local origin (used as anchor at world (0, 0, 0))."""
    return DummyObject(
        name="table", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1))
    )


def _box():
    """0.2m cube."""
    return DummyObject(
        name="box", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2))
    )


# =============================================================================
# NotOnLossStrategy
# =============================================================================


@pytest.mark.parametrize("child_pos", [(2.0, 0.4, 0.5), (0.4, 2.0, 0.5)], ids=["outside_x", "outside_y"])
def test_not_on_zero_when_child_outside_parent(child_pos):
    """A single-axis escape (X or Y) is enough to drive NotOn loss to zero."""
    loss = NotOnLossStrategy(slope=100.0).compute_loss(
        NotOn(_table()), torch.tensor(child_pos), _box().bounding_box, _table().bounding_box
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_not_on_positive_when_child_centered_on_parent():
    """At the center of the parent's footprint the loss is slope * min(inside_x, inside_y)."""
    loss = NotOnLossStrategy(slope=100.0).compute_loss(
        NotOn(_table()), torch.tensor([0.4, 0.4, 0.5]), _box().bounding_box, _table().bounding_box
    )
    # valid_x in [0, 0.8], child_x=0.4 -> inside_x = 0.4; symmetric on Y. loss = 100 * 0.4 = 40.
    assert torch.isclose(loss, torch.tensor(40.0), atol=1e-4)


def test_not_on_gradient_pushes_outward():
    """The gradient at an interior point pushes the child toward the nearer edge of the binding axis."""
    child_pos = torch.tensor([0.3, 0.4, 0.5], requires_grad=True)
    # X is the binding axis (inside_x=0.3 < inside_y=0.4); descent direction is -X.
    NotOnLossStrategy(slope=100.0).compute_loss(
        NotOn(_table()), child_pos, _box().bounding_box, _table().bounding_box
    ).backward()
    assert child_pos.grad[0].item() > 0
    assert torch.isclose(child_pos.grad[1], torch.tensor(0.0), atol=1e-6)


def test_not_on_margin_extends_penalty_past_parent_edge():
    """margin_m widens the penalized region beyond the parent's footprint."""
    child_pos = torch.tensor([0.85, 0.4, 0.5])  # 5 cm past parent's X edge
    loss_no_margin = NotOnLossStrategy(slope=100.0, margin_m=0.0).compute_loss(
        NotOn(_table()), child_pos, _box().bounding_box, _table().bounding_box
    )
    loss_with_margin = NotOnLossStrategy(slope=100.0, margin_m=0.1).compute_loss(
        NotOn(_table()), child_pos, _box().bounding_box, _table().bounding_box
    )
    assert torch.isclose(loss_no_margin, torch.tensor(0.0), atol=1e-6)
    assert loss_with_margin > 0.0


# =============================================================================
# NotNextToLossStrategy
# =============================================================================


def test_not_next_to_positive_at_target_position():
    """Loss is positive when the child sits at the NextTo zone's center (all escapes = 0)."""
    relation = NotNextTo(_table(), side=Side.POSITIVE_X, distance_m=0.05)
    loss = NotNextToLossStrategy(slope=10.0, margin_m=0.05).compute_loss(
        relation, torch.tensor([1.05, 0.4, 0.5]), _box().bounding_box, _table().bounding_box
    )
    # All three escapes = 0, all gaps = margin = 0.05. loss = slope * min(0.05, 0.05, 0.05) = 0.5.
    assert torch.isclose(loss, torch.tensor(0.5), atol=1e-4)


@pytest.mark.parametrize(
    "child_pos,reason",
    [
        ((2.0, 0.4, 0.5), "far past target"),
        ((0.3, 0.4, 0.5), "wrong side"),
        ((1.05, 5.0, 0.5), "outside cross band"),
    ],
)
def test_not_next_to_zero_when_any_escape_past_margin(child_pos, reason):
    """Any single escape (distance / half-plane / cross-band) past margin -> loss = 0."""
    relation = NotNextTo(_table(), side=Side.POSITIVE_X, distance_m=0.05)
    loss = NotNextToLossStrategy(slope=10.0, margin_m=0.05).compute_loss(
        relation, torch.tensor(child_pos), _box().bounding_box, _table().bounding_box
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected zero when {reason}"


@pytest.mark.parametrize(
    "side,target",
    [
        (Side.POSITIVE_X, (1.05, 0.4, 0.5)),
        (Side.NEGATIVE_X, (-0.25, 0.4, 0.5)),
        (Side.POSITIVE_Y, (0.4, 1.05, 0.5)),
        (Side.NEGATIVE_Y, (0.4, -0.25, 0.5)),
    ],
)
def test_not_next_to_each_side_variant_is_positive_at_its_target(side, target):
    """All four Side variants produce a positive loss at their respective NextTo target."""
    relation = NotNextTo(_table(), side=side, distance_m=0.05)
    loss = NotNextToLossStrategy(slope=10.0, margin_m=0.05).compute_loss(
        relation, torch.tensor(target), _box().bounding_box, _table().bounding_box
    )
    assert loss > 0.0


# =============================================================================
# Solver integration — proves the strategy is wired through end-to-end.
# =============================================================================


def test_solver_drives_mug_off_forbidden_table():
    """Mug must sit on left_table but NOT on right_table. Starts on right, ends on left."""
    left, right = _table(), _table()
    mug = DummyObject(
        name="mug", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.12))
    )
    left.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    right.set_initial_pose(Pose(position_xyz=(1.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    left.add_relation(IsAnchor())
    right.add_relation(IsAnchor())
    mug.add_relation(On(left))
    mug.add_relation(NotOn(right))

    initial = {left: (0.0, 0.0, 0.0), right: (1.5, 0.0, 0.0), mug: (1.9, 0.4, 0.13)}
    solver = RelationSolver(params=RelationSolverParams(verbose=False, save_position_history=False, max_iters=400))
    final = solver.solve(objects=[left, right, mug], initial_positions=[initial])[0]

    mug_x, mug_y, _ = final[mug]
    assert 0.0 <= mug_x <= 1.0 and 0.0 <= mug_y <= 1.0, f"Mug should be on the left table; got {final[mug]}"
    assert not (1.5 <= mug_x <= 2.5), f"Mug should NOT be on the right table; got {final[mug]}"
