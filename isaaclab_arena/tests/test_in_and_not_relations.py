# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for In and Not relation types and their loss strategies."""

import torch

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import InLossStrategy, NotRelationLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import In, IsAnchor, Not, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


# =============================================================================
# Helpers
# =============================================================================


def _create_box(name: str = "box", size: float = 0.1) -> DummyObject:
    """Create a small box (local bbox centered at origin)."""
    half = size / 2.0
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(-half, -half, -half), max_point=(half, half, half)),
    )


def _create_bowl(name: str = "bowl") -> DummyObject:
    """Create a bowl-like container (wide XY, short Z)."""
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.15, -0.15, 0.0), max_point=(0.15, 0.15, 0.1)),
    )


def _create_table() -> DummyObject:
    """Create a table-like object (wide, thin)."""
    return DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.5, -0.5, 0.0), max_point=(0.5, 0.5, 0.05)),
    )


# =============================================================================
# InLossStrategy unit tests
# =============================================================================


def test_in_loss_zero_when_child_inside_parent():
    """In loss should be ~0 when child is centered inside parent's XY footprint."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    strategy = InLossStrategy(slope=100.0)

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([[0.5, 0.5, 0.92]])

    loss = strategy.compute_loss(
        relation=In(bowl),
        child_pos=child_pos,
        child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss.shape == (1,)
    assert loss[0].item() < 0.5, f"Expected near-zero loss for centered child, got {loss[0].item()}"


def test_in_loss_positive_when_child_outside_parent():
    """In loss should be > 0 when child is far outside parent's XY footprint."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    strategy = InLossStrategy(slope=100.0)

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([[2.0, 2.0, 0.92]])

    loss = strategy.compute_loss(
        relation=In(bowl),
        child_pos=child_pos,
        child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss[0].item() > 10.0, f"Expected significant loss for child far outside, got {loss[0].item()}"


def test_in_loss_z_is_soft():
    """In loss Z component should be weaker than XY (z_slope_ratio < 1)."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    strategy = InLossStrategy(slope=100.0, z_slope_ratio=0.1)

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))

    child_at_target_z = torch.tensor([[0.5, 0.5, 0.92]])
    loss_at_target = strategy.compute_loss(
        relation=In(bowl), child_pos=child_at_target_z, child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )

    child_offset_z = torch.tensor([[0.5, 0.5, 1.2]])
    loss_offset_z = strategy.compute_loss(
        relation=In(bowl), child_pos=child_offset_z, child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )

    child_offset_xy = torch.tensor([[2.0, 0.5, 0.92]])
    loss_offset_xy = strategy.compute_loss(
        relation=In(bowl), child_pos=child_offset_xy, child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )

    z_contribution = loss_offset_z[0].item() - loss_at_target[0].item()
    xy_contribution = loss_offset_xy[0].item() - loss_at_target[0].item()
    assert z_contribution < xy_contribution, (
        f"Z loss ({z_contribution}) should be less than XY loss ({xy_contribution}) due to z_slope_ratio"
    )


# =============================================================================
# NotRelationLossStrategy unit tests
# =============================================================================


def test_not_loss_zero_when_child_outside_xy():
    """Not loss should be 0 when child is far outside parent's XY footprint."""
    child = _create_box("child", size=0.1)
    table = _create_table()
    not_strategy = NotRelationLossStrategy()

    table_world_bbox = table.get_bounding_box().translated((0.0, 0.0, 0.0))
    child_pos = torch.tensor([[3.0, 3.0, 0.0]])

    loss = not_strategy.compute_loss(
        relation=Not(On(table), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=child.get_bounding_box(),
        parent_world_bbox=table_world_bbox,
    )
    assert loss.shape == (1,)
    assert loss[0].item() < 1e-6, f"Expected ~0 loss when child is far outside XY, got {loss[0].item()}"


def test_not_loss_positive_when_child_overlaps_xy():
    """Not loss should be positive when child's XY footprint overlaps parent's."""
    child = _create_box("child", size=0.1)
    table = _create_table()
    not_strategy = NotRelationLossStrategy()

    table_world_bbox = table.get_bounding_box().translated((0.0, 0.0, 0.0))
    child_pos = torch.tensor([[0.0, 0.0, 0.11]])

    loss = not_strategy.compute_loss(
        relation=Not(On(table), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=child.get_bounding_box(),
        parent_world_bbox=table_world_bbox,
    )
    assert loss[0].item() > 0.1, f"Expected positive loss when child overlaps parent XY, got {loss[0].item()}"


def test_not_in_loss_zero_when_child_outside():
    """Not(In(bowl)) should have ~0 loss when child is far from bowl."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    not_strategy = NotRelationLossStrategy()

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([[2.0, 2.0, 0.8]])

    loss = not_strategy.compute_loss(
        relation=Not(In(bowl), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss[0].item() < 1e-6, f"Expected ~0 loss when child is far outside, got {loss[0].item()}"


def test_not_in_loss_positive_when_child_inside():
    """Not(In(bowl)) should have positive loss when child is centered in bowl."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    not_strategy = NotRelationLossStrategy()

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([[0.5, 0.5, 0.92]])

    loss = not_strategy.compute_loss(
        relation=Not(In(bowl), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss[0].item() > 0.1, f"Expected positive loss when child is inside bowl, got {loss[0].item()}"


def test_not_margin_m_controls_repulsion_width():
    """Larger margin_m should expand the repulsion zone."""
    child = _create_box("child", size=0.1)
    table = _create_table()
    not_strategy = NotRelationLossStrategy()

    table_world_bbox = table.get_bounding_box().translated((0.0, 0.0, 0.0))
    # Place child just outside the table's XY edge (table goes to 0.5, child edge to 0.55)
    child_pos = torch.tensor([[0.5, 0.0, 0.11]])

    loss_small_margin = not_strategy.compute_loss(
        relation=Not(On(table), margin_m=0.01),
        child_pos=child_pos,
        child_bbox=child.get_bounding_box(),
        parent_world_bbox=table_world_bbox,
    )
    loss_large_margin = not_strategy.compute_loss(
        relation=Not(On(table), margin_m=0.20),
        child_pos=child_pos,
        child_bbox=child.get_bounding_box(),
        parent_world_bbox=table_world_bbox,
    )
    assert loss_large_margin[0].item() >= loss_small_margin[0].item(), (
        f"Larger margin should produce >= loss: small={loss_small_margin[0].item()}, "
        f"large={loss_large_margin[0].item()}"
    )


def test_not_loss_has_gradient():
    """Not loss should provide non-zero gradient when child overlaps parent XY."""
    child = _create_box("child", size=0.1)
    bowl = _create_bowl("bowl")
    not_strategy = NotRelationLossStrategy()

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([[0.52, 0.48, 0.92]], requires_grad=True)

    loss = not_strategy.compute_loss(
        relation=Not(In(bowl), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=child.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    loss.sum().backward()
    assert child_pos.grad is not None, "Should have gradient"
    xy_grad_norm = (child_pos.grad[0, 0] ** 2 + child_pos.grad[0, 1] ** 2) ** 0.5
    assert xy_grad_norm > 0.0, f"XY gradient should be non-zero, got norm={xy_grad_norm}"


# =============================================================================
# Single-env (3,) input shape tests
# =============================================================================


def test_in_loss_single_env_input():
    """In loss should work with single-env (3,) input (backward compat)."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    strategy = InLossStrategy(slope=100.0)

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([0.5, 0.5, 0.92])

    loss = strategy.compute_loss(
        relation=In(bowl), child_pos=child_pos, child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss.dim() == 0, f"Expected scalar loss for (3,) input, got shape {loss.shape}"


def test_not_loss_single_env_input():
    """Not loss should work with single-env (3,) input (backward compat)."""
    child = _create_box("child", size=0.1)
    table = _create_table()
    not_strategy = NotRelationLossStrategy()

    table_world_bbox = table.get_bounding_box().translated((0.0, 0.0, 0.0))
    child_pos = torch.tensor([3.0, 3.0, 0.0])

    loss = not_strategy.compute_loss(
        relation=Not(On(table), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=child.get_bounding_box(),
        parent_world_bbox=table_world_bbox,
    )
    assert loss.dim() == 0, f"Expected scalar loss for (3,) input, got shape {loss.shape}"


# =============================================================================
# Batched tests
# =============================================================================


def test_in_loss_batched():
    """In loss should work with batched positions (N, 3)."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    strategy = InLossStrategy(slope=100.0)

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([
        [0.5, 0.5, 0.92],
        [2.0, 2.0, 0.92],
    ])

    loss = strategy.compute_loss(
        relation=In(bowl), child_pos=child_pos, child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss.shape == (2,)
    assert loss[0].item() < loss[1].item(), "Child inside should have less loss than child outside"


def test_not_loss_batched():
    """Not loss should work with batched positions (N, 3)."""
    avocado = _create_box("avocado", size=0.05)
    bowl = _create_bowl("bowl")
    not_strategy = NotRelationLossStrategy()

    bowl_world_bbox = bowl.get_bounding_box().translated((0.5, 0.5, 0.8))
    child_pos = torch.tensor([
        [0.5, 0.5, 0.92],
        [2.0, 2.0, 0.92],
    ])

    loss = not_strategy.compute_loss(
        relation=Not(In(bowl), margin_m=0.05),
        child_pos=child_pos,
        child_bbox=avocado.get_bounding_box(),
        parent_world_bbox=bowl_world_bbox,
    )
    assert loss.shape == (2,)
    assert loss[0].item() > loss[1].item(), "Child inside bowl should have MORE Not loss than child outside"


# =============================================================================
# Full solver integration tests
# =============================================================================


def test_solver_not_in_separates_objects():
    """Solver with Not(In(B)) should push A outside B's footprint."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    bowl = _create_bowl("bowl")
    bowl.add_relation(On(table))

    avocado = _create_box("avocado", size=0.05)
    avocado.add_relation(On(table))
    avocado.add_relation(Not(In(bowl), margin_m=0.05))

    objects = [table, bowl, avocado]
    initial_positions = [{
        table: (0.0, 0.0, 0.0),
        bowl: (0.0, 0.0, 0.1),
        avocado: (0.02, 0.02, 0.1),
    }]

    solver = RelationSolver(
        params=RelationSolverParams(max_iters=600, verbose=False, save_position_history=False)
    )
    results = solver.solve(objects, initial_positions)

    avocado_pos = results[0][avocado]
    bowl_pos = results[0][bowl]

    xy_distance = ((avocado_pos[0] - bowl_pos[0]) ** 2 + (avocado_pos[1] - bowl_pos[1]) ** 2) ** 0.5
    assert xy_distance > 0.1, (
        f"Expected avocado to be pushed away from bowl, but XY distance is {xy_distance:.4f}m"
    )


def test_solver_in_and_not_in_coexist():
    """Solver should converge when both In and Not(In) are on different objects."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    bowl = _create_bowl("bowl")
    bowl.add_relation(On(table))

    obj_in = _create_box("obj_in", size=0.04)
    obj_in.add_relation(In(bowl))

    obj_out = _create_box("obj_out", size=0.04)
    obj_out.add_relation(On(table))
    obj_out.add_relation(Not(In(bowl), margin_m=0.05))

    objects = [table, bowl, obj_in, obj_out]
    initial_positions = [{
        table: (0.0, 0.0, 0.0),
        bowl: (0.0, 0.0, 0.1),
        obj_in: (0.02, 0.02, 0.15),
        obj_out: (-0.02, -0.02, 0.1),
    }]

    solver = RelationSolver(
        params=RelationSolverParams(max_iters=600, verbose=False, save_position_history=False)
    )
    results = solver.solve(objects, initial_positions)

    bowl_pos = results[0][bowl]
    obj_in_pos = results[0][obj_in]
    obj_out_pos = results[0][obj_out]

    in_xy_dist = ((obj_in_pos[0] - bowl_pos[0]) ** 2 + (obj_in_pos[1] - bowl_pos[1]) ** 2) ** 0.5
    out_xy_dist = ((obj_out_pos[0] - bowl_pos[0]) ** 2 + (obj_out_pos[1] - bowl_pos[1]) ** 2) ** 0.5

    assert in_xy_dist < out_xy_dist, (
        f"obj_in should be closer to bowl ({in_xy_dist:.4f}m) "
        f"than obj_out ({out_xy_dist:.4f}m)"
    )

    assert solver.last_loss_history[-1] < solver.last_loss_history[0], (
        "Solver should reduce loss over iterations"
    )
