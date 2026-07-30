# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RelationSolver built-in no-overlap loss."""

import math
import torch

from isaaclab_arena.relations.loss_primitives import interval_overlap_axis_loss
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.tests.dummy_object import DummyObject
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


def test_solver_uses_rotated_bbox_for_collision():
    """Test that a yaw-rotated env bbox passed to solve() changes the no-overlap loss (solver consumes it)."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    # Long box (spans X) and a cube offset in +Y. No On/NextTo relations, so the only loss is
    # the built-in no-overlap loss between the two non-anchors.
    long_box = DummyObject(
        name="long_box",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.3, -0.05, -0.05), max_point=(0.3, 0.05, 0.05)),
    )
    cube = DummyObject(
        name="cube",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.05, -0.05, -0.05), max_point=(0.05, 0.05, 0.05)),
    )
    objects = [table, long_box, cube]
    # Boxes sit high above the table (z=0.5) so neither collides with the table.
    initial = [{table: (0.0, 0.0, 0.0), long_box: (0.0, 0.0, 0.5), cube: (0.0, 0.2, 0.5)}]

    # max_iters=0: solve() only computes the initial loss and stores last_loss_per_env.
    solver = RelationSolver(params=RelationSolverParams(max_iters=0, verbose=False))

    solver.solve(objects, initial)
    assert solver.last_loss_per_env is not None
    loss_unrotated = solver.last_loss_per_env[0].item()

    # Hand the solver a 90° conservative bbox for the long box via the env_bboxes channel.
    rotated = {long_box: long_box.get_bounding_box().rotated_around_z(math.pi / 2)}
    solver.solve(objects, initial, env_bboxes=rotated)
    assert solver.last_loss_per_env is not None
    loss_rotated = solver.last_loss_per_env[0].item()

    # Unrotated: long box spans Y [-0.05, 0.05], clear of the cube at Y=0.2 -> no overlap.
    assert loss_unrotated == 0.0
    # Rotated 90°: long box now spans Y [-0.3, 0.3], overlapping the cube -> positive loss.
    assert loss_rotated > 0.0


def _single_pair_no_overlap_loss(
    slope: float,
    clearance_m: float,
    child_pos: torch.Tensor,
    child_bbox: AxisAlignedBoundingBox,
    parent_world_bbox: AxisAlignedBoundingBox,
) -> torch.Tensor:
    """Single-pair no-overlap loss; the reference the vectorized solver path must reproduce."""
    single_input = child_pos.dim() == 1
    if single_input:
        child_pos = child_pos.unsqueeze(0)

    c = clearance_m
    parent_x_min = parent_world_bbox.min_point[:, 0] - c
    parent_x_max = parent_world_bbox.max_point[:, 0] + c
    parent_y_min = parent_world_bbox.min_point[:, 1] - c
    parent_y_max = parent_world_bbox.max_point[:, 1] + c
    parent_z_min = parent_world_bbox.min_point[:, 2] - c
    parent_z_max = parent_world_bbox.max_point[:, 2] + c

    child_world_min = child_pos + child_bbox.min_point
    child_world_max = child_pos + child_bbox.max_point

    overlap_x = interval_overlap_axis_loss(child_world_min[:, 0], child_world_max[:, 0], parent_x_min, parent_x_max)
    overlap_y = interval_overlap_axis_loss(child_world_min[:, 1], child_world_max[:, 1], parent_y_min, parent_y_max)
    overlap_z = interval_overlap_axis_loss(child_world_min[:, 2], child_world_max[:, 2], parent_z_min, parent_z_max)

    total_loss = slope * (overlap_x * overlap_y * overlap_z)
    return total_loss.squeeze(0) if single_input else total_loss


# =============================================================================
# Single-pair no-overlap reference tests
# =============================================================================


def test_no_collision_zero_loss_when_fully_separated():
    """Test that NoCollision loss is zero when AABBs do not overlap on any axis."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")

    # Child at origin -> world X [0, 0.2]. Parent at x=1 -> world X [1, 1.2]. No X overlap => volume 0.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((1.0, 0.0, 0.0))

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_no_collision_zero_loss_when_separated_on_one_axis_only():
    """Test that NoCollision loss is zero when separated on one axis (overlap_x=0 => volume=0)."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")

    # Child X [0, 0.2], parent X [0.5, 0.7] -> no X overlap. Y and Z overlapping.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.5, 0.0, 0.0))

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_no_collision_zero_loss_when_just_touching():
    """Test that NoCollision loss is zero when intervals just touch (clearance_m=0)."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")

    # Child X [0, 0.2], parent X [0.2, 0.4]. Just touching.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.2, 0.0, 0.0))

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.0, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_no_collision_positive_loss_when_just_touching():
    """Test that NoCollision loss is positive when just touching with default clearance."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")

    # Child X [0, 0.2], parent X [0.2, 0.4]. Just touching; default clearance expands parent so overlap > 0.
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.2, 0.0, 0.0))

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert loss > 0.0


def test_no_collision_positive_loss_when_3d_overlap():
    """Test that NoCollision loss is positive when AABBs overlap in all three axes."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")

    # Child at (0.1, 0.1, 0), parent at (0.05, 0.05, 0) -> overlap in X, Y, Z.
    child_pos = torch.tensor([0.1, 0.1, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.05, 0.05, 0.0))

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert loss > 0.0


def test_no_collision_loss_scales_with_slope():
    """Test that NoCollision loss scales with slope (loss = slope * overlap_volume)."""
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")

    child_pos = torch.tensor([0.1, 0.1, 0.0])
    parent_world_bbox = box_b.get_bounding_box().translated((0.05, 0.05, 0.0))

    loss_10 = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    loss_20 = _single_pair_no_overlap_loss(
        20.0, clearance_m=0.01, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert torch.isclose(loss_20, 2.0 * loss_10, rtol=1e-5)


def test_no_collision_loss_volume_formula():
    """Test that NoCollision loss equals slope * overlap volume for known overlap (clearance_m=0)."""
    box_a = _create_box("box_a", size=0.2)
    box_b = _create_box("box_b", size=0.2)

    child_pos = torch.tensor([0.1, 0.1, 0.1])
    parent_world_bbox = box_b.get_bounding_box().translated((0.15, 0.15, 0.15))
    # Overlap [0.15, 0.3]^3, volume 0.15^3. Expected loss = 10 * 0.15^3.
    expected_loss = 10.0 * (0.15**3)

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.0, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
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


def test_no_overlap_skips_direct_on_non_anchor_pair():
    """No-overlap should not fight an On relation whose parent is also movable."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())

    support = _create_box("support")
    child = _create_box("child")
    support.add_relation(On(table, clearance_m=0.0))
    child.add_relation(On(support, clearance_m=0.0))

    objects = [table, support, child]
    initial_positions = [{
        table: (0.0, 0.0, 0.0),
        support: (2.0, 2.0, 0.0),
        child: (2.0, 2.0, 0.0),
    }]
    state = RelationSolverState(objects, initial_positions, device=torch.device("cpu"))
    solver = RelationSolver(params=RelationSolverParams(max_iters=0))

    loss = solver._compute_no_overlap_loss(state)  # pyright: ignore[reportPrivateUsage]

    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_negative_clearance_m_raises():
    """Negative clearance_m should be rejected."""
    import pytest

    with pytest.raises(AssertionError):
        RelationSolverParams(clearance_m=-0.01)


def test_validation_accepts_on_parent_overlap():
    """Non-anchor sitting On(anchor) should pass validation (On pairs are skipped)."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import NoOverlapValidator

    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box = _create_box("box")
    box.add_relation(On(table, clearance_m=0.01))

    # Box at z=0.11 sits on top of table (bbox 0..0.1). With clearance_m=0.01
    # the expanded table extends to 0.11, so they just touch. The On pair
    # should be skipped by validation.
    positions = {table: (0.0, 0.0, 0.0), box: (0.4, 0.4, 0.11)}
    env_bboxes = {obj: obj.get_bounding_box() for obj in positions}

    validator = NoOverlapValidator(ObjectPlacerParams())
    assert validator._validate_no_overlap(positions, env_bboxes)


def test_validation_rejects_non_anchor_overlap():
    """Two overlapping non-anchor boxes should fail validation."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validators import NoOverlapValidator

    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")
    box_a.add_relation(On(table, clearance_m=0.01))
    box_b.add_relation(On(table, clearance_m=0.01))

    # Both boxes at nearly the same position -- they overlap
    positions = {table: (0.0, 0.0, 0.0), box_a: (0.3, 0.3, 0.11), box_b: (0.35, 0.35, 0.11)}
    env_bboxes = {obj: obj.get_bounding_box() for obj in positions}

    validator = NoOverlapValidator(ObjectPlacerParams())
    assert not validator._validate_no_overlap(positions, env_bboxes)


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

    child_pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
    parent_world_bbox = AxisAlignedBoundingBox(
        min_point=torch.tensor([[1.0, 0.0, 0.0], [0.05, 0.05, 0.0]]),
        max_point=torch.tensor([[1.2, 0.2, 0.2], [0.25, 0.25, 0.2]]),
    )

    loss = _single_pair_no_overlap_loss(
        10.0, clearance_m=0.0, child_pos=child_pos, child_bbox=box_a.bounding_box, parent_world_bbox=parent_world_bbox
    )
    assert loss.shape == (2,)
    assert torch.isclose(loss[0], torch.tensor(0.0), atol=1e-5)
    assert loss[1] > 0.0


def _reference_pairwise_no_overlap_loss(solver: RelationSolver, state: RelationSolverState) -> torch.Tensor:
    """Per-pair reference implementation; the correctness oracle for the vectorized path."""
    device = state.device
    total = torch.zeros(state.batch_size, device=device, dtype=torch.float32)
    non_anchors = state.optimizable_objects
    anchors = list(state.anchor_objects)
    clearance = solver.params.clearance_m
    slope = solver._no_collision_strategy.slope  # pyright: ignore[reportPrivateUsage]

    on_pairs: set[tuple[int, int]] = set()
    for obj in [*non_anchors, *anchors]:
        for rel in obj.get_relations():
            if isinstance(rel, On):
                on_pairs.add((id(obj), id(rel.parent)))
                on_pairs.add((id(rel.parent), id(obj)))

    for i, child in enumerate(non_anchors):
        child_pos = state.get_position(child)
        child_bbox = state.get_bbox(child)
        for anchor in anchors:
            if (id(child), id(anchor)) in on_pairs:
                continue
            # Re-reads anchor.get_world_bounding_box() per pair; agrees with production's
            # once-per-solve cache because DummyObject bounding boxes are immutable.
            total = total + _single_pair_no_overlap_loss(
                slope,
                clearance_m=clearance,
                child_pos=child_pos,
                child_bbox=child_bbox,
                parent_world_bbox=anchor.get_world_bounding_box().to(device),
            )
        for j in range(i + 1, len(non_anchors)):
            other = non_anchors[j]
            if (id(child), id(other)) in on_pairs:
                continue
            other_pos = state.get_position(other)
            other_bbox = state.get_bbox(other)
            total = total + _single_pair_no_overlap_loss(
                slope,
                clearance_m=clearance,
                child_pos=child_pos,
                child_bbox=child_bbox,
                parent_world_bbox=other_bbox.translated(other_pos.detach()),
            )
            total = total + _single_pair_no_overlap_loss(
                slope,
                clearance_m=clearance,
                child_pos=other_pos,
                child_bbox=other_bbox,
                parent_world_bbox=child_bbox.translated(child_pos.detach()),
            )
    return total


def _assert_vectorized_matches_reference(
    objects, initial_positions, expect_positive: bool, expected_pair_count: int | None = None
) -> None:
    """Vectorized no-overlap loss and its gradient must match the per-pair oracle (CPU)."""
    solver = RelationSolver(params=RelationSolverParams(verbose=False))

    state_new = RelationSolverState(objects, initial_positions, device=torch.device("cpu"))
    loss_new = solver._compute_no_overlap_loss(state_new)  # pyright: ignore[reportPrivateUsage]
    loss_new.sum().backward()
    grad_new = state_new.optimizable_positions.grad.clone()

    state_ref = RelationSolverState(objects, initial_positions, device=torch.device("cpu"))
    loss_ref = _reference_pairwise_no_overlap_loss(solver, state_ref)
    loss_ref.sum().backward()
    grad_ref = state_ref.optimizable_positions.grad.clone()

    assert torch.allclose(loss_new, loss_ref, atol=1e-6), f"loss mismatch: {loss_new} vs {loss_ref}"
    assert torch.allclose(grad_new, grad_ref, atol=1e-6), f"gradient mismatch:\n{grad_new}\nvs\n{grad_ref}"
    if expect_positive:
        assert (loss_new > 0).any(), "scene should produce a positive loss to exercise the overlap branch"
    if expected_pair_count is not None:
        pair_count = solver._last_no_overlap_pair_count  # pyright: ignore[reportPrivateUsage]
        assert pair_count == expected_pair_count, f"expected {expected_pair_count} pairs, got {pair_count}"


def test_vectorized_no_overlap_matches_reference_non_anchor_pairs():
    """Vectorized loss/gradient match the per-pair oracle for overlapping non-anchor boxes."""
    table, box_a, box_b = _create_no_collision_scene()  # box_a/box_b On(table); anchor pairs are skipped
    objects = [table, box_a, box_b]
    initial_positions = [
        {table: (0.0, 0.0, 0.0), box_a: (0.30, 0.30, 0.11), box_b: (0.35, 0.35, 0.11)},
        {table: (0.0, 0.0, 0.0), box_a: (0.50, 0.50, 0.11), box_b: (0.90, 0.90, 0.11)},
    ]
    # Both boxes On(table) -> anchor pairs skipped; only the box_a/box_b pair, both directions.
    _assert_vectorized_matches_reference(objects, initial_positions, expect_positive=True, expected_pair_count=2)


def test_vectorized_no_overlap_matches_reference_anchor_pairs():
    """Free (non-On) boxes overlapping the anchor exercise the anchor-pair + expand branch."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")  # no On relation: child-vs-anchor pairs are active
    objects = [table, box_a, box_b]
    # Boxes sit low enough to overlap the table slab (z in [0, 0.1]) and each other.
    initial_positions = [
        {table: (0.0, 0.0, 0.0), box_a: (0.20, 0.20, 0.05), box_b: (0.25, 0.25, 0.05)},
        {table: (0.0, 0.0, 0.0), box_a: (0.40, 0.40, 0.05), box_b: (0.42, 0.42, 0.05)},
    ]
    # 2 boxes x 1 anchor = 2 anchor pairs + box_a/box_b pair both directions = 4 pairs.
    _assert_vectorized_matches_reference(objects, initial_positions, expect_positive=True, expected_pair_count=4)


def test_vectorized_no_overlap_matches_reference_single_non_anchor():
    """One free box vs one anchor exercises the anchor-pair path with no non-anchor pair loop."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box = _create_box("box")  # free (non-On): the single child-vs-anchor pair is active
    objects = [table, box]
    # Box overlaps the table slab (z in [0, 0.1]).
    initial_positions = [{table: (0.0, 0.0, 0.0), box: (0.20, 0.20, 0.05)}]
    # 1 box x 1 anchor = 1 anchor pair; no second non-anchor, so no non-anchor pairs.
    _assert_vectorized_matches_reference(objects, initial_positions, expect_positive=True, expected_pair_count=1)


def test_vectorized_no_overlap_matches_reference_multiple_anchors():
    """Two anchors double the anchor-pair loop; vectorized loss/gradient/count match the oracle."""
    table_a = _create_table()
    table_a.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table_a.add_relation(IsAnchor())
    table_b = _create_table()
    table_b.set_initial_pose(Pose(position_xyz=(3.0, 3.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table_b.add_relation(IsAnchor())
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")  # free (non-On): every child-vs-anchor pair is active
    objects = [table_a, table_b, box_a, box_b]
    # Boxes overlap table_a (slab z in [0, 0.1]) and each other; table_b is far away (zero pair).
    initial_positions = [
        {table_a: (0.0, 0.0, 0.0), table_b: (3.0, 3.0, 0.0), box_a: (0.20, 0.20, 0.05), box_b: (0.25, 0.25, 0.05)},
        {table_a: (0.0, 0.0, 0.0), table_b: (3.0, 3.0, 0.0), box_a: (0.40, 0.40, 0.05), box_b: (0.42, 0.42, 0.05)},
    ]
    # 2 boxes x 2 anchors = 4 anchor pairs + box_a/box_b pair both directions = 6 pairs.
    _assert_vectorized_matches_reference(objects, initial_positions, expect_positive=True, expected_pair_count=6)


def test_vectorized_no_overlap_empty_pairs_returns_zero():
    """A lone non-anchor related only On the anchor leaves no scored pairs -> zero loss."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box = _create_box("box")
    box.add_relation(On(table, clearance_m=0.01))  # the only pair (box, table) is On -> skipped
    objects = [table, box]
    initial_positions = [{table: (0.0, 0.0, 0.0), box: (0.3, 0.3, 0.11)}]

    solver = RelationSolver(params=RelationSolverParams(verbose=False))
    state = RelationSolverState(objects, initial_positions, device=torch.device("cpu"))
    loss = solver._compute_no_overlap_loss(state)  # pyright: ignore[reportPrivateUsage]

    assert solver._last_no_overlap_pair_count == 0  # pyright: ignore[reportPrivateUsage]
    assert loss.shape == (1,)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_vectorized_no_overlap_is_per_env_independent():
    """Per-env losses stay independent: one env overlaps, the other is clear (no batch-axis collapse)."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    box_a = _create_box("box_a")
    box_b = _create_box("box_b")  # free (non-On): child-vs-anchor pairs are active
    objects = [table, box_a, box_b]
    initial_positions = [
        {table: (0.0, 0.0, 0.0), box_a: (0.20, 0.20, 0.05), box_b: (0.22, 0.22, 0.05)},  # overlapping
        {table: (0.0, 0.0, 0.0), box_a: (0.20, 0.20, 5.0), box_b: (5.0, 5.0, 5.0)},  # well separated
    ]

    solver = RelationSolver(params=RelationSolverParams(verbose=False))
    state = RelationSolverState(objects, initial_positions, device=torch.device("cpu"))
    loss = solver._compute_no_overlap_loss(state)  # pyright: ignore[reportPrivateUsage]

    assert loss[0] > 0
    assert torch.allclose(loss[1], torch.zeros_like(loss[1]))


def test_vectorized_no_overlap_debug_prints_breakdown(capsys):
    """debug=True prints a per-pair [NoOverlap] breakdown."""
    table, box_a, box_b = _create_no_collision_scene()
    objects = [table, box_a, box_b]
    initial_positions = [{table: (0.0, 0.0, 0.0), box_a: (0.20, 0.20, 0.11), box_b: (0.22, 0.22, 0.11)}]

    solver = RelationSolver(params=RelationSolverParams(verbose=False))
    state = RelationSolverState(objects, initial_positions, device=torch.device("cpu"))
    solver._compute_no_overlap_loss(state, debug=True)  # pyright: ignore[reportPrivateUsage]

    assert "[NoOverlap]" in capsys.readouterr().out


def test_solver_profile_prints_timing(capsys):
    """profile=True prints a timing line (covers the ms/iter division) without error."""
    table, box_a, box_b = _create_no_collision_scene()
    objects = [table, box_a, box_b]
    initial_positions = [{table: (0.0, 0.0, 0.0), box_a: (0.2, 0.2, 0.11), box_b: (0.25, 0.25, 0.11)}]

    solver = RelationSolver(params=RelationSolverParams(max_iters=5, verbose=False, profile=True))
    solver.solve(objects=objects, initial_positions=initial_positions)

    out = capsys.readouterr().out
    assert "[RelationSolver] solve:" in out
    assert "ms/iter" in out
    assert "batch=" in out
    assert "no-overlap pairs=" in out


def test_solver_profile_zero_iters_does_not_raise():
    """profile=True with max_iters=0 must skip the ms/iter division (empty loss_history guard)."""
    table, box_a, box_b = _create_no_collision_scene()
    objects = [table, box_a, box_b]
    initial_positions = [{table: (0.0, 0.0, 0.0), box_a: (0.2, 0.2, 0.11), box_b: (0.25, 0.25, 0.11)}]

    solver = RelationSolver(params=RelationSolverParams(max_iters=0, verbose=False, profile=True))
    solver.solve(objects=objects, initial_positions=initial_positions)  # must not raise ZeroDivisionError


def test_compute_loss_batched_direct():
    """compute_loss_batched returns (num_pairs, batch_size) loss for known world-space extents."""
    strategy = NoCollisionLossStrategy(slope=10.0)
    # Two pairs, batch_size=1. Pair 0 overlaps by 0.1 on each axis; pair 1 is fully separated.
    subject_min = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    subject_max = torch.tensor([[[0.2, 0.2, 0.2]], [[0.2, 0.2, 0.2]]])
    obstacle_min = torch.tensor([[[0.1, 0.1, 0.1]], [[1.0, 1.0, 1.0]]])
    obstacle_max = torch.tensor([[[0.3, 0.3, 0.3]], [[1.2, 1.2, 1.2]]])

    loss = strategy.compute_loss_batched(0.0, subject_min, subject_max, obstacle_min, obstacle_max)

    assert loss.shape == (2, 1)
    assert torch.isclose(loss[0, 0], torch.tensor(10.0 * 0.1**3), rtol=1e-4)  # slope * overlap volume
    assert torch.isclose(loss[1, 0], torch.tensor(0.0), atol=1e-6)


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
