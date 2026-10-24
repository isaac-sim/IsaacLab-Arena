# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RelationSolver built-in no-overlap loss."""

import math
import torch

import pytest

from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes, update_candidate_bounds
from isaaclab_arena.relations.loss_primitives import interval_overlap_axis_loss
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidate, PlacementCandidateBatch
from isaaclab_arena.relations.placement_candidate_generator import PlacementCandidateGenerator
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import IsAnchor, On, RotateAroundSolution
from isaaclab_arena.relations.validation.pre_physics import NoOverlapValidator
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, CachedOverlapVolumeUpperBound, OrientedBoundingBox
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
    from isaaclab_arena.relations.validation.pre_physics import NoOverlapValidator

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
    batch = PlacementCandidateBatch([PlacementCandidate(0, 0, positions, {}, env_bboxes)])
    assert validator.validate_batch(batch, []) == [True]


def test_validation_rejects_non_anchor_overlap():
    """Two overlapping non-anchor boxes should fail validation."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.validation.pre_physics import NoOverlapValidator

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
    batch = PlacementCandidateBatch([PlacementCandidate(0, 0, positions, {}, env_bboxes)])
    assert validator.validate_batch(batch, []) == [False]


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


def _harmonic_face_axis_volume(
    subject: OrientedBoundingBox, obstacle: OrientedBoundingBox, margin: float
) -> torch.Tensor:
    """Harmonic mean of the interval-overlap volumes along the subject's and the obstacle's face axes."""
    from isaaclab.utils.math import matrix_from_quat

    bound = CachedOverlapVolumeUpperBound(
        subject.half_extents,
        matrix_from_quat(subject.rotation_W_B_xyzw),
        obstacle.half_extents,
        matrix_from_quat(obstacle.rotation_W_B_xyzw),
    )
    volume_A, volume_B = bound.compute_face_axis_volumes(subject.center_W, obstacle.center_W, margin)
    return 2 * volume_A * volume_B / (volume_A + volume_B)


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


def test_solve_candidates_preserves_assigned_oriented_bounds_and_candidate_order():
    """Narrow rods clear each other and a passive obstacle; wide variants and obstacle intersections fail."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())
    narrow = AxisAlignedBoundingBox((-0.3, -0.02, -0.05), (0.3, 0.02, 0.05))
    wide = AxisAlignedBoundingBox((-0.3, -0.08, -0.05), (0.3, 0.08, 0.05))
    first, second = DummyObject("first", narrow), DummyObject("second", narrow)
    obstacle = DummyObject(
        "passive_obstacle",
        AxisAlignedBoundingBox((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
        initial_pose=Pose(position_xyz=(0.15, -0.15, 1.0)),
    )
    objects = [table, first, second]
    env_bounds = [
        {table: table.get_bounding_box(), first: narrow, second: narrow},
        {table: table.get_bounding_box(), first: wide, second: wide},
    ]
    positions = {table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (-0.08, 0.08, 1.0)}
    orientations = {first: math.pi / 4, second: math.pi / 4}
    batch = PlacementCandidateBatch([
        PlacementCandidate(1, 4, dict(positions), orientations, env_bounds[1]),
        PlacementCandidate(0, 2, dict(positions), orientations, env_bounds[0]),
        PlacementCandidate(0, 5, {**positions, first: (0.15, -0.15, 1.0)}, orientations, env_bounds[0]),
    ])
    update_candidate_bounds(batch, env_bounds)
    clear = batch.candidates[1]
    assert (
        clear.bboxes[first]
        .translated(positions[first])
        .overlaps(clear.bboxes[second].translated(positions[second]))
        .item()
    )
    assert clear.bboxes[first].translated(positions[first]).overlaps(obstacle.get_world_bounding_box()).item()

    solver = RelationSolver(RelationSolverParams(max_iters=0, verbose=False))
    solver.solve_candidates(objects, batch, [obstacle])

    assert [(candidate.env_id, candidate.candidate_id) for candidate in batch.candidates] == [(1, 4), (0, 2), (0, 5)]
    assert batch.candidates[0].loss > 0
    assert batch.candidates[1].loss == 0
    assert batch.candidates[2].loss > 0
    validator = NoOverlapValidator(ObjectPlacerParams(solver_params=solver.params))
    assert validator.validate_batch(batch, [obstacle]) == [False, True, False]
    assert all(obstacle not in candidate.positions for candidate in batch.candidates)
    assert obstacle.get_initial_pose().position_xyz == (0.15, -0.15, 1.0)


def test_solve_candidates_separates_yawed_boxes_while_preserving_on_support():
    """Optimization escapes a real rotated overlap without escaping above or outside the supporting table."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())
    bounds = AxisAlignedBoundingBox((-0.25, -0.04, 0.0), (0.25, 0.04, 0.1))
    first, second = DummyObject("first", bounds), DummyObject("second", bounds)
    first.add_relation(On(table, clearance_m=0.01))
    second.add_relation(On(table, clearance_m=0.01))
    objects = [table, first, second]
    positions = {table: (0.0, 0.0, 0.0), first: (0.45, 0.45, 0.11), second: (0.43, 0.47, 0.11)}
    orientations = {first: math.pi / 4, second: math.pi / 4}
    env_bounds = [{obj: obj.get_bounding_box() for obj in objects}]
    batch = PlacementCandidateBatch([PlacementCandidate(0, 0, positions, orientations, env_bounds[0])])
    update_candidate_bounds(batch, env_bounds)
    quaternion = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    assert (
        bounds.transformed(positions[first], quaternion)
        .overlaps(bounds.transformed(positions[second], quaternion))
        .item()
    )

    solver = RelationSolver(RelationSolverParams(max_iters=200, convergence_threshold=1e-4, verbose=False))
    solver.solve_candidates(objects, batch, [])

    candidate = batch.candidates[0]
    first_world = bounds.transformed(candidate.positions[first], quaternion)
    second_world = bounds.transformed(candidate.positions[second], quaternion)
    assert not first_world.overlaps(second_world, margin=solver.params.clearance_m).item()
    table_world = table.get_world_bounding_box()
    for box in (first_world, second_world):
        lower, upper = box.get_axis_aligned_bounds()
        torch.testing.assert_close(lower[:, 2], table_world.max_point[:, 2] + 0.01, atol=1e-5, rtol=0)
        assert (lower[:, :2] >= table_world.min_point[:, :2]).all()
        assert (upper[:, :2] <= table_world.max_point[:, :2]).all()
    assert candidate.loss < solver.params.convergence_threshold


def test_direct_solver_orients_raw_bounds_and_preserves_explicit_enclosing_bounds():
    """Raw per-env geometry follows yaw; already-enclosing custom bounds keep their conservative interval loss."""
    table = _create_table()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())
    default_bounds = AxisAlignedBoundingBox((-0.3, -0.12, -0.05), (0.3, 0.12, 0.05))
    first, second = DummyObject("first", default_bounds), DummyObject("second", default_bounds)
    raw_bounds = AxisAlignedBoundingBox((-0.3, -0.02, -0.05), (0.3, 0.02, 0.05))
    objects = [table, first, second]
    positions = {table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (-0.08, 0.08, 1.0)}
    orientations = [
        {first: math.pi / 4, second: math.pi / 4},
        {first: -math.pi / 4, second: -math.pi / 4},
    ]
    solver = RelationSolver(RelationSolverParams(max_iters=0, verbose=False))
    solver.solve(
        objects, [positions, positions], env_bboxes={first: raw_bounds, second: raw_bounds}, orientations=orientations
    )
    assert solver.last_loss_per_env[0] == 0
    assert solver.last_loss_per_env[1] > 0

    enclosing = raw_bounds.rotated_around_z(math.pi / 4)
    solver.solve(
        objects,
        [positions, positions],
        env_bboxes={first: enclosing, second: enclosing},
        env_bboxes_include_yaw=True,
        orientations=orientations,
    )
    expected = _single_pair_no_overlap_loss(
        solver._no_collision_strategy.slope,
        solver.params.clearance_m,
        torch.tensor(positions[first]),
        enclosing,
        enclosing.translated(positions[second]),
    )
    expected += _single_pair_no_overlap_loss(
        solver._no_collision_strategy.slope,
        solver.params.clearance_m,
        torch.tensor(positions[second]),
        enclosing,
        enclosing.translated(positions[first]),
    )
    torch.testing.assert_close(solver.last_loss_per_env.cpu(), expected.expand(2))
    assert (solver.last_loss_per_env > 0).all()


def test_direct_solver_orients_objects_omitted_from_partial_collision_bounds():
    """Partial supplied collision bounds keep precedence while omitted objects still receive their yaw."""
    table = _create_table()
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())
    rod = DummyObject("rod", AxisAlignedBoundingBox((-0.4, -0.02, -0.05), (0.4, 0.02, 0.05)))
    cube = DummyObject("cube", AxisAlignedBoundingBox((-0.005, -0.005, -0.05), (0.005, 0.005, 0.05)))
    supplied_bounds = AxisAlignedBoundingBox((-0.14, -0.02, -0.05), (0.14, 0.02, 0.05))
    positions = {table: (0.0, 0.0, 0.0), rod: (0.0, 0.0, 1.0), cube: (0.2, 0.3, 1.0)}
    collision_bboxes = {cube: supplied_bounds.to_obb()}
    solver = RelationSolver(RelationSolverParams(max_iters=0, clearance_m=0.0, verbose=False))

    solver.solve(
        [table, rod, cube],
        [positions, positions],
        orientations=[{rod: math.pi / 4}, {rod: -math.pi / 4}],
        env_bboxes={cube: cube.get_bounding_box()},
        collision_bboxes=collision_bboxes,
    )

    # The supplied wide cube reaches the diagonal rod; its tiny default bounds do not.
    assert solver.last_loss_per_env[0] > 0
    assert solver.last_loss_per_env[1] == 0
    # The caller's bounds are not extended with the defaults.
    assert set(collision_bboxes) == {cube}


@pytest.mark.parametrize(
    "marker_angles, candidate_yaws",
    [
        pytest.param((0.0, 0.0, 0.0), (0.0, math.pi / 2, -math.pi / 2, -math.pi), id="candidate-yaws"),
        pytest.param((math.pi / 2, 0.0, 0.0), (None,), id="authored-roll"),
    ],
)
def test_exact_quarter_turns_preserve_aabb_collision_loss_and_gradients(marker_angles, candidate_yaws):
    """Exact quarter turns, with authored headings from the candidate generator, match AABB losses and gradients."""
    table, first, second = _create_no_collision_scene()
    bounds = AxisAlignedBoundingBox((-0.2, -0.08, -0.04), (0.4, 0.12, 0.1))
    for obj in (first, second):
        obj.bounding_box = bounds
        obj.add_relation(RotateAroundSolution(*marker_angles))
    objects = [table, first, second]
    base_bounds = {obj: obj.get_bounding_box() for obj in objects}
    positions = {table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (0.03, 0.04, 1.02)}
    generated_orientations = PlacementCandidateGenerator(ObjectPlacerParams()).generate_orientations(objects, {table})
    candidates = []
    for index, yaw in enumerate(candidate_yaws):
        orientations = generated_orientations if yaw is None else {first: yaw, second: yaw}
        candidates.append(PlacementCandidate(0, index, positions, orientations, base_bounds))
    batch = PlacementCandidateBatch(candidates)
    update_candidate_bounds(batch, [base_bounds])
    collision_boxes = batch.stacked_collision_bboxes()
    assert all(isinstance(bounds, AxisAlignedBoundingBox) for bounds in collision_boxes.values())
    initial_positions = [candidate.positions for candidate in candidates]
    state = RelationSolverState(
        objects,
        initial_positions,
        device=torch.device("cpu"),
        env_bboxes=batch.stacked_bboxes(),
        collision_bboxes=collision_boxes,
    )
    reference = RelationSolverState(
        objects, initial_positions, device=torch.device("cpu"), env_bboxes=batch.stacked_bboxes()
    )
    assert state.collision_boxes_axis_aligned
    solver = RelationSolver(RelationSolverParams(verbose=False))

    loss = solver._compute_no_overlap_loss(state)
    reference_loss = _reference_pairwise_no_overlap_loss(solver, reference)
    loss.sum().backward()
    reference_loss.sum().backward()

    assert (loss > 0).all()
    assert state.optimizable_positions.grad.abs().sum() > 0
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(state.optimizable_positions.grad, reference.optimizable_positions.grad)


def test_near_quarter_turn_remains_oriented_collision_geometry():
    """A nearby angle must not be snapped to the quarter-turn AABB collision path."""
    table, first, second = _create_no_collision_scene()
    objects = [table, first, second]
    bounds = {obj: obj.get_bounding_box() for obj in objects}
    positions = {table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (0.03, 0.04, 1.02)}
    batch = PlacementCandidateBatch([PlacementCandidate(0, 0, positions, {first: math.pi / 2 + 1e-6}, bounds)])
    update_candidate_bounds(batch, [bounds])

    assert not isinstance(batch.collision_bboxes[first], AxisAlignedBoundingBox)
    state = RelationSolverState(
        objects, [positions], device=torch.device("cpu"), collision_bboxes=batch.stacked_collision_bboxes()
    )
    assert not state.collision_boxes_axis_aligned


def test_mixed_collision_geometry_rejects_unbounded_aabbs_without_changing_interval_behavior():
    """Unbounded intervals keep AABB semantics but cannot enter finite oriented-box geometry."""
    table, first, second = _create_no_collision_scene()
    first.bounding_box = AxisAlignedBoundingBox((-math.inf, -0.1, -0.1), (math.inf, 0.1, 0.1))
    objects = [table, first, second]
    positions = [{table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (0.03, 0.04, 1.02)}]
    state = RelationSolverState(objects, positions, device=torch.device("cpu"))
    assert state.collision_boxes_axis_aligned
    solver = RelationSolver(RelationSolverParams(verbose=False))

    loss = solver._compute_no_overlap_loss(state)
    torch.testing.assert_close(loss, _reference_pairwise_no_overlap_loss(solver, state))
    assert torch.isfinite(loss).all() and (loss > 0).all()

    quaternion = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    oriented = second.get_bounding_box().transformed((0.0, 0.0, 0.0), quaternion)
    with pytest.raises(AssertionError, match="finite"):
        RelationSolverState(objects, positions, device=torch.device("cpu"), collision_bboxes={second: oriented})


@pytest.mark.parametrize("clearance_m, expected_valid", [(0.0, True), (0.01, False)])
def test_oriented_collision_loss_and_validator_apply_clearance(clearance_m, expected_valid):
    """Parallel diagonal rods with a 5 mm surface gap violate a 10 mm clearance requirement."""
    table = _create_table()
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())
    bounds = AxisAlignedBoundingBox((-0.25, -0.02, -0.03), (0.25, 0.02, 0.03))
    first, second = DummyObject("first", bounds), DummyObject("second", bounds)
    offset = (0.04 + 0.005) / math.sqrt(2)
    positions = {table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (-offset, offset, 1.0)}
    objects = [table, first, second]
    boxes = {obj: obj.get_bounding_box() for obj in objects}
    batch = PlacementCandidateBatch(
        [PlacementCandidate(0, 0, positions, {first: math.pi / 4, second: math.pi / 4}, boxes)]
    )
    update_candidate_bounds(batch, [boxes])
    params = RelationSolverParams(max_iters=0, clearance_m=clearance_m, verbose=False)
    solver = RelationSolver(params)

    solver.solve_candidates(objects, batch, [])

    if expected_valid:
        assert batch.candidates[0].loss == 0.0
    else:
        assert batch.candidates[0].loss > 0.0
    validator = NoOverlapValidator(ObjectPlacerParams(solver_params=params, verbose=False))
    assert validator.validate_batch(batch, []) == [expected_valid]

    # A fixed cube 5 mm from the first rod's long face is held to the same clearance.
    cube_distance = 0.02 + 0.02 * math.sqrt(2) + 0.005
    obstacle = DummyObject(
        "obstacle",
        AxisAlignedBoundingBox((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
        initial_pose=Pose(position_xyz=(-cube_distance / math.sqrt(2), cube_distance / math.sqrt(2), 1.0)),
    )
    first_boxes = {table: boxes[table], first: boxes[first]}
    alone = PlacementCandidateBatch([
        PlacementCandidate(0, 0, {table: positions[table], first: positions[first]}, {first: math.pi / 4}, first_boxes)
    ])
    update_candidate_bounds(alone, [first_boxes])
    assert validator.validate_batch(alone, [obstacle]) == [expected_valid]


def test_oriented_loss_pushes_out_a_bar_whose_smaller_face_axis_volume_is_flat():
    """A yawed bar whose own-axis volume is constant still gets a gradient from the obstacle's frame."""
    cube = DummyObject(
        "cube",
        AxisAlignedBoundingBox((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )
    bar = DummyObject("bar", AxisAlignedBoundingBox((-0.2, -0.005, -0.05), (0.2, 0.005, 0.05)))
    quaternion = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    bar_collision_box = bar.get_bounding_box().transformed((0.0, 0.0, 0.0), quaternion)
    state = RelationSolverState(
        [cube, bar],
        [{cube: (0.0, 0.0, 1.0), bar: (0.05, 0.0, 1.0)}],
        device=torch.device("cpu"),
        collision_bboxes={bar: bar_collision_box},
    )
    solver = RelationSolver(RelationSolverParams(clearance_m=0.0, verbose=False))

    loss = solver._compute_no_overlap_loss(state)  # pyright: ignore[reportPrivateUsage]
    loss.sum().backward()

    # The smaller face-axis volume, along the bar's own axes, does not change when the bar moves.
    position = torch.tensor([[0.05, 0.0, 1.0]], requires_grad=True)
    smaller_volume = bar_collision_box.translated(position).overlap_volume_upper_bound(cube.get_world_bounding_box())
    smaller_volume.sum().backward()
    assert loss.item() > 0 and position.grad.abs().sum() == 0
    # Descending the loss moves the bar away from the cube along +x.
    assert state.optimizable_positions.grad[0, 0, 0] < 0


def test_oriented_loss_pushes_zero_thickness_panel_out_of_fixed_block():
    """A yawed panel with zero thickness partly inside a fixed block has a loss to reduce, and the solver moves it out."""
    table = _create_table()
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())
    panel_bounds = AxisAlignedBoundingBox((-0.2, 0.0, -0.1), (0.2, 0.0, 0.1))
    panel = DummyObject("panel", panel_bounds)
    block = DummyObject(
        "block",
        AxisAlignedBoundingBox((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0)),
    )
    objects = [table, panel]
    boxes = {table: table.get_bounding_box(), panel: panel_bounds}
    yaw = math.pi / 6
    # The panel's center is 0.1 m from the block's along the panel's long axis, so one end sticks out.
    positions = {table: (0.0, 0.0, 0.0), panel: (0.1 * math.cos(yaw), 0.1 * math.sin(yaw), 1.0)}
    params = RelationSolverParams(max_iters=0, verbose=False)
    validator = NoOverlapValidator(ObjectPlacerParams(solver_params=params, verbose=False))
    results = []
    for max_iters in (0, 300):
        batch = PlacementCandidateBatch([PlacementCandidate(0, 0, dict(positions), {panel: yaw}, boxes)])
        update_candidate_bounds(batch, [boxes])
        params.max_iters = max_iters
        RelationSolver(params).solve_candidates(objects, batch, [block])
        results.append((batch.candidates[0].loss, validator.validate_batch(batch, [block])))

    (start_loss, start_valid), (final_loss, final_valid) = results
    assert start_loss > 0.0 and start_valid == [False]
    assert final_loss < start_loss and final_valid == [True]


def test_oriented_collision_uses_off_origin_quarter_turned_anchor_world_bounds():
    """A non-parent anchor contributes its rotated world bounds exactly once."""
    quarter = math.sqrt(0.5)
    anchor_position = (0.4, -0.2, 0.1)
    anchor = DummyObject(
        "anchor",
        AxisAlignedBoundingBox((-0.2, -0.025, -0.05), (0.2, 0.025, 0.05)),
        initial_pose=Pose(position_xyz=anchor_position, rotation_xyzw=(0.0, 0.0, quarter, quarter)),
        relations=[IsAnchor()],
    )
    rod = DummyObject("rod", AxisAlignedBoundingBox((-0.08, -0.01, -0.02), (0.08, 0.01, 0.02)))
    objects = [anchor, rod]
    per_env = build_per_env_bounding_boxes(objects, 1).get_bounding_boxes_for_all_envs()
    batch = PlacementCandidateBatch([
        PlacementCandidate(0, index, {anchor: anchor_position, rod: position}, {rod: math.pi / 4}, per_env[0])
        for index, position in enumerate(((0.4, -0.08, 0.1), (0.8, -0.4, 0.2)))
    ])
    update_candidate_bounds(batch, per_env)
    params = RelationSolverParams(max_iters=0, clearance_m=0.0, verbose=False)
    solver = RelationSolver(params)

    solver.solve_candidates(objects, batch, [])

    assert batch.candidates[0].loss > 0.0
    assert batch.candidates[1].loss == 0.0
    validator = NoOverlapValidator(ObjectPlacerParams(solver_params=params, verbose=False))
    assert validator.validate_batch(batch, []) == [False, True]


def test_oriented_collision_cache_uses_current_positions_on_repeated_backward():
    """Moving heterogeneous boxes between gradient steps agrees with fresh directed pair calculations."""
    table, first, second = _create_no_collision_scene()
    objects = [table, first, second]
    positions = [
        {table: (0.0, 0.0, 0.0), first: (0.0, 0.0, 1.0), second: (0.1, 0.02, 1.01)},
        {table: (0.0, 0.0, 0.0), first: (0.04, 0.0, 1.0), second: (0.08, 0.05, 1.02)},
    ]
    base_bounds = AxisAlignedBoundingBox(
        torch.tensor([[-0.2, -0.06, -0.04], [-0.3, -0.04, -0.03]]),
        torch.tensor([[0.2, 0.06, 0.04], [0.3, 0.04, 0.03]]),
    )
    quaternion = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    collision_boxes = {
        first: base_bounds.transformed((0.0, 0.0, 0.0), quaternion),
        second: base_bounds.transformed((0.0, 0.0, 0.0), (0.0, 0.0, -quaternion[2], quaternion[3])),
    }
    state = RelationSolverState(objects, positions, device=torch.device("cpu"), collision_bboxes=collision_boxes)
    solver = RelationSolver(RelationSolverParams(clearance_m=0.01, verbose=False))
    for step in range(2):
        if step:
            with torch.no_grad():
                state.optimizable_positions[:, 1, 0] += 0.035
            state.optimizable_positions.grad = None
        reference_positions = state.optimizable_positions.detach().clone().requires_grad_(True)
        reference_loss = 0.0
        for subject_index, subject, obstacle_index, obstacle in ((0, first, 1, second), (1, second, 0, first)):
            subject_world = collision_boxes[subject].translated(reference_positions[:, subject_index])
            obstacle_world = collision_boxes[obstacle].translated(reference_positions[:, obstacle_index].detach())
            reference_loss += solver._no_collision_strategy.slope * _harmonic_face_axis_volume(
                subject_world, obstacle_world, margin=solver.params.clearance_m
            )
        loss = solver._compute_no_overlap_loss(state)
        loss.sum().backward()
        reference_loss.sum().backward()

        assert (loss > 0).all()
        assert reference_positions.grad.abs().sum() > 0
        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(state.optimizable_positions.grad, reference_positions.grad)
