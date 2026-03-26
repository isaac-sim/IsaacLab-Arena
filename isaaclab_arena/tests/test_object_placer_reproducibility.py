# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPlacer and RelationSolver reproducibility."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box
from isaaclab_arena.utils.pose import Pose


def _create_test_objects() -> tuple[DummyObject, DummyObject, DummyObject]:
    """Create test objects with relations (without setting initial poses for non-anchors)."""
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    desk.add_relation(IsAnchor())

    box1 = DummyObject(
        name="box1",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box2 = DummyObject(
        name="box2",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
    )

    box1.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(On(desk, clearance_m=0.01))
    box2.add_relation(NextTo(box1, side=Side.POSITIVE_X, distance_m=0.05))

    return desk, box1, box2


def test_get_random_pose_same_seed_produces_identical_result():
    """Test that get_random_pose_within_bounding_box with same seed produces identical poses."""
    bbox = AxisAlignedBoundingBox(min_point=(-1.0, -1.0, 0.0), max_point=(1.0, 1.0, 1.0))

    pose1 = get_random_pose_within_bounding_box(bbox, seed=42)
    pose2 = get_random_pose_within_bounding_box(bbox, seed=42)

    assert pose1.position_xyz == pose2.position_xyz


def test_relation_solver_same_inputs_produces_identical_result():
    """Test that RelationSolver with identical initial positions produces identical results."""
    desk_pos = (0.0, 0.0, 0.0)
    fixed_box1_pos = (0.5, 0.5, 0.5)
    fixed_box2_pos = (0.3, 0.7, 0.3)
    solver_params = RelationSolverParams(max_iters=10)

    # Run 1
    desk1, box1_run1, box2_run1 = _create_test_objects()
    initial_positions1 = {desk1: desk_pos, box1_run1: fixed_box1_pos, box2_run1: fixed_box2_pos}

    solver1 = RelationSolver(params=solver_params)
    result1 = solver1.solve(objects=[desk1, box1_run1, box2_run1], initial_positions=initial_positions1)

    # Run 2 (fresh objects, same initial positions)
    desk2, box1_run2, box2_run2 = _create_test_objects()
    initial_positions2 = {desk2: desk_pos, box1_run2: fixed_box1_pos, box2_run2: fixed_box2_pos}

    solver2 = RelationSolver(params=solver_params)
    result2 = solver2.solve(objects=[desk2, box1_run2, box2_run2], initial_positions=initial_positions2)

    # Compare by name (different object instances)
    for obj1 in result1:
        pos1 = result1[obj1]
        pos2 = next(result2[obj2] for obj2 in result2 if obj2.name == obj1.name)
        assert pos1 == pos2, f"Mismatch for {obj1.name}: {pos1} != {pos2}"


def test_object_placer_same_seed_produces_identical_result():
    """Test that ObjectPlacer with same seed produces identical final results."""
    seed = 42
    solver_params = RelationSolverParams(max_iters=10)

    # Run 1
    desk1, box1_run1, box2_run1 = _create_test_objects()
    objects1 = [desk1, box1_run1, box2_run1]
    placer1 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=seed, solver_params=solver_params))
    result1 = placer1.place(objects=objects1)

    # Run 2
    desk2, box1_run2, box2_run2 = _create_test_objects()
    objects2 = [desk2, box1_run2, box2_run2]
    placer2 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=seed, solver_params=solver_params))
    result2 = placer2.place(objects=objects2)

    # Compare by name
    for obj1, obj2 in zip(objects1, objects2):
        pos1 = result1.positions[obj1]
        pos2 = result2.positions[obj2]
        assert pos1 == pos2, f"Mismatch for {obj1.name}: {pos1} != {pos2}"


def test_object_placer_different_seeds_produce_different_results():
    """Test that ObjectPlacer with different seeds produces different results."""
    solver_params = RelationSolverParams(max_iters=10)

    # Run 1 with seed 42
    desk1, box1_run1, box2_run1 = _create_test_objects()
    objects1 = [desk1, box1_run1, box2_run1]
    placer1 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=42, solver_params=solver_params))
    result1 = placer1.place(objects=objects1)

    # Run 2 with seed 123
    desk2, box1_run2, box2_run2 = _create_test_objects()
    objects2 = [desk2, box1_run2, box2_run2]
    placer2 = ObjectPlacer(params=ObjectPlacerParams(placement_seed=123, solver_params=solver_params))
    result2 = placer2.place(objects=objects2)

    # Check that at least one non-anchor position differs
    any_different = False
    for obj1, obj2 in zip(objects1[1:], objects2[1:]):  # Skip anchor
        pos1 = result1.positions[obj1]
        pos2 = result2.positions[obj2]
        if pos1 != pos2:
            any_different = True
            break

    assert any_different, "Different seeds should produce different results"
