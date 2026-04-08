# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh-level collision detection in ObjectPlacer."""

import trimesh

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import CollisionMode, ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NoCollision, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _create_cylinder_object(name: str, radius: float = 0.05, height: float = 0.2) -> DummyObject:
    """Create a cylinder-shaped object with an AABB that is larger than the mesh."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height)
    bbox_half = max(radius, radius)
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-bbox_half, -bbox_half, -height / 2),
            max_point=(bbox_half, bbox_half, height / 2),
        ),
        collision_mesh=mesh,
    )


def _create_box_object(name: str, size: float = 0.2) -> DummyObject:
    """Create a box-shaped object where AABB == mesh extents."""
    mesh = trimesh.creation.box(extents=(size, size, size))
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-size / 2, -size / 2, -size / 2),
            max_point=(size / 2, size / 2, size / 2),
        ),
        collision_mesh=mesh,
    )


def _create_l_shaped_object(name: str) -> DummyObject:
    """Create an L-shaped object from two boxes.

    The AABB covers the full 0.3 x 0.3 x 0.1 envelope, but the actual mesh
    only occupies two arms of the L, leaving a 0.2 x 0.2 gap in one corner.
    """
    arm_a = trimesh.creation.box(extents=(0.3, 0.1, 0.1))
    arm_a.apply_translation((0.0, -0.1, 0.0))
    arm_b = trimesh.creation.box(extents=(0.1, 0.3, 0.1))
    arm_b.apply_translation((-0.1, 0.0, 0.0))
    mesh = trimesh.util.concatenate([arm_a, arm_b])
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-0.15, -0.15, -0.05),
            max_point=(0.15, 0.15, 0.05),
        ),
        collision_mesh=mesh,
    )


# =============================================================================
# Direct mesh collision manager tests
# =============================================================================


def test_two_cylinders_aabb_overlap_but_mesh_no_collision():
    """Two cylinders placed diagonally: AABBs overlap, but actual meshes do not."""
    cyl_a = _create_cylinder_object("cyl_a", radius=0.05, height=0.2)
    cyl_b = _create_cylinder_object("cyl_b", radius=0.05, height=0.2)

    # Place them diagonally so their square AABBs overlap but round cross-sections don't.
    # AABB of each is 0.1 x 0.1.  At offset (0.08, 0.08, 0), AABBs overlap by 0.02
    # on each axis, but circles of radius 0.05 at distance sqrt(0.08^2+0.08^2)=0.113
    # do not overlap (sum of radii = 0.10).
    pos_a = (0.0, 0.0, 0.0)
    pos_b = (0.08, 0.08, 0.0)

    positions = {cyl_a: pos_a, cyl_b: pos_b}

    # AABB mode should reject (overlap)
    placer_aabb = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.AABB))
    assert not placer_aabb._validate_no_overlap_aabb(positions), "AABB should detect overlap"

    # Mesh mode should accept (no mesh collision)
    placer_mesh = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.MESH))
    assert placer_mesh._validate_no_overlap_mesh(positions), "Mesh should not detect collision for diagonal cylinders"


def test_two_boxes_true_collision_detected_by_both_modes():
    """Two boxes that truly overlap should be rejected by both AABB and mesh modes."""
    box_a = _create_box_object("box_a", size=0.2)
    box_b = _create_box_object("box_b", size=0.2)

    positions = {box_a: (0.0, 0.0, 0.0), box_b: (0.1, 0.1, 0.0)}

    placer_aabb = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.AABB))
    assert not placer_aabb._validate_no_overlap_aabb(positions)

    placer_mesh = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.MESH))
    assert not placer_mesh._validate_no_overlap_mesh(positions)


def test_l_shaped_objects_mesh_allows_interlocking():
    """Two L-shaped objects can interlock in the gap; mesh mode accepts, AABB rejects."""
    l_a = _create_l_shaped_object("l_a")
    l_b = _create_box_object("small_box", size=0.08)

    # Place the small box in the gap of the L shape (upper-right quadrant).
    positions = {l_a: (0.0, 0.0, 0.0), l_b: (0.07, 0.07, 0.0)}

    placer_aabb = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.AABB))
    assert not placer_aabb._validate_no_overlap_aabb(positions), "AABB should detect overlap for L + box"

    placer_mesh = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.MESH))
    assert placer_mesh._validate_no_overlap_mesh(positions), "Mesh should allow box in L-shape gap"


def test_separated_objects_pass_both_modes():
    """Objects far apart should pass both AABB and mesh validation."""
    box_a = _create_box_object("box_a", size=0.2)
    box_b = _create_box_object("box_b", size=0.2)

    positions = {box_a: (0.0, 0.0, 0.0), box_b: (1.0, 1.0, 0.0)}

    placer_aabb = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.AABB))
    assert placer_aabb._validate_no_overlap_aabb(positions)

    placer_mesh = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.MESH))
    assert placer_mesh._validate_no_overlap_mesh(positions)


def test_on_relation_pairs_skipped_in_mesh_mode():
    """Objects linked by On relation should not be flagged as colliding in mesh mode."""
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
        collision_mesh=trimesh.creation.box(extents=(1.0, 1.0, 0.1)),
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())

    box = _create_box_object("box", size=0.2)
    box.add_relation(On(table, clearance_m=0.01))

    # Box placed on table surface -- their meshes overlap vertically but On pair is skipped.
    positions = {table: (0.0, 0.0, 0.0), box: (0.5, 0.5, 0.11)}

    placer = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.MESH))
    assert placer._validate_no_overlap_mesh(positions)


def test_mesh_mode_falls_back_to_aabb_box_when_no_mesh():
    """When an object has no collision mesh, mesh mode should generate an AABB box as fallback."""
    obj_with_mesh = _create_box_object("with_mesh", size=0.2)
    obj_without_mesh = DummyObject(
        name="no_mesh",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
    )

    # Overlapping positions
    positions = {obj_with_mesh: (0.0, 0.0, 0.0), obj_without_mesh: (0.05, 0.05, 0.0)}

    placer = ObjectPlacer(params=ObjectPlacerParams(collision_mode=CollisionMode.MESH))
    assert not placer._validate_no_overlap_mesh(positions), "Fallback AABB box should still detect overlap"


# =============================================================================
# Integration: ObjectPlacer.place() with mesh mode
# =============================================================================


def test_placer_place_with_mesh_mode():
    """ObjectPlacer.place() should work end-to-end with CollisionMode.MESH."""
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
        collision_mesh=trimesh.creation.box(extents=(1.0, 1.0, 0.1)),
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())

    box1 = _create_box_object("box1", size=0.15)
    box2 = _create_box_object("box2", size=0.15)
    box1.add_relation(On(table, clearance_m=0.01))
    box2.add_relation(On(table, clearance_m=0.01))
    box2.add_relation(NoCollision(box1))

    solver_params = RelationSolverParams(max_iters=200, convergence_threshold=1e-3)
    params = ObjectPlacerParams(
        placement_seed=42,
        solver_params=solver_params,
        collision_mode=CollisionMode.MESH,
    )
    placer = ObjectPlacer(params=params)
    result = placer.place([table, box1, box2])

    assert result.success, "Placement should succeed with mesh collision mode"
    assert box1 in result.positions
    assert box2 in result.positions
