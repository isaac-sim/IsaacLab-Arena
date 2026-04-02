# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for On-relation-guided initialization in ObjectPlacer."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def _make_desk():
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def test_on_init_x_y_within_parent_footprint():
    """Object with On(anchor) is initialized with its bbox fully within parent's X/Y footprint."""
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    positions = placer._generate_initial_positions([desk, box], {desk})

    x, y, _ = positions[box]
    child_bbox = box.get_bounding_box()
    desk_world = desk.get_world_bounding_box()

    assert x + child_bbox.min_point[0] >= desk_world.min_point[0] - 1e-6
    assert x + child_bbox.max_point[0] <= desk_world.max_point[0] + 1e-6
    assert y + child_bbox.min_point[1] >= desk_world.min_point[1] - 1e-6
    assert y + child_bbox.max_point[1] <= desk_world.max_point[1] + 1e-6


def test_on_init_z_places_bottom_at_parent_top():
    """Object with On(anchor) is initialized with its bottom face at parent top + clearance."""
    desk = _make_desk()
    clearance_m = 0.01
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=clearance_m))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    positions = placer._generate_initial_positions([desk, box], {desk})

    _, _, z = positions[box]
    child_bbox = box.get_bounding_box()
    desk_world = desk.get_world_bounding_box()

    child_bottom = z + child_bbox.min_point[2]
    expected_bottom = desk_world.max_point[2] + clearance_m
    assert abs(child_bottom - expected_bottom) < 1e-6


def test_on_init_clamps_to_center_when_child_wider_than_parent():
    """Object wider than its On parent in X/Y is clamped to parent center, not an invalid range."""
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())

    big_box = DummyObject(
        name="big_box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 0.5, 0.2)),
    )
    big_box.add_relation(On(desk, clearance_m=0.0))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    positions = placer._generate_initial_positions([desk, big_box], {desk})

    x, y, _ = positions[big_box]
    desk_world = desk.get_world_bounding_box()
    center_x = (desk_world.min_point[0] + desk_world.max_point[0]) / 2.0
    center_y = (desk_world.min_point[1] + desk_world.max_point[1]) / 2.0

    assert abs(x - center_x) < 1e-6
    assert abs(y - center_y) < 1e-6


def test_no_on_relation_initializes_within_anchor_world_bbox():
    """Object origin with no On relation is initialized randomly within the first anchor's world bbox."""
    desk = _make_desk()
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(NextTo(desk, side=Side.POSITIVE_X, distance_m=0.05))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    positions = placer._generate_initial_positions([desk, box], {desk})

    x, y, z = positions[box]
    desk_world = desk.get_world_bounding_box()

    assert desk_world.min_point[0] <= x <= desk_world.max_point[0]
    assert desk_world.min_point[1] <= y <= desk_world.max_point[1]
    assert desk_world.min_point[2] <= z <= desk_world.max_point[2]


def test_on_non_anchor_parent_with_anchor_grandparent_uses_proxy():
    """Object with On(non-anchor) uses the grandparent anchor's bbox as proxy when parent has On(anchor)."""
    desk = _make_desk()
    plate = DummyObject(
        name="plate",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.02)),
    )
    plate.add_relation(On(desk, clearance_m=0.01))

    mug = DummyObject(
        name="mug",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.12)),
    )
    mug.add_relation(On(plate, clearance_m=0.0))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    positions = placer._generate_initial_positions([desk, plate, mug], {desk})

    x, y, z = positions[mug]
    desk_world = desk.get_world_bounding_box()

    # Mug's parent (plate) is non-anchor with On(desk): uses desk's bbox as proxy
    assert desk_world.min_point[0] <= x <= desk_world.max_point[0]
    assert desk_world.min_point[1] <= y <= desk_world.max_point[1]
    # Z: desk top (0.1) + clearance (0.0) - mug bbox min_z (0.0) = 0.1
    mug_bbox = mug.get_bounding_box()
    assert abs(z - (desk_world.max_point[2] + 0.0 - mug_bbox.min_point[2])) < 1e-6


def test_on_non_anchor_parent_without_on_uses_fallback_bbox():
    """Object with On(non-anchor) where parent has no On relation falls back to first anchor's world bbox."""
    desk = _make_desk()
    stand = DummyObject(
        name="stand",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.5)),
    )
    stand.add_relation(NextTo(desk, side=Side.POSITIVE_X, distance_m=0.1))

    mug = DummyObject(
        name="mug",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.12)),
    )
    mug.add_relation(On(stand, clearance_m=0.0))

    placer = ObjectPlacer(params=ObjectPlacerParams())
    positions = placer._generate_initial_positions([desk, stand, mug], {desk})

    x, y, z = positions[mug]
    desk_world = desk.get_world_bounding_box()

    # Parent (stand) has no On relation: falls back to desk's world bbox
    assert desk_world.min_point[0] <= x <= desk_world.max_point[0]
    assert desk_world.min_point[1] <= y <= desk_world.max_point[1]
    # Z: desk.max_z (fallback) + clearance (0.0) - mug.min_z (0.0) = 0.1
    assert abs(z - (desk_world.max_point[2] + 0.0 - mug.get_bounding_box().min_point[2])) < 1e-6


def test_on_init_reproducible_with_placement_seed():
    """Same placement_seed produces identical On-guided init positions across independent runs."""
    solver_params = RelationSolverParams(max_iters=0, save_position_history=False, verbose=False)
    params = ObjectPlacerParams(placement_seed=42, apply_positions_to_objects=False, solver_params=solver_params)

    def _run():
        desk = _make_desk()
        box = DummyObject(
            name="box",
            bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
        )
        box.add_relation(On(desk, clearance_m=0.01))
        return ObjectPlacer(params=params).place([desk, box])

    result1 = _run()
    result2 = _run()

    pos1 = next(pos for obj, pos in result1.positions.items() if obj.name == "box")
    pos2 = next(pos for obj, pos in result2.positions.items() if obj.name == "box")
    assert pos1 == pos2, f"Expected identical positions with same seed, got {pos1} vs {pos2}"
