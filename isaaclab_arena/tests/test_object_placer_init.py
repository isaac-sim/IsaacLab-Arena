# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement initializers that seed the relation solver."""

import torch

import pytest

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_initializers import AnchorInitializer
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

ALL_INITIALIZERS = [AnchorInitializer]


def _make_desk():
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    return desk


def _make_box(name, size=0.2, height=0.2):
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(size, size, height)),
    )


def _env_bboxes(objects):
    return {obj: obj.get_bounding_box() for obj in objects}


def _seed(initializer, objects, anchors, env_bboxes=None, generator=None):
    return initializer.generate_initial_positions(objects, anchors, env_bboxes or _env_bboxes(objects), generator)


def _assert_footprint_within(position, child_bbox, parent_bbox, tolerance=1e-6):
    """Assert the child's X/Y footprint at position lies inside the parent's world footprint."""
    x, y, _ = position
    for axis, value in ((0, x), (1, y)):
        assert value + float(child_bbox.min_point[0, axis]) >= float(parent_bbox.min_point[0, axis]) - tolerance
        assert value + float(child_bbox.max_point[0, axis]) <= float(parent_bbox.max_point[0, axis]) + tolerance


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_on_init_x_y_within_parent_footprint(initializer_cls):
    """Object with On(anchor) is initialized with its bbox fully within parent's X/Y footprint."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.01))

    positions = _seed(initializer_cls(), [desk, box], {desk})

    _assert_footprint_within(positions[box], box.get_bounding_box(), desk.get_world_bounding_box())


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_on_init_z_places_bottom_at_parent_top(initializer_cls):
    """Object with On(anchor) is initialized with its bottom face at parent top + clearance."""
    desk = _make_desk()
    clearance_m = 0.01
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=clearance_m))

    positions = _seed(initializer_cls(), [desk, box], {desk})

    _, _, z = positions[box]
    child_bottom = z + float(box.get_bounding_box().min_point[0, 2])
    expected_bottom = float(desk.get_world_bounding_box().max_point[0, 2]) + clearance_m
    assert abs(child_bottom - expected_bottom) < 1e-6


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_on_init_uses_env_specific_parent_bbox(initializer_cls):
    """Object with On(anchor set) should initialize against that env's assigned bbox."""
    table_set = DummyObject(
        name="table_set",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 2.0, 0.5)),
    )
    table_set.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table_set.add_relation(IsAnchor())

    small_table_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.1))
    box_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.05, 0.05, 0.05))
    box = DummyObject(name="box", bounding_box=box_bbox)
    box.add_relation(On(table_set, clearance_m=0.02, edge_margin_m=0.0))

    positions = _seed(
        initializer_cls(),
        [table_set, box],
        {table_set},
        env_bboxes={table_set: small_table_bbox, box: box_bbox},
    )

    x, y, z = positions[box]
    assert small_table_bbox.min_point[0, 0] <= x <= small_table_bbox.max_point[0, 0]
    assert small_table_bbox.min_point[0, 1] <= y <= small_table_bbox.max_point[0, 1]
    assert abs(z - float(small_table_bbox.max_point[0, 2] + 0.02 - box_bbox.min_point[0, 2])) < 1e-6


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_on_init_clamps_to_center_when_child_wider_than_parent(initializer_cls):
    """Object wider than its On parent in X/Y is clamped to parent center, not an invalid range."""
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())

    big_box = _make_box("big_box", size=0.5)
    big_box.add_relation(On(desk, clearance_m=0.0))

    positions = _seed(initializer_cls(), [desk, big_box], {desk})

    x, y, _ = positions[big_box]
    desk_center = desk.get_world_bounding_box().center[0]
    assert abs(x - float(desk_center[0])) < 1e-6
    assert abs(y - float(desk_center[1])) < 1e-6


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_no_on_relation_initializes_at_anchor_center(initializer_cls):
    """Object with no On relation is initialized at the first anchor's center; solver handles placement."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(NextTo(desk, side=Side.POSITIVE_X, distance_m=0.05))

    positions = _seed(initializer_cls(), [desk, box], {desk})

    center = desk.get_world_bounding_box().center[0]
    for axis, value in enumerate(positions[box]):
        assert abs(value - float(center[axis])) < 1e-6


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_on_init_overlap_can_leave_parent_footprint(initializer_cls):
    """Overlap initialization samples against the original support, ignoring a large edge margin.

    A containment reading of this relation would be infeasible (margin 0.6 on a 1.0 m desk), so
    every draw would collapse to the desk center. Overlap instead admits any pose whose footprint
    intersects the desk, which includes origins outside the desk itself.
    """
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, overlap=True, edge_margin_m=0.6))
    generator = torch.Generator().manual_seed(0)

    samples = [_seed(initializer_cls(), [desk, box], {desk}, generator=generator)[box] for _ in range(200)]

    desk_world = desk.get_world_bounding_box()
    child_bbox = box.get_bounding_box()
    # The overlap interval for the child origin is [desk_min - child_max, desk_max - child_min].
    for x, y, _ in samples:
        for axis, value in ((0, x), (1, y)):
            assert value >= float(desk_world.min_point[0, axis] - child_bbox.max_point[0, axis]) - 1e-6
            assert value <= float(desk_world.max_point[0, axis] - child_bbox.min_point[0, axis]) + 1e-6
    assert any(x < float(desk_world.min_point[0, 0]) for x, _, _ in samples), "Overlap never left the footprint"


def test_anchor_init_on_non_anchor_parent_uses_grandparent_proxy():
    """AnchorInitializer resolves only one On level, seeding a grandchild across the anchor."""
    desk = _make_desk()
    plate = DummyObject(
        name="plate",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.02)),
    )
    plate.add_relation(On(desk, clearance_m=0.01))
    mug = _make_box("mug", size=0.1, height=0.12)
    mug.add_relation(On(plate, clearance_m=0.0))

    positions = _seed(AnchorInitializer(), [desk, plate, mug], {desk})

    x, y, z = positions[mug]
    desk_world = desk.get_world_bounding_box()
    assert desk_world.min_point[0, 0] <= x <= desk_world.max_point[0, 0]
    assert desk_world.min_point[0, 1] <= y <= desk_world.max_point[0, 1]
    # Z comes from the desk proxy, not the plate the mug is actually on.
    assert abs(z - float(desk_world.max_point[0, 2] + 0.0 - mug.get_bounding_box().min_point[0, 2])) < 1e-6


def test_anchor_init_on_parent_without_on_falls_back_to_anchor():
    """AnchorInitializer falls back to the first anchor when the parent has no On relation."""
    desk = _make_desk()
    stand = DummyObject(
        name="stand",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.5)),
    )
    stand.add_relation(NextTo(desk, side=Side.POSITIVE_X, distance_m=0.1))
    mug = _make_box("mug", size=0.1, height=0.12)
    mug.add_relation(On(stand, clearance_m=0.0))

    positions = _seed(AnchorInitializer(), [desk, stand, mug], {desk})

    x, y, z = positions[mug]
    desk_world = desk.get_world_bounding_box()
    assert desk_world.min_point[0, 0] <= x <= desk_world.max_point[0, 0]
    assert desk_world.min_point[0, 1] <= y <= desk_world.max_point[0, 1]
    assert abs(z - float(desk_world.max_point[0, 2] + 0.0 - mug.get_bounding_box().min_point[0, 2])) < 1e-6


@pytest.mark.parametrize("initializer_cls", ALL_INITIALIZERS)
def test_on_init_reproducible_with_placement_seed(initializer_cls):
    """Same placement_seed produces identical On-guided init positions across independent runs."""
    solver_params = RelationSolverParams(max_iters=0, save_position_history=False, verbose=False)

    def _run():
        params = ObjectPlacerParams(
            placement_seed=42,
            apply_positions_to_objects=False,
            solver_params=solver_params,
            initializer=initializer_cls(),
        )
        desk = _make_desk()
        box = _make_box("box")
        box.add_relation(On(desk, clearance_m=0.01))
        (result,) = ObjectPlacer(params=params).place([desk, box])
        return next(pos for obj, pos in result.positions.items() if obj.name == "box")

    assert _run() == _run()
