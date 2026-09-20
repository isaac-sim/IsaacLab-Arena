# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement initializers that seed the relation solver."""

import torch

import pytest

from isaaclab_arena.relations.initializers.anchor_initializer import AnchorInitializer
from isaaclab_arena.relations.initializers.on_tree_initializer import OnTreeInitializer
from isaaclab_arena.relations.initializers.placement_initializer_base import InitializerType
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, PositionLimitsBox, Side
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

# Both initializers seed anchors, unparented objects, and On children the same way; the tests
# that cover those shared rules run against each.
ALL_INITIALIZERS = [AnchorInitializer, OnTreeInitializer]
_TYPE_BY_CLASS = {AnchorInitializer: InitializerType.ANCHOR, OnTreeInitializer: InitializerType.ON_TREE}


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


def _default_bboxes(objects):
    return {obj: obj.get_bounding_box() for obj in objects}


def _seed(initializer, objects, anchors, asset_to_bbox=None, generator=None):
    return initializer.generate_initial_positions(
        objects, anchors, asset_to_bbox or _default_bboxes(objects), generator
    )


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
        asset_to_bbox={table_set: small_table_bbox, box: box_bbox},
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


def test_anchor_init_on_non_anchor_parent_seeds_against_anchor_ancestor():
    """An object on a non-anchor parent is seeded across the anchor above that parent."""
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
    # Z comes from the desk, not the plate the mug is actually on.
    assert abs(z - float(desk_world.max_point[0, 2] + 0.0 - mug.get_bounding_box().min_point[0, 2])) < 1e-6


def test_anchor_init_walks_past_several_on_levels_to_reach_the_anchor():
    """A chain deeper than one On level still resolves to the anchor at the top of the chain."""
    desk = _make_desk()
    tray = DummyObject(
        name="tray",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.4, 0.4, 0.02)),
    )
    tray.add_relation(On(desk, clearance_m=0.0))
    plate = DummyObject(
        name="plate",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.02)),
    )
    plate.add_relation(On(tray, clearance_m=0.0))
    mug = _make_box("mug", size=0.1, height=0.12)
    mug.add_relation(On(plate, clearance_m=0.0))

    positions = _seed(AnchorInitializer(), [desk, tray, plate, mug], {desk})

    desk_world = desk.get_world_bounding_box()
    _assert_footprint_within(positions[mug], mug.get_bounding_box(), desk_world)
    assert abs(positions[mug][2] - float(desk_world.max_point[0, 2] - mug.get_bounding_box().min_point[0, 2])) < 1e-6


def test_anchor_init_falls_back_when_on_chain_loops():
    """A cycle in the On chain falls back to the first anchor instead of looping forever."""
    desk = _make_desk()
    left = _make_box("left")
    right = _make_box("right")
    left.add_relation(On(right, clearance_m=0.0))
    right.add_relation(On(left, clearance_m=0.0))

    positions = _seed(AnchorInitializer(), [desk, left, right], {desk})

    desk_world = desk.get_world_bounding_box()
    for obj in (left, right):
        _assert_footprint_within(positions[obj], obj.get_bounding_box(), desk_world)


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
            initializer_type=_TYPE_BY_CLASS[initializer_cls],
        )
        desk = _make_desk()
        box = _make_box("box")
        box.add_relation(On(desk, clearance_m=0.01))
        (result,) = ObjectPlacer(params=params).place([desk, box])
        return next(pos for obj, pos in result.positions.items() if obj.name == "box")

    assert _run() == _run()


def test_ontree_init_seeds_child_on_its_real_parent():
    """OnTreeInitializer places a grandchild on the parent's sampled footprint, not the anchor's."""
    desk = _make_desk()
    plate = DummyObject(
        name="plate",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.02)),
    )
    plate.add_relation(On(desk, clearance_m=0.01))
    mug = _make_box("mug", size=0.1, height=0.12)
    mug.add_relation(On(plate, clearance_m=0.0))
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        positions = _seed(OnTreeInitializer(), [desk, plate, mug], {desk}, generator=generator)
        plate_world = plate.get_bounding_box().translated(positions[plate])
        _assert_footprint_within(positions[mug], mug.get_bounding_box(), plate_world)
        # Z sits on the plate's top surface, which is above the desk's.
        expected_z = float(plate_world.max_point[0, 2] - mug.get_bounding_box().min_point[0, 2])
        assert abs(positions[mug][2] - expected_z) < 1e-6


def test_ontree_init_orders_children_after_parents_regardless_of_input_order():
    """A child listed before its parent is still seeded against the parent's sampled footprint."""
    desk = _make_desk()
    plate = DummyObject(
        name="plate",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.02)),
    )
    plate.add_relation(On(desk, clearance_m=0.0))
    mug = _make_box("mug", size=0.1, height=0.12)
    mug.add_relation(On(plate, clearance_m=0.0))

    objects = [mug, plate, desk]
    positions = _seed(OnTreeInitializer(), objects, {desk})

    assert list(positions) == objects, "Initializer must return positions in the caller's object order"
    _assert_footprint_within(
        positions[mug], mug.get_bounding_box(), plate.get_bounding_box().translated(positions[plate])
    )


def test_ontree_init_rejects_on_relation_cycle():
    """A cycle in the On graph is reported rather than silently dropping objects."""
    desk = _make_desk()
    left = _make_box("left")
    right = _make_box("right")
    left.add_relation(On(right))
    right.add_relation(On(left))

    with pytest.raises(AssertionError, match="forest rooted at anchors"):
        _seed(OnTreeInitializer(), [desk, left, right], {desk})


def test_ontree_init_handles_two_disjoint_on_trees():
    """Two independent On trees are each seeded against their own root, not against each other."""
    desk = _make_desk()
    left_tray = DummyObject(
        name="left_tray",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.02)),
    )
    left_tray.add_relation(On(desk, clearance_m=0.0))
    left_mug = _make_box("left_mug", size=0.05, height=0.08)
    left_mug.add_relation(On(left_tray, clearance_m=0.0))

    right_tray = DummyObject(
        name="right_tray",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.3, 0.3, 0.02)),
    )
    right_tray.add_relation(On(desk, clearance_m=0.0))
    right_mug = _make_box("right_mug", size=0.05, height=0.08)
    right_mug.add_relation(On(right_tray, clearance_m=0.0))

    objects = [desk, left_tray, left_mug, right_tray, right_mug]
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        positions = _seed(OnTreeInitializer(), objects, {desk}, generator=generator)
        for tray, mug in ((left_tray, left_mug), (right_tray, right_mug)):
            tray_world = tray.get_bounding_box().translated(positions[tray])
            _assert_footprint_within(positions[mug], mug.get_bounding_box(), tray_world)


def test_ontree_init_narrows_to_position_limits_box():
    """A PositionLimitsBox narrows the sampled interval so the seed starts inside the limits."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    box.add_relation(PositionLimitsBox(x_min=0.1, x_max=0.2, y_min=0.6, y_max=0.7))
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        x, y, _unused_z = _seed(OnTreeInitializer(), [desk, box], {desk}, generator=generator)[box]
        assert 0.1 - 1e-6 <= x <= 0.2 + 1e-6
        assert 0.6 - 1e-6 <= y <= 0.7 + 1e-6


def test_ontree_init_intersects_several_narrowing_constraints():
    """Two PositionLimitsBox relations narrow to the intersection of both, not just the last."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    box.add_relation(PositionLimitsBox(x_min=0.1, x_max=0.5))
    box.add_relation(PositionLimitsBox(x_min=0.3, x_max=0.7))
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        x, _unused_y, _unused_z = _seed(OnTreeInitializer(), [desk, box], {desk}, generator=generator)[box]
        assert 0.3 - 1e-6 <= x <= 0.5 + 1e-6, "Expected the intersection of both limit boxes"


def test_ontree_init_ignores_relations_that_do_not_narrow():
    """A relation with no narrowing support leaves the footprint interval untouched."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    box.add_relation(NextTo(desk, side=Side.POSITIVE_X, distance_m=0.05))
    generator = torch.Generator().manual_seed(0)

    samples = [_seed(OnTreeInitializer(), [desk, box], {desk}, generator=generator)[box] for _ in range(50)]

    desk_world = desk.get_world_bounding_box()
    for position in samples:
        _assert_footprint_within(position, box.get_bounding_box(), desk_world)
    assert len({round(x, 4) for x, _, _ in samples}) > 1, "NextTo must not pin the seed to one point"


def test_ontree_init_clamps_to_footprint_when_bounds_are_unreachable():
    """Bounds that sit off the parent seed the nearest reachable point on the parent instead."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    # Valid origins on the desk span [0.0, 0.8]; these limits sit entirely beyond that.
    box.add_relation(PositionLimitsBox(x_min=2.0, x_max=3.0))

    x, _unused_y, _unused_z = _seed(OnTreeInitializer(), [desk, box], {desk})[box]

    assert abs(x - 0.8) < 1e-6, "Expected the footprint edge closest to the bounds"


def test_anchor_init_is_not_narrowed_by_position_limits():
    """AnchorInitializer keeps its original behaviour and ignores narrowing constraints."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    box.add_relation(PositionLimitsBox(x_min=0.1, x_max=0.2))
    generator = torch.Generator().manual_seed(0)

    samples = [_seed(AnchorInitializer(), [desk, box], {desk}, generator=generator)[box] for _ in range(50)]

    assert any(x > 0.2 + 1e-6 for x, _, _ in samples), "AnchorInitializer must not apply narrowing"
