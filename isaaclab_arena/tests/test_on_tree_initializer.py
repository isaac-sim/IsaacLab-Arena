# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OnTreeInitializer, which seeds each object against its own sampled parent."""

import torch

import pytest

from isaaclab_arena.relations.initializers.on_tree_initializer import (
    OnTreeInitializer,
    _bounding_box_from_partial_limits,
)
from isaaclab_arena.relations.relations import NextTo, On, PositionLimitsBox, Side
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.tests.test_object_placer_init import _assert_footprint_within, _make_box, _make_desk, _seed
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


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


def test_partial_limits_leave_unset_axes_unbounded():
    """None on a side means unbounded there, so intersecting leaves that axis alone."""
    partial = _bounding_box_from_partial_limits((0.3, None, None), (0.4, None, None))
    footprint = AxisAlignedBoundingBox(min_point=(0.0, 0.1, 0.2), max_point=(1.0, 0.9, 0.8))

    narrowed = footprint.intersected(partial)

    assert narrowed.min_point[0].tolist() == pytest.approx([0.3, 0.1, 0.2])
    assert narrowed.max_point[0].tolist() == pytest.approx([0.4, 0.9, 0.8])


def test_partial_limits_with_nothing_set_is_the_intersection_identity():
    """A fully unset box narrows nothing, which is how the fold starts."""
    unbounded = _bounding_box_from_partial_limits((None, None, None), (None, None, None))
    footprint = AxisAlignedBoundingBox(min_point=(0.0, 0.1, 0.2), max_point=(1.0, 0.9, 0.8))

    narrowed = footprint.intersected(unbounded)

    assert narrowed.min_point[0].tolist() == footprint.min_point[0].tolist()
    assert narrowed.max_point[0].tolist() == footprint.max_point[0].tolist()


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


def test_ontree_init_keeps_reachable_axes_when_another_axis_is_unreachable():
    """An unsatisfiable axis drops only itself, not the narrowing on the others.

    ``On`` pins Z to one value, so a relation that also bounds Z empties that axis. Dropping the
    whole box then would silently discard the X and Y narrowing with it.
    """
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    # X is satisfiable on the desk; Z cannot be, because On fixes the box's Z at the desk top.
    box.add_relation(PositionLimitsBox(x_min=0.1, x_max=0.2, z_min=0.5, z_max=1.5))

    generator = torch.Generator()
    for seed in range(8):
        generator.manual_seed(seed)
        x, _unused_y, _unused_z = _seed(OnTreeInitializer(), [desk, box], {desk}, generator=generator)[box]
        assert 0.1 <= x <= 0.2, "Expected the X narrowing to survive the unreachable Z bound"


def test_ontree_init_ignores_bounds_that_are_unreachable_on_the_parent():
    """Bounds no position on the parent can satisfy are dropped rather than pulled towards."""
    desk = _make_desk()
    box = _make_box("box")
    box.add_relation(On(desk, clearance_m=0.0))
    # Valid origins on the desk span [0.0, 0.8]; these limits sit entirely beyond that.
    box.add_relation(PositionLimitsBox(x_min=2.0, x_max=3.0))

    x, _unused_y, _unused_z = _seed(OnTreeInitializer(), [desk, box], {desk})[box]

    assert 0.0 <= x <= 0.8, "Expected seeding on the parent, as if the limits were absent"
