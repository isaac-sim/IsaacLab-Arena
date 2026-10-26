# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OnTreeInitializer, which seeds each object against its own sampled parent."""

import torch

import pytest

from isaaclab_arena.relations.initializers.on_tree_initializer import (
    RELATION_TO_AABB_STRATEGIES,
    OnTreeInitializer,
    RelationToAABBStrategy,
    _bounding_box_from_partial_limits,
    _order_parents_before_children,
)
from isaaclab_arena.relations.relations import AlignWith, NextTo, On, PositionLimitsBox, Relation, Side
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.tests.test_object_placer_init import _assert_footprint_within, _make_box, _make_desk, _seed
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


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


def test_ontree_samples_across_alignment_band_of_moving_reference():
    """Moving references are sampled first, retaining spread throughout the alignment band."""
    axis, offset_m = "x", 0.1
    desk = _make_desk()
    reference = _make_box("reference", size=0.2)
    reference.add_relation(On(desk))
    reference.add_relation(PositionLimitsBox(x_min=0.3, x_max=0.4, y_min=0.3, y_max=0.4))
    subject = DummyObject("subject", AxisAlignedBoundingBox((-0.05, 0.02, 0.0), (0.15, 0.12, 0.1)))
    subject.add_relation(On(desk))
    subject.add_relation(AlignWith(reference, axis, offset_m=offset_m, tolerance_m=0.04))
    objects = [subject, desk, reference]
    generator = torch.Generator().manual_seed(42)
    errors = []
    unconstrained = []
    axis_index = "xy".index(axis)
    for _ in range(100):
        positions = _seed(OnTreeInitializer(), objects, {desk}, generator=generator)
        assert list(positions) == objects
        subject_center_W = torch.tensor(positions[subject]) + subject.get_bounding_box().center[0]
        reference_center_W = torch.tensor(positions[reference]) + reference.get_bounding_box().center[0]
        error = (subject_center_W - reference_center_W)[axis_index].item() - offset_m
        assert abs(error) <= 0.040001
        _assert_footprint_within(positions[subject], subject.get_bounding_box(), desk.get_world_bounding_box())
        errors.append(error)
        unconstrained.append(positions[subject][1 - axis_index])

    # Check that seeding fills the interior rather than collapsing at the centre or one edge.
    assert min(errors) < -0.032 and max(errors) > 0.032
    assert torch.tensor(errors).std().item() > 0.018
    assert max(unconstrained) - min(unconstrained) > 0.5


def test_ontree_intersects_alignment_bands_with_position_box():
    """Multiple references and a position box jointly narrow the sampling interval."""
    desk = _make_desk()
    left = _make_box("left", size=0.1)
    right = _make_box("right", size=0.1)
    subject = _make_box("subject", size=0.1)
    for reference, x in ((left, 0.4), (right, 0.5)):
        reference.add_relation(On(desk))
        reference.add_relation(PositionLimitsBox(x_min=x, x_max=x + 0.02, y_min=0.6, y_max=0.62))
    subject.add_relation(On(desk))
    subject.add_relation(AlignWith(left, "x", tolerance_m=0.1))
    subject.add_relation(AlignWith(right, "x", tolerance_m=0.1))
    subject.add_relation(AlignWith(left, "y", tolerance_m=0.05))
    subject.add_relation(PositionLimitsBox(x_min=0.42))

    generator = torch.Generator().manual_seed(42)
    for _ in range(30):
        positions = _seed(OnTreeInitializer(), [subject, right, desk, left], {desk}, generator=generator)
        x, y, _ = positions[subject]
        assert x >= 0.42 - 1e-6
        assert abs(x - positions[left][0]) <= 0.100001
        assert abs(x - positions[right][0]) <= 0.100001
        assert abs(y - positions[left][1]) <= 0.050001


def test_ontree_unreachable_alignment_axis_keeps_other_narrowing():
    """An impossible Z alignment keeps both On support and feasible X/Y sampling bands."""
    desk = _make_desk()
    reference = _make_box("reference")
    reference.add_relation(On(desk))
    reference.add_relation(PositionLimitsBox(x_min=0.4, x_max=0.41))
    subject = _make_box("subject")
    subject.add_relation(On(desk))
    subject.add_relation(AlignWith(reference, "x", tolerance_m=0.03))
    subject.add_relation(AlignWith(reference, "z", offset_m=1.0, tolerance_m=0.03))
    subject.add_relation(PositionLimitsBox(y_min=0.2, y_max=0.3))

    positions = _seed(OnTreeInitializer(), [subject, desk, reference], {desk})

    x, y, z = positions[subject]
    assert abs(x - positions[reference][0]) <= 0.030001
    assert 0.2 <= y <= 0.3
    assert z == pytest.approx(positions[reference][2])


def test_ontree_aligns_to_translated_anchor_with_zero_tolerance():
    """An anchored reference uses its world centre, including its initial translation."""
    desk = _make_desk()
    desk.set_initial_pose(Pose(position_xyz=(3.0, -2.0, 1.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    subject = _make_box("subject")
    subject.add_relation(On(desk))
    subject.add_relation(AlignWith(desk, "x", offset_m=0.1, tolerance_m=0.0))

    positions = _seed(OnTreeInitializer(), [subject, desk], {desk})

    assert positions[subject][0] == pytest.approx(3.5)
    _assert_footprint_within(positions[subject], subject.get_bounding_box(), desk.get_world_bounding_box())


class _MatchReferenceX(Relation):
    """Test relation whose origin uses its reference's world minimum X."""


class _MatchReferenceXToAABBStrategy(RelationToAABBStrategy):
    def get_dependencies(self, relation):
        return (relation.parent,)

    def get_bounds(self, relation, subject_bbox, sampled_world_bboxes):
        x = sampled_world_bboxes[relation.parent].min_point[0, 0].item()
        return _bounding_box_from_partial_limits((x, None, None), (x, None, None))


def test_ontree_orders_registered_dependencies_across_levels_and_cycles(monkeypatch):
    """Use generic references across levels, breaking cycles without breaking support order."""
    monkeypatch.setitem(RELATION_TO_AABB_STRATEGIES, _MatchReferenceX, _MatchReferenceXToAABBStrategy())
    desk = _make_desk()
    tray = _make_box("tray", size=0.5)
    child = _make_box("child", size=0.1)
    middle = _make_box("middle", size=0.1)
    subject = _make_box("subject", size=0.1)
    extra = _make_box("extra", size=0.1)
    distracting = _make_box("distracting", size=0.1)
    for obj in (tray, middle, subject, extra, distracting):
        obj.add_relation(On(desk))
    child.add_relation(On(tray))
    tray.add_relation(AlignWith(middle, "x"))
    for obj in (middle, subject, extra):
        obj.add_relation(_MatchReferenceX(child))
    distracting.add_relation(AlignWith(middle, "x"))
    distracting.add_relation(AlignWith(subject, "y"))
    child.add_relation(NextTo(extra, side=Side.POSITIVE_X))  # No sampling strategy: must not block child.
    objects = [distracting, middle, subject, extra, child, tray, desk]

    # All eligible objects initially wait for references. Distracting has two missing
    # dependencies; tray has one and opens a subtree needed by three waiting objects.
    # Once tray is sampled, child must precede its shallower-level dependents.
    assert _order_parents_before_children(objects, {desk}) == [desk, tray, child, middle, subject, distracting, extra]
    positions = _seed(OnTreeInitializer(), objects, {desk}, generator=torch.Generator().manual_seed(42))
    repeated = _seed(OnTreeInitializer(), objects, {desk}, generator=torch.Generator().manual_seed(42))
    assert positions == repeated
    assert list(positions) == objects
    for obj in (middle, subject, extra):
        assert positions[obj][0] == pytest.approx(positions[child][0])
    for obj in objects:
        if obj is not desk:
            support = tray if obj is child else desk
            support_world = support.get_bounding_box().translated(positions[support])
            _assert_footprint_within(positions[obj], obj.get_bounding_box(), support_world)
