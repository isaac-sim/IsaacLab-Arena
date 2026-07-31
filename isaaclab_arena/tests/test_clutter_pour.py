# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for planning clutter drop poses into layouts."""

from __future__ import annotations

import torch

import pytest

from isaaclab_arena.relations.clutter_groups import get_clutter_groups
from isaaclab_arena.relations.clutter_pour import plan_clutter_drops, region_above_support
from isaaclab_arena.relations.relations import ClutteredOn, IsAnchor
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


class _Asset:
    """Minimal stand-in exposing only what pour planning reads."""

    def __init__(self, name: str, position=(0.0, 0.0, 0.0)):
        self.name = name
        self.relations: list = []
        self._pose = Pose(position_xyz=position, rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    def add_relation(self, relation) -> None:
        self.relations.append(relation)

    def get_relations(self) -> list:
        return self.relations

    def has_relation(self, relation_type: type) -> bool:
        return any(isinstance(relation, relation_type) for relation in self.relations)

    def get_initial_pose(self) -> Pose:
        return self._pose


class _Layout:
    """Stands in for PlacementResult, which needs validation machinery we do not exercise."""

    def __init__(self):
        self.positions: dict = {}
        self.rotations: dict = {}


def _box(half_x: float, half_y: float, half_z: float) -> AxisAlignedBoundingBox:
    return AxisAlignedBoundingBox(
        min_point=(-half_x, -half_y, -half_z),
        max_point=(half_x, half_y, half_z),
    )


def _scene(member_count: int, support_position=(0.0, 0.0, 0.0), **relation_kwargs):
    support = _Asset("table", position=support_position)
    support.add_relation(IsAnchor())
    members = []
    for index in range(member_count):
        member = _Asset(f"item_{index}")
        member.add_relation(ClutteredOn(support, group="tools", **relation_kwargs))
        members.append(member)

    bounding_boxes = {support: _box(0.5, 0.5, 0.1)}
    for member in members:
        bounding_boxes[member] = _box(0.03, 0.03, 0.04)
    return support, members, bounding_boxes


def test_region_sits_on_the_support_top_face():
    support_bbox = _box(0.5, 0.4, 0.1)
    region = region_above_support((1.0, 2.0, 0.75), support_bbox)

    assert region.min_x == pytest.approx(0.5)
    assert region.max_x == pytest.approx(1.5)
    assert region.min_y == pytest.approx(1.6)
    assert region.max_y == pytest.approx(2.4)
    # Floor is the support's top surface, not its origin.
    assert region.floor_z == pytest.approx(0.85)


def test_spread_shrinks_the_region_about_its_centre():
    support_bbox = _box(0.5, 0.5, 0.1)
    full = region_above_support((0.0, 0.0, 0.0), support_bbox)
    half = region_above_support((0.0, 0.0, 0.0), support_bbox, spread=0.5)

    assert (half.max_x - half.min_x) == pytest.approx(0.5 * (full.max_x - full.min_x))
    assert (half.max_x + half.min_x) == pytest.approx(full.max_x + full.min_x)
    assert half.floor_z == pytest.approx(full.floor_z)


def test_every_member_receives_a_drop_pose():
    support, members, bounding_boxes = _scene(6)
    layout = _Layout()
    groups = get_clutter_groups([support, *members])

    plan_clutter_drops(layout, groups, bounding_boxes, torch.Generator().manual_seed(0))

    assert set(layout.positions) == set(members)
    assert set(layout.rotations) == set(members)
    for member in members:
        assert len(layout.rotations[member]) == 4


def test_drops_start_above_the_support_surface():
    support, members, bounding_boxes = _scene(8, support_position=(0.0, 0.0, 0.75))
    layout = _Layout()
    groups = get_clutter_groups([support, *members])

    plan_clutter_drops(layout, groups, bounding_boxes, torch.Generator().manual_seed(0))

    support_top = 0.75 + 0.1
    for member in members:
        assert layout.positions[member][2] > support_top


def test_drops_stay_within_the_support_footprint():
    support, members, bounding_boxes = _scene(10)
    layout = _Layout()
    groups = get_clutter_groups([support, *members])

    plan_clutter_drops(layout, groups, bounding_boxes, torch.Generator().manual_seed(0))

    for member in members:
        x, y, _ = layout.positions[member]
        assert -0.5 <= x <= 0.5
        assert -0.5 <= y <= 0.5


def test_same_seed_reproduces_the_same_pile():
    support, members, bounding_boxes = _scene(8)
    groups = get_clutter_groups([support, *members])

    first, second = _Layout(), _Layout()
    plan_clutter_drops(first, groups, bounding_boxes, torch.Generator().manual_seed(7))
    plan_clutter_drops(second, groups, bounding_boxes, torch.Generator().manual_seed(7))

    assert first.positions == second.positions
    assert first.rotations == second.rotations


def test_different_seeds_produce_different_piles():
    support, members, bounding_boxes = _scene(8)
    groups = get_clutter_groups([support, *members])

    first, second = _Layout(), _Layout()
    plan_clutter_drops(first, groups, bounding_boxes, torch.Generator().manual_seed(1))
    plan_clutter_drops(second, groups, bounding_boxes, torch.Generator().manual_seed(2))

    assert first.positions != second.positions


def test_two_groups_on_one_support_are_both_poured():
    support = _Asset("table")
    support.add_relation(IsAnchor())
    left = _Asset("left")
    right = _Asset("right")
    left.add_relation(ClutteredOn(support, group="left_pile"))
    right.add_relation(ClutteredOn(support, group="right_pile"))

    bounding_boxes = {support: _box(0.5, 0.5, 0.1), left: _box(0.03, 0.03, 0.04), right: _box(0.03, 0.03, 0.04)}
    layout = _Layout()
    plan_clutter_drops(
        layout, get_clutter_groups([support, left, right]), bounding_boxes, torch.Generator().manual_seed(0)
    )

    assert set(layout.positions) == {left, right}


def test_missing_member_bounding_box_is_rejected():
    support, members, bounding_boxes = _scene(3)
    del bounding_boxes[members[1]]
    groups = get_clutter_groups([support, *members])

    with pytest.raises(AssertionError, match="without bounding boxes"):
        plan_clutter_drops(_Layout(), groups, bounding_boxes, torch.Generator().manual_seed(0))


def test_missing_support_bounding_box_is_rejected():
    support, members, bounding_boxes = _scene(3)
    del bounding_boxes[support]
    groups = get_clutter_groups([support, *members])

    with pytest.raises(AssertionError, match="no bounding box"):
        plan_clutter_drops(_Layout(), groups, bounding_boxes, torch.Generator().manual_seed(0))


def test_solved_support_position_is_preferred_over_the_declared_one():
    support, members, bounding_boxes = _scene(4, support_position=(0.0, 0.0, 0.0))
    groups = get_clutter_groups([support, *members])

    layout = _Layout()
    # The solver moved the support; the pile must follow it rather than its declared pose.
    layout.positions[support] = (3.0, 0.0, 0.0)
    plan_clutter_drops(layout, groups, bounding_boxes, torch.Generator().manual_seed(0))

    for member in members:
        assert 2.5 <= layout.positions[member][0] <= 3.5


def test_oversized_member_is_rejected_rather_than_placed_outside():
    support = _Asset("table")
    support.add_relation(IsAnchor())
    huge = _Asset("huge")
    huge.add_relation(ClutteredOn(support, group="tools"))
    bounding_boxes = {support: _box(0.1, 0.1, 0.05), huge: _box(0.5, 0.5, 0.05)}

    with pytest.raises(AssertionError, match="does not fit"):
        plan_clutter_drops(
            _Layout(), get_clutter_groups([support, huge]), bounding_boxes, torch.Generator().manual_seed(0)
        )


def test_pour_avoids_an_object_already_on_the_support():
    """A solver-placed object on the same surface must not be dropped into."""
    from isaaclab_arena.relations.clutter_pour import occupied_footprints_in_region, region_above_support

    support, members, bounding_boxes = _scene(6)
    resident = _Asset("resident")
    bounding_boxes[resident] = _box(0.12, 0.12, 0.1)

    layout = _Layout()
    layout.positions[resident] = (0.0, 0.0, 0.2)
    groups = get_clutter_groups([support, *members])

    region = region_above_support((0.0, 0.0, 0.0), bounding_boxes[support])
    occupied = occupied_footprints_in_region(region, layout.positions, bounding_boxes, exclude={support})
    assert len(occupied) == 1, "the resident object should be seen as occupying the surface"

    plan_clutter_drops(layout, groups, bounding_boxes, torch.Generator().manual_seed(0))

    resident_top = 0.2 + 0.05
    for member in members:
        x, y, z = layout.positions[member]
        overlaps = abs(x - 0.0) < 0.06 + 0.03 and abs(y - 0.0) < 0.06 + 0.03
        if overlaps:
            assert z > resident_top, f"{member.name} was dropped into the resident object at z={z:.3f}"


def test_objects_below_the_support_surface_are_ignored():
    """Something under the table is not in the way of a pile poured on top of it."""
    from isaaclab_arena.relations.clutter_pour import occupied_footprints_in_region, region_above_support

    support, _members, bounding_boxes = _scene(2)
    underneath = _Asset("underneath")
    bounding_boxes[underneath] = _box(0.1, 0.1, 0.05)

    region = region_above_support((0.0, 0.0, 0.5), bounding_boxes[support])
    positions = {underneath: (0.0, 0.0, 0.0)}
    assert occupied_footprints_in_region(region, positions, bounding_boxes, exclude=set()) == []


def test_objects_outside_the_region_are_ignored():
    from isaaclab_arena.relations.clutter_pour import occupied_footprints_in_region, region_above_support

    support, _members, bounding_boxes = _scene(2)
    far_away = _Asset("far_away")
    bounding_boxes[far_away] = _box(0.1, 0.1, 0.1)

    region = region_above_support((0.0, 0.0, 0.0), bounding_boxes[support])
    positions = {far_away: (5.0, 0.0, 0.2)}
    assert occupied_footprints_in_region(region, positions, bounding_boxes, exclude=set()) == []
