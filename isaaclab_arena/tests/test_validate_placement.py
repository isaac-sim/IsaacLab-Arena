# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPlacer placement validation (_validate_placement, _validate_no_overlap, _validate_on_relations)."""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _make_box(name: str, size: float = 0.2) -> DummyObject:
    half = size / 2
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(-half, -half, -half), max_point=(half, half, half)),
    )


def _make_desk() -> DummyObject:
    return DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.5, -0.5, 0.0), max_point=(0.5, 0.5, 0.05)),
    )


def test_no_overlap_returns_true():
    """Test that two boxes far apart pass validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    assert placer._validate_placement(positions) is True


def test_overlapping_returns_false():
    """Test that two boxes at the same position fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 0.0)}
    assert placer._validate_placement(positions) is False


def test_partial_overlap_returns_false():
    """Test that two boxes with partial 3D overlap fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a", size=0.2)
    b = _make_box("b", size=0.2)
    positions = {a: (0.0, 0.0, 0.0), b: (0.1, 0.1, 0.0)}
    assert placer._validate_placement(positions) is False


def test_separated_in_z_passes():
    """Test that two boxes sharing XY footprint but separated in Z pass validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 5.0)}
    assert placer._validate_placement(positions) is True


def test_object_on_surface_no_overlap():
    """Test that box above desk surface with no 3D overlap passes validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    # Desk top at z=0.05; box at z=0.16 → box occupies z=[0.06, 0.26], clear of desk
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.16)}
    assert placer._validate_placement(positions) is True


def test_colocated_siblings_overlap_rejected():
    """Test that two objects at the same position fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    a = _make_box("a", size=0.2)
    b = _make_box("b", size=0.2)
    positions = {desk: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.15), b: (0.0, 0.0, 0.15)}
    assert placer._validate_placement(positions) is False


def test_overlap_check_separated_returns_true():
    """Test that two separated boxes pass overlap check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    assert placer._validate_no_overlap(positions) is True


def test_overlap_check_overlapping_returns_false():
    """Test that two overlapping boxes fail overlap check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 0.0)}
    assert placer._validate_no_overlap(positions) is False


def test_on_relation_check_no_relation_returns_true():
    """Test that objects with no On relation pass On-relation check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    assert placer._validate_on_relations(positions) is True


def test_on_relation_check_child_inside_xy_z_in_band_passes():
    """Test that child inside parent XY with Z in (parent_top, parent_top+clearance_m] passes On-relation check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05 → valid Z band (0.05, 0.06]
    # Child bottom 0.06 (at upper bound); box half-height 0.1 → center z = 0.16.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.16)}
    assert placer._validate_on_relations(positions) is True


def test_validate_on_relations_child_z_within_clearance_band_passes():
    """Test that child bottom in (parent_top, parent_top+clearance_m] passes On-relation Z check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; valid band (0.05, 0.06]
    # Child bottom 0.055 (inside band); box half-height 0.1 → center z = 0.155.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.155)}
    assert placer._validate_on_relations(positions) is True


def test_validate_on_relations_child_z_above_clearance_fails():
    """Test that child bottom above parent_top + clearance_m fails On-relation Z check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; valid band (0.05, 0.06]
    # Child bottom 1.0 (above 0.06).
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 1.1)}
    assert placer._validate_on_relations(positions) is False


def test_validate_on_relations_child_z_at_or_below_parent_top_fails():
    """Test that child bottom at or below parent top fails On-relation Z check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # Desk top 0.05; child bottom must be strictly above parent top.
    # Child bottom 0.05 (equals parent top).
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.15)}
    assert placer._validate_on_relations(positions) is False


def test_on_relation_check_child_outside_xy_returns_false():
    """Test that child outside parent XY fails On-relation check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))
    positions = {desk: (0.0, 0.0, 0.0), box: (10.0, 10.0, 0.1)}
    assert placer._validate_on_relations(positions) is False
