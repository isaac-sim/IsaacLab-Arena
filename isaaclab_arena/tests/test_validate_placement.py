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


def test_on_relation_check_no_relation_returns_true():
    """Test that objects with no On relation pass On-relation check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    assert placer._validate_on_relations(positions) is True


def test_on_relation_check_child_inside_xy_z_in_band_passes():
    """Test that child inside parent XY with Z in (parent_top, parent_top+clearance_m] passes On-relation check."""
    # Valid Z band (0.05, 0.06]; child_bottom 0.06 is inside band.
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05
    # Child bottom 0.06 (at upper bound); box half-height 0.1 → center z = 0.16.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.16)}
    assert placer._validate_on_relations(positions) is True


def test_validate_on_relations_child_z_above_clearance_fails():
    """Test that child bottom above parent_top + clearance_m fails On-relation Z check."""
    # Valid Z band (0.05, 0.06]; child_bottom 1.0 is above band.
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05
    # Child bottom 1.0 is above band.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 1.1)}
    assert placer._validate_on_relations(positions) is False


def test_validate_on_relations_child_z_within_tolerance_above_clearance_passes():
    """Test that child bottom slightly above parent_top+clearance passes when on_relation_z_tolerance_m is set."""
    # on_relation_z_tolerance_m=5e-3 → valid Z band (0.045, 0.065]; child_bottom 0.063 is inside band.
    placer = ObjectPlacer(params=ObjectPlacerParams(on_relation_z_tolerance_m=5e-3))
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))
    # Child bottom 0.063 → box center z = 0.063 + 0.1 = 0.163.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.163)}
    assert placer._validate_on_relations(positions) is True


def test_validate_on_relations_child_z_at_or_below_parent_top_fails():
    """Test that child bottom at or below parent top fails when on_relation_z_tolerance_m is zero."""
    # on_relation_z_tolerance_m=0 → valid Z band (0.05, 0.06]; child_bottom 0.05 (equals parent_top) is not in band.
    placer = ObjectPlacer(params=ObjectPlacerParams(on_relation_z_tolerance_m=0.0))
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05
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
