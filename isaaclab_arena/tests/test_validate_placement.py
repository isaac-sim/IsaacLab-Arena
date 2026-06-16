# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPlacer placement validation (_validate_placement, _validate_no_overlap, _validate_on_relations)."""

import math
import torch

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import On, RotateAroundSolution
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _make_box(name: str, size: float = 0.2) -> DummyObject:
    half = size / 2
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(-half, -half, -half), max_point=(half, half, half)),
    )


def _make_long_box(name: str, half_x: float = 0.3, half_y: float = 0.05, half_z: float = 0.05) -> DummyObject:
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=(-half_x, -half_y, -half_z), max_point=(half_x, half_y, half_z)),
    )


def _make_desk() -> DummyObject:
    return DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.5, -0.5, 0.0), max_point=(0.5, 0.5, 0.05)),
    )


def _env_bboxes(positions: dict[DummyObject, tuple[float, float, float]]):
    return {obj: obj.get_bounding_box() for obj in positions}


def _stack_rows(bbox: AxisAlignedBoundingBox, n: int) -> AxisAlignedBoundingBox:
    """Repeat a single-env bbox into n stacked rows (one per candidate)."""
    return AxisAlignedBoundingBox(min_point=bbox.min_point.repeat(n, 1), max_point=bbox.max_point.repeat(n, 1))


def test_no_overlap_returns_true():
    """Test that two boxes far apart pass validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    assert (
        placer._validate_placement(positions, _env_bboxes(positions)).do_all_required_validation_checks_pass() is True
    )


def test_overlapping_returns_false():
    """Test that two boxes at the same position fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 0.0)}
    assert (
        placer._validate_placement(positions, _env_bboxes(positions)).do_all_required_validation_checks_pass() is False
    )


def test_partial_overlap_returns_false():
    """Test that two boxes with partial 3D overlap fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a", size=0.2)
    b = _make_box("b", size=0.2)
    positions = {a: (0.0, 0.0, 0.0), b: (0.1, 0.1, 0.0)}
    assert (
        placer._validate_placement(positions, _env_bboxes(positions)).do_all_required_validation_checks_pass() is False
    )


def test_separated_in_z_passes():
    """Test that two boxes sharing XY footprint but separated in Z pass validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 5.0)}
    assert (
        placer._validate_placement(positions, _env_bboxes(positions)).do_all_required_validation_checks_pass() is True
    )


def test_object_on_surface_no_overlap():
    """Test that box above desk surface with no 3D overlap passes validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    # Desk top at z=0.05; box at z=0.16 → box occupies z=[0.06, 0.26], clear of desk
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.16)}
    assert (
        placer._validate_placement(positions, _env_bboxes(positions)).do_all_required_validation_checks_pass() is True
    )


def test_colocated_siblings_overlap_rejected():
    """Test that two objects at the same position fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    a = _make_box("a", size=0.2)
    b = _make_box("b", size=0.2)
    positions = {desk: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.15), b: (0.0, 0.0, 0.15)}
    assert (
        placer._validate_placement(positions, _env_bboxes(positions)).do_all_required_validation_checks_pass() is False
    )


def test_rotation_aware_overlap_uses_yaw():
    """Test that a long box clears a +Y cube axis-aligned but overlaps it after a 90° conservative rotation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_long_box("a")  # x in [-0.3, 0.3], y in [-0.05, 0.05]
    b = _make_box("b", size=0.1)
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.2, 0.0)}
    axis_aligned = {a: a.get_bounding_box(), b: b.get_bounding_box()}
    assert placer._validate_placement(positions, axis_aligned).do_all_required_validation_checks_pass() is True
    rotated = {a: a.get_bounding_box().rotated_around_z(math.pi / 2), b: b.get_bounding_box()}
    assert placer._validate_placement(positions, rotated).do_all_required_validation_checks_pass() is False


def test_candidate_bbox_aligns_with_candidate_yaw():
    """Per-candidate yaw, bbox row, and validation stay index-aligned: only the matched row is collision-free."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_long_box("a")  # long in X
    b = _make_box("b", size=0.1)
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.2, 0.0)}

    # Two candidates share positions but assign distinct yaws to `a`.
    candidate_bboxes = {a: _stack_rows(a.get_bounding_box(), 2), b: _stack_rows(b.get_bounding_box(), 2)}
    rotated = ObjectPlacer._rotate_candidate_bboxes([a, b], candidate_bboxes, [{a: 0.0}, {a: math.pi / 2}])

    # Mirrors _place_ranked: each candidate validates against its own bbox row.
    validations = [
        placer._validate_placement(
            positions, ObjectPlacer._get_bounding_boxes_for_candidate_index(rotated, idx)
        ).do_all_required_validation_checks_pass()
        for idx in range(2)
    ]
    # Axis-aligned `a` clears b; rotated 90° it sweeps into b. A row/candidate swap would flip both.
    assert validations == [True, False]


def test_rotate_candidate_bboxes_encloses_marker_plus_sampled_yaw():
    """_rotate_candidate_bboxes folds the marker yaw into the box, not just the sampled yaw."""
    box = _make_long_box("box")
    marker_yaw, sampled_yaw = math.pi / 6, math.pi / 3
    box.add_relation(RotateAroundSolution(yaw_rad=marker_yaw))

    rotated = ObjectPlacer._rotate_candidate_bboxes([box], {box: box.get_bounding_box()}, [{box: sampled_yaw}])

    expected = box.get_bounding_box().rotated_around_z(marker_yaw + sampled_yaw)
    torch.testing.assert_close(rotated[box].min_point, expected.min_point, atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated[box].max_point, expected.max_point, atol=1e-6, rtol=0)
    # Dropping the marker (sampled yaw only) would enclose an undersized, misaligned footprint.
    sampled_only = box.get_bounding_box().rotated_around_z(sampled_yaw)
    assert not torch.allclose(rotated[box].max_point, sampled_only.max_point, atol=1e-6)


def test_on_relation_containment_uses_rotated_bbox():
    """Test that a child fits the parent rim axis-aligned but spills past it once yaw-inflated 90°."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()  # XY in [-0.5, 0.5]
    child = _make_long_box("child")  # x in [-0.3, 0.3], y in [-0.05, 0.05]
    child.add_relation(On(desk, clearance_m=0.01, edge_margin_m=0.0))
    # Near the +Y rim: axis-aligned half-Y 0.05 stays inside; rotated 90° half-Y 0.3 spills past +0.5.
    positions = {desk: (0.0, 0.0, 0.0), child: (0.0, 0.44, 0.105)}

    axis_aligned = {desk: desk.get_bounding_box(), child: child.get_bounding_box()}
    assert placer._validate_on_relations(positions, axis_aligned) is True
    rotated = {desk: desk.get_bounding_box(), child: child.get_bounding_box().rotated_around_z(math.pi / 2)}
    assert placer._validate_on_relations(positions, rotated) is False


def test_on_relation_check_no_relation_returns_true():
    """Test that objects with no On relation pass On-relation check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    assert placer._validate_on_relations(positions, _env_bboxes(positions)) is True


def test_on_relation_check_child_inside_xy_z_in_band_passes():
    """Test that child inside parent XY with Z in (parent_top, parent_top+clearance_m] passes On-relation check."""
    # Valid Z band (0.05, 0.06]; child_bottom 0.06 is inside band.
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05
    # Child bottom 0.06 (at upper bound); box half-height 0.1 → center z = 0.16.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.16)}
    assert placer._validate_on_relations(positions, _env_bboxes(positions)) is True


def test_validate_on_relations_child_z_above_clearance_fails():
    """Test that child bottom above parent_top + clearance_m fails On-relation Z check."""
    # Valid Z band (0.05, 0.06]; child_bottom 1.0 is above band.
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05
    # Child bottom 1.0 is above band.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 1.1)}
    assert placer._validate_on_relations(positions, _env_bboxes(positions)) is False


def test_validate_on_relations_child_z_within_tolerance_above_clearance_passes():
    """Test that child bottom slightly above parent_top+clearance passes when on_relation_z_tolerance_m is set."""
    # on_relation_z_tolerance_m=5e-3 → valid Z band (0.045, 0.065]; child_bottom 0.063 is inside band.
    placer = ObjectPlacer(params=ObjectPlacerParams(on_relation_z_tolerance_m=5e-3))
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))
    # Child bottom 0.063 → box center z = 0.063 + 0.1 = 0.163.
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.163)}
    assert placer._validate_on_relations(positions, _env_bboxes(positions)) is True


def test_validate_on_relations_child_z_at_or_below_parent_top_fails():
    """Test that child bottom at or below parent top fails when on_relation_z_tolerance_m is zero."""
    # on_relation_z_tolerance_m=0 → valid Z band (0.05, 0.06]; child_bottom 0.05 (equals parent_top) is not in band.
    placer = ObjectPlacer(params=ObjectPlacerParams(on_relation_z_tolerance_m=0.0))
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))  # clearance_m=0.01; desk top 0.05
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.15)}
    assert placer._validate_on_relations(positions, _env_bboxes(positions)) is False


def test_on_relation_check_child_outside_xy_returns_false():
    """Test that child outside parent XY fails On-relation check."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk))
    positions = {desk: (0.0, 0.0, 0.0), box: (10.0, 10.0, 0.1)}
    assert placer._validate_on_relations(positions, _env_bboxes(positions)) is False


def _validate_box_on_desk(edge_margin_m: float, box_x: float) -> bool:
    """Validate a 0.2m box at (box_x, 0) on a desk (XY in [-0.5, 0.5]) under the given edge margin."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    box.add_relation(On(desk, edge_margin_m=edge_margin_m))
    positions = {desk: (0.0, 0.0, 0.0), box: (box_x, 0.0, 0.16)}
    return placer._validate_on_relations(positions, _env_bboxes(positions))


def test_on_relation_edge_margin_within_inset_band_passes():
    # Box right edge 0.3 + 0.1 == rim 0.5 - margin 0.1: on the inset bound, still inside.
    assert _validate_box_on_desk(edge_margin_m=0.1, box_x=0.3) is True


def test_on_relation_edge_margin_inside_rim_but_in_margin_gap_fails():
    # Box right edge 0.45 is inside the rim but past the inset bound 0.4 (in the margin gap).
    assert _validate_box_on_desk(edge_margin_m=0.1, box_x=0.35) is False


def test_on_relation_edge_margin_too_large_for_surface_rejected():
    # Desk free span 0.8 caps the margin at 0.4; 0.5 inverts the inset band so containment fails.
    assert _validate_box_on_desk(edge_margin_m=0.5, box_x=0.0) is False
