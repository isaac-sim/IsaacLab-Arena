# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ObjectPlacer placement validation (_validate_geometry, _validate_no_overlap, _validate_on_relations)."""

import math
import torch

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult, ValidationReport
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
    assert placer._validate_geometry(positions, _env_bboxes(positions)).passed is True


def test_overlapping_returns_false():
    """Test that two boxes at the same position fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 0.0)}
    assert placer._validate_geometry(positions, _env_bboxes(positions)).passed is False


def test_partial_overlap_returns_false():
    """Test that two boxes with partial 3D overlap fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a", size=0.2)
    b = _make_box("b", size=0.2)
    positions = {a: (0.0, 0.0, 0.0), b: (0.1, 0.1, 0.0)}
    assert placer._validate_geometry(positions, _env_bboxes(positions)).passed is False


def test_separated_in_z_passes():
    """Test that two boxes sharing XY footprint but separated in Z pass validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 5.0)}
    assert placer._validate_geometry(positions, _env_bboxes(positions)).passed is True


def test_object_on_surface_no_overlap():
    """Test that box above desk surface with no 3D overlap passes validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    box = _make_box("box", size=0.2)
    # Desk top at z=0.05; box at z=0.16 → box occupies z=[0.06, 0.26], clear of desk
    positions = {desk: (0.0, 0.0, 0.0), box: (0.0, 0.0, 0.16)}
    assert placer._validate_geometry(positions, _env_bboxes(positions)).passed is True


def test_colocated_siblings_overlap_rejected():
    """Test that two objects at the same position fail validation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    desk = _make_desk()
    a = _make_box("a", size=0.2)
    b = _make_box("b", size=0.2)
    positions = {desk: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.15), b: (0.0, 0.0, 0.15)}
    assert placer._validate_geometry(positions, _env_bboxes(positions)).passed is False


def test_rotation_aware_overlap_uses_yaw():
    """Test that a long box clears a +Y cube axis-aligned but overlaps it after a 90° conservative rotation."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_long_box("a")  # x in [-0.3, 0.3], y in [-0.05, 0.05]
    b = _make_box("b", size=0.1)
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.2, 0.0)}
    axis_aligned = {a: a.get_bounding_box(), b: b.get_bounding_box()}
    assert placer._validate_geometry(positions, axis_aligned).passed is True
    rotated = {a: a.get_bounding_box().rotated_around_z(math.pi / 2), b: b.get_bounding_box()}
    assert placer._validate_geometry(positions, rotated).passed is False


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
        placer._validate_geometry(positions, ObjectPlacer._get_bounding_boxes_for_candidate_index(rotated, idx)).passed
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
    child.add_relation(On(desk, clearance_m=0.01))
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


def test_validate_geometry_reports_named_checks_for_valid_placement():
    """_validate_geometry should report both named checks passing for a valid placement."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (1.0, 0.0, 0.0)}
    report = placer._validate_geometry(positions, _env_bboxes(positions))
    assert report.checks == {"no_overlap": True, "on_relations": True}
    assert report.passed is True
    assert report.failed_checks == ()


def test_validate_geometry_isolates_the_failing_check():
    """_validate_geometry should fail only no_overlap when objects overlap with relations satisfied."""
    placer = ObjectPlacer(params=ObjectPlacerParams())
    a = _make_box("a")
    b = _make_box("b")
    positions = {a: (0.0, 0.0, 0.0), b: (0.0, 0.0, 0.0)}
    report = placer._validate_geometry(positions, _env_bboxes(positions))
    assert report.checks["no_overlap"] is False
    assert report.checks["on_relations"] is True
    assert report.passed is False
    assert report.failed_checks == ("no_overlap",)


def test_validation_report_passed_and_failed_checks():
    """ValidationReport should derive passed and failed_checks from its checks map."""
    assert ValidationReport(checks={"a": True, "b": True}).passed is True
    mixed = ValidationReport(checks={"a": True, "b": False, "c": False})
    assert mixed.passed is False
    assert mixed.failed_checks == ("b", "c")


def test_validation_report_empty_fails_closed():
    """An empty report should fail closed so an unvalidated layout is never treated as valid."""
    assert ValidationReport(checks={}).passed is False
    assert ValidationReport(checks={}).failed_checks == ()


def test_validation_report_snapshots_caller_dict():
    """Mutating the caller's dict after construction must not change the report."""
    source = {"no_overlap": True}
    report = ValidationReport(checks=source)
    source["no_overlap"] = False
    source["on_relations"] = False
    assert report.passed is True
    assert dict(report.checks) == {"no_overlap": True}


def test_validation_report_checks_are_read_only():
    """checks is a read-only view, so a frozen report can't be mutated in place."""
    import pytest

    report = ValidationReport(checks={"no_overlap": True})
    with pytest.raises(TypeError):
        report.checks["no_overlap"] = False


def test_validation_report_survives_deepcopy():
    """Reports must deepcopy/pickle (Isaac Lab deepcopies the EventTermCfg params that carry them)."""
    import copy

    report = ValidationReport(checks={"no_overlap": True, "on_relations": False})
    clone = copy.deepcopy(report)
    assert dict(clone.checks) == {"no_overlap": True, "on_relations": False}
    assert clone.failed_checks == ("on_relations",)


def test_validation_report_with_check_adds_sibling_without_mutating_original():
    """with_check derives a new report with a further check, leaving the original intact."""
    geometry = ValidationReport(checks={"no_overlap": True, "on_relations": True})
    extended = geometry.with_check("extra_check", False)
    # Original is untouched (immutable); the derived report carries the sibling check.
    assert dict(geometry.checks) == {"no_overlap": True, "on_relations": True}
    assert dict(extended.checks) == {"no_overlap": True, "on_relations": True, "extra_check": False}
    # A failing sibling check flips acceptance, so a custom layout_filter can require it.
    assert geometry.passed is True
    assert extended.passed is False
    assert extended.failed_checks == ("extra_check",)


def test_placement_result_success_is_derived_from_validation():
    """PlacementResult.success should mirror its ValidationReport's passed flag."""
    failing = ValidationReport(checks={"no_overlap": False, "on_relations": True})
    result = PlacementResult(positions={}, final_loss=1.0, attempts=1, validation=failing)
    assert result.success is False
    assert result.validation.failed_checks == ("no_overlap",)
