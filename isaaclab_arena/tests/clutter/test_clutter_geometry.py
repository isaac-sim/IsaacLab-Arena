# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Release constraints and rest acceptance independent of simulation."""

import math
import torch

import pytest


def test_resting_containment_uses_rotated_bounds():
    from isaaclab_arena.offline_placement.clutter_geometry import check_resting_poses, get_placement_region
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters

    support = AxisAlignedBoundingBox((-0.5, -0.2, -0.1), (0.5, 0.2, 0))
    half_yaw = math.radians(90.5) / 2
    rotation = (0, 0, math.sin(half_yaw), math.cos(half_yaw))
    assert quaternion_to_90_deg_z_quarters(rotation) == 1
    region = get_placement_region((0, 0, 1), support, rotation)
    with pytest.raises(AssertionError, match="Only 90°"):
        get_placement_region((0, 0, 1), support, (0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
    child = AxisAlignedBoundingBox((-0.3, -0.1, -0.05), (0.3, 0.1, 0.05))
    bounds = child.rotated_by_quat((0, 0, 2**-0.5, 2**-0.5)).translated(
        torch.tensor([[0, 0, 1.05], [0.15, 0, 1.05], [0, 0, 0.9]])
    )
    verdict = check_resting_poses(bounds, region, containment_margin_m=0.0, fall_through_tolerance_m=0.01)
    assert verdict.diverged == []
    assert verdict.fell_off == [1]
    assert verdict.fell_through == [2]


def test_only_clutter_roots_are_exempt_from_shift_limits():
    from isaaclab_arena.offline_placement.post_physics_validation import PoseShiftValidator
    from isaaclab_arena.offline_placement.settled_batch import SettledBatch
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor
    from isaaclab_arena.relations.validation.types import PlacementValidationResults
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    bounds = AxisAlignedBoundingBox((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05))
    table = DummyObject("table", bounds, relations=[IsAnchor()])
    cube = DummyObject("cube", bounds, relations=[ClutterOn(table)])
    initial = {key: torch.tensor([[0.0, 0, 1, 0, 0, 0, 1]]) for key in ("cube", "neighbor")}
    final = {key: pose.clone() for key, pose in initial.items()}
    final["cube"][0, 2] -= 0.5
    layout = PlacementResult(PlacementValidationResults({}), {cube: (0, 0, 1)}, 0, 1)
    batch = SettledBatch({0: layout}, [0], initial, final, {}, {}, {})
    validator = PoseShiftValidator()
    assert validator.validate(batch)[0].passed
    final["neighbor"][0, 0] += 0.01
    report = validator.validate(batch)[0]
    assert not report.passed
    assert "neighbor" in report.reason
