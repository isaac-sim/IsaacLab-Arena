# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Release constraints and rest acceptance independent of simulation."""

import math
import torch

import pytest


def test_resting_containment_uses_rotated_bounds():
    from isaaclab_arena.offline_placement.clutter_validators import check_resting_poses
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters

    support = AxisAlignedBoundingBox((-0.5, -0.2, -0.1), (0.5, 0.2, 0))
    half_yaw = math.radians(90.5) / 2
    rotation = (0, 0, math.sin(half_yaw), math.cos(half_yaw))
    assert quaternion_to_90_deg_z_quarters(rotation) == 1
    support_bounds = support.rotated_90_around_z(quaternion_to_90_deg_z_quarters(rotation)).translated((0, 0, 1))
    with pytest.raises(AssertionError, match="Only 90°"):
        quaternion_to_90_deg_z_quarters((0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
    child = AxisAlignedBoundingBox((-0.3, -0.1, -0.05), (0.3, 0.1, 0.05))
    bounds = child.rotated_by_quat((0, 0, 2**-0.5, 2**-0.5)).translated(
        torch.tensor([[0, 0, 1.05], [0.15, 0, 1.05], [0, 0, 0.9]])
    )
    verdict = check_resting_poses(bounds, support_bounds, containment_margin_m=0.0, fall_through_tolerance_m=0.01)
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


def test_clutter_preparation_is_noop_without_clutter():
    from isaaclab_arena.offline_placement.clutter_preparation import prepare_clutter_settling
    from isaaclab_arena.relations.relations import RequiresReachability
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    target = DummyObject("target", AxisAlignedBoundingBox((-1, -1, -1), (1, 1, 1)), relations=[RequiresReachability()])
    # No scene access or clutter-specific reachability restriction applies to an ordinary scene.
    prepare_clutter_settling(None, [target])


def test_container_height_keeps_footprint_and_fall_through_checks():
    from isaaclab_arena.offline_placement.clutter_validators import check_resting_poses
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    support = AxisAlignedBoundingBox((-0.5, -0.2, -0.1), (0.5, 0.2, 0.2))
    support_bounds = support.rotated_90_around_z(1).translated((1, 2, 3))
    torch.testing.assert_close(support_bounds.min_point[0, :2], torch.tensor([0.8, 1.5]))
    torch.testing.assert_close(support_bounds.max_point[0, :2], torch.tensor([1.2, 2.5]))
    child = AxisAlignedBoundingBox((-0.01, -0.01, 0), (0.01, 0.01, 0.02))
    bounds = child.translated(torch.tensor([[1, 2, 3.03], [1, 2, 2.99], [1.25, 2, 3.03]]))
    result = check_resting_poses(
        bounds,
        support_bounds,
        containment_margin_m=0,
        fall_through_tolerance_m=0.001,
        minimum_resting_height_m=3.02,
    )
    assert result.fell_through == [1]
    assert result.fell_off == [2]
    assert not result.diverged
    # Without an override, objects below the rim fail the default top-surface check.
    default = check_resting_poses(bounds, support_bounds, containment_margin_m=0, fall_through_tolerance_m=0.001)
    assert default.fell_through == [0, 1, 2]


@pytest.mark.parametrize("height", [0.1, 0.7])
def test_minimum_resting_height_accepts_both_bounds(height):
    from types import SimpleNamespace

    from isaaclab_arena.offline_placement.clutter_validators import SupportContainmentValidator
    from isaaclab_arena.offline_placement.settled_batch import SettledBatch, SettledGeometry
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.relations import ClutterOn, IsAnchor
    from isaaclab_arena.relations.validation.types import PlacementCheck, PlacementValidationResults
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    support_bounds = AxisAlignedBoundingBox((-0.5, -0.5, 0.1), (0.5, 0.5, 0.7))
    support_pose = Pose((1, 2, 3))
    support = DummyObject("support", support_bounds, initial_pose=support_pose, relations=[IsAnchor()])
    child_bounds = AxisAlignedBoundingBox((-0.01, -0.01, 0), (0.01, 0.01, 0.02))
    child = DummyObject("child", child_bounds, relations=[ClutterOn(support)])
    validator = SupportContainmentValidator(minimum_resting_heights_m={"support": height})
    env = SimpleNamespace(arena_world=SimpleNamespace(get_aabb_in_local_frame=lambda key: support_bounds))
    validator.validate_scene(env, [child, support])
    checks = PlacementValidationResults({PlacementCheck.NO_OVERLAP: True, PlacementCheck.CLUTTER_ON_RELATION: True})
    layout = PlacementResult(checks, {child: (1, 2, 3 + height)}, 0, 1)
    batch = SettledBatch({0: layout}, [0], {}, {}, {}, {}, {})
    support_pose_e = support_pose.to_tensor("cpu").unsqueeze(0)
    child_pose_e = Pose((1, 2, 3 + height)).to_tensor("cpu").unsqueeze(0)
    batch.geometry = {
        "support": SettledGeometry(support_pose_e, support_pose_e, support_bounds),
        "child": SettledGeometry(child_pose_e.clone(), child_pose_e, child_bounds),
    }
    report = validator.validate(batch)[0]
    assert report.passed, report.reason
    child_pose_e[0, 2] -= 0.02
    assert not validator.validate(batch)[0].passed
    validator.minimum_resting_heights_m["support"] = height + (-0.01 if height == 0.1 else 0.01)
    with pytest.raises(AssertionError, match="local Z bounds"):
        validator.validate_scene(env, [child, support])
