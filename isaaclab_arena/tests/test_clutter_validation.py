# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for clutter rest and settle verdicts."""

from __future__ import annotations

import math
import torch

import pytest

from isaaclab_arena.relations.clutter_drop_poses import ClutterRegion
from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, SettleTracker, check_resting_poses

IDENTITY = (0.0, 0.0, 0.0, 1.0)
REGION = ClutterRegion(min_x=-0.5, min_y=-0.5, max_x=0.5, max_y=0.5, floor_z=0.75)


def _rotations(count: int, quaternion=IDENTITY) -> torch.Tensor:
    return torch.tensor([quaternion] * count, dtype=torch.float32)


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def test_objects_at_rest_on_the_support_pass():
    positions = torch.tensor([[0.0, 0.0, 0.78], [0.2, -0.3, 0.80]])
    assert check_resting_poses(positions, REGION).ok


def test_object_outside_the_footprint_is_reported_as_fallen_off():
    positions = torch.tensor([[0.0, 0.0, 0.78], [0.9, 0.0, 0.78]])
    verdict = check_resting_poses(positions, REGION)
    assert verdict.fell_off == [1]
    assert not verdict.ok


def test_object_below_the_support_is_reported_as_fallen_through():
    positions = torch.tensor([[0.0, 0.0, 0.78], [0.0, 0.0, 0.50]])
    verdict = check_resting_poses(positions, REGION)
    assert verdict.fell_through == [1]


def test_resting_slightly_below_the_surface_is_tolerated():
    positions = torch.tensor([[0.0, 0.0, 0.745]])
    assert check_resting_poses(positions, REGION).ok


def test_non_finite_pose_is_reported_as_diverged_only():
    positions = torch.tensor([[float("nan"), float("nan"), float("nan")]])
    verdict = check_resting_poses(positions, REGION)
    assert verdict.diverged == [0]
    assert verdict.fell_off == []
    assert verdict.fell_through == []


def test_containment_margin_admits_objects_just_past_the_edge():
    positions = torch.tensor([[0.52, 0.0, 0.78]])
    assert not check_resting_poses(positions, REGION).ok
    lenient = ClutterSettleParams(containment_margin_m=0.05)
    assert check_resting_poses(positions, REGION, lenient).ok


def test_verdict_describes_offenders_by_name():
    positions = torch.tensor([[0.9, 0.0, 0.78], [0.0, 0.0, 0.2]])
    verdict = check_resting_poses(positions, REGION)
    description = verdict.describe(["mug", "can"])
    assert "fell off" in description and "mug" in description
    assert "fell through" in description and "can" in description


def test_first_poll_is_never_settled():
    tracker = SettleTracker()
    assert not tracker.update(torch.zeros(3, 3), _rotations(3))


def test_settling_requires_consecutive_quiet_windows():
    tracker = SettleTracker(ClutterSettleParams(required_quiet_windows=2))
    positions = torch.zeros(3, 3)
    rotations = _rotations(3)

    assert not tracker.update(positions, rotations)  # baseline
    assert not tracker.update(positions, rotations)  # one quiet window
    assert tracker.update(positions, rotations)  # two, settled
    assert tracker.settled


def test_movement_resets_the_quiet_streak():
    tracker = SettleTracker(ClutterSettleParams(required_quiet_windows=2))
    rotations = _rotations(1)
    still = torch.zeros(1, 3)

    tracker.update(still, rotations)
    tracker.update(still, rotations)
    assert tracker.quiet_windows == 1

    tracker.update(torch.tensor([[0.0, 0.0, 0.05]]), rotations)
    assert tracker.quiet_windows == 0
    assert not tracker.settled


def test_rotation_alone_prevents_settling():
    tracker = SettleTracker(ClutterSettleParams(required_quiet_windows=1))
    still = torch.zeros(1, 3)

    tracker.update(still, _rotations(1))
    assert not tracker.update(still, _rotations(1, _yaw_quaternion(30.0)))


def test_divergence_resets_and_blocks_settling():
    tracker = SettleTracker(ClutterSettleParams(required_quiet_windows=1))
    still = torch.zeros(2, 3)
    rotations = _rotations(2)

    tracker.update(still, rotations)
    assert tracker.update(still, rotations)
    assert tracker.settled

    nan_positions = torch.full((2, 3), float("nan"))
    assert not tracker.update(nan_positions, rotations)
    assert not tracker.settled


def test_tracker_is_unaffected_by_caller_mutating_the_snapshot():
    tracker = SettleTracker(ClutterSettleParams(required_quiet_windows=1))
    positions = torch.zeros(1, 3)
    rotations = _rotations(1)

    tracker.update(positions, rotations)
    positions[0, 2] = 1.0
    positions[0, 2] = 0.0
    assert tracker.update(positions, rotations)


@pytest.mark.parametrize(
    "position,margin,fell_off,fell_through",
    [
        ((0.0, 0.0, 0.80), 0.0, [], []),
        ((0.35, 0.0, 0.80), 0.0, [0], []),
        ((0.35, 0.0, 0.80), 0.06, [], []),
        ((0.0, 0.0, 0.77), 0.0, [], [0]),
    ],
)
def test_containment_uses_rotated_body_extents(position, margin, fell_off, fell_through):
    from isaaclab_arena.relations.clutter_pour import resting_extents
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    bbox = AxisAlignedBoundingBox(min_point=(-0.1, -0.2, -0.05), max_point=(0.3, 0.4, 0.05))
    obj = DummyObject("offset_box", bbox)
    layout = PlacementResult(
        PlacementValidationResults(), {obj: position}, 0.0, 1, rotations={obj: _yaw_quaternion(90)}
    )
    extents = resting_extents(obj, layout, bbox)
    assert extents == pytest.approx((-0.4, -0.1, 0.2, 0.3, -0.05))
    verdict = check_resting_poses(
        torch.tensor([position]), REGION, ClutterSettleParams(containment_margin_m=margin), [extents]
    )
    assert verdict.fell_off == fell_off
    assert verdict.fell_through == fell_through


def test_passive_drift_is_independent_of_quiet_thresholds(capsys):
    from isaaclab_arena.relations.clutter_preparation import _poses_unchanged

    initial = torch.tensor([[0.0, 0.0, 0.0, *IDENTITY]])
    current = initial.clone()
    current[0, 0] = 0.005
    params = ClutterSettleParams(move_thresh_m=0.0001, passive_move_thresh_m=0.01)
    assert _poses_unchanged(initial, current, params, "env 2, robot links")
    params.passive_move_thresh_m = 0.002
    assert not _poses_unchanged(initial, current, params, "env 2, robot links")
    output = capsys.readouterr().out
    assert "env 2, robot links" in output and "0.005000 m" in output


def test_passive_rotation_has_its_own_tolerance():
    from isaaclab_arena.relations.clutter_preparation import _poses_unchanged

    initial = torch.tensor([[0.0, 0.0, 0.0, *IDENTITY]])
    current = torch.tensor([[0.0, 0.0, 0.0, *_yaw_quaternion(3)]])
    params = ClutterSettleParams(turn_thresh_deg=0.1, passive_turn_thresh_deg=4)
    assert _poses_unchanged(initial, current, params, "fixture")
    params.passive_turn_thresh_deg = 2
    assert not _poses_unchanged(initial, current, params, "fixture")
