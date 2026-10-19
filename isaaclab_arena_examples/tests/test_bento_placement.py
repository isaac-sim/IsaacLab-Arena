# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that bento scoring rejects incomplete or unstable placements."""

import math

import pytest

from isaaclab_arena_examples.bento.meal import FOODS, TRAY_FLOOR, Food, placement_ok


@pytest.mark.parametrize("food", FOODS, ids=lambda food: food.name)
def test_destination_fits_portion(food):
    pose = [*food.target_xy, TRAY_FLOOR + food.size[2] / 2, 0.0, 0.0, 0.0, 1.0]
    assert placement_ok(food, pose, [0.0] * 6, released=True)
    assert not placement_ok(food, pose, [0.0] * 6, released=False)
    assert not placement_ok(food, pose, [0.1, 0, 0, 0, 0, 0], released=True)
    assert not placement_ok(food, pose, [0, 0, 0, 0, 0, 1], released=True)
    pose[2] += 0.025
    assert not placement_ok(food, pose, [0.0] * 6, released=True)


def test_portion_center_inside_is_not_enough():
    food = FOODS[0]
    pose = [food.bounds_xy[0] + 0.01, food.target_xy[1], TRAY_FLOOR + food.size[2] / 2, 0, 0, 0, 1]
    assert not placement_ok(food, pose, [0.0] * 6, released=True)


def test_rotated_portion_over_divider_is_rejected():
    food = FOODS[1]
    # The narrow egg fits here at zero yaw; its long edge crosses the divider at pi/2.
    pose = [food.target_xy[0], food.bounds_xy[2] + 0.03, TRAY_FLOOR + food.size[2] / 2, 0, 0, 0, 1]
    assert placement_ok(food, pose, [0.0] * 6, released=True)
    pose[5:] = [math.sqrt(0.5), math.sqrt(0.5)]
    assert not placement_ok(food, pose, [0.0] * 6, released=True)


def test_sphere_rotation_does_not_change_containment():
    food = Food("sphere", (0.05, 0.05, 0.05), (0, 0, 0), (0, 0), (0.67, 0.16), FOODS[2].bounds_xy, 0.025, True)
    pose = [*food.target_xy, TRAY_FLOOR + food.size[2] / 2, 0.5, 0.5, 0.5, 0.5]
    assert placement_ok(food, pose, [0.0] * 6, released=True)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_pose_cannot_succeed(bad_value):
    food = FOODS[0]
    pose = [*food.target_xy, TRAY_FLOOR + food.size[2] / 2, 0, 0, 0, 1]
    pose[0] = bad_value
    assert not placement_ok(food, pose, [0.0] * 6, released=True)
