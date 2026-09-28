# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

import pytest

from isaaclab_arena.utils.physics_settle import get_pose_drift


def test_pose_drift_is_zero_for_unchanged_and_sign_equivalent_poses():
    half_sqrt = math.sqrt(0.5)
    initial = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, half_sqrt, half_sqrt])
    current = initial.clone()
    assert get_pose_drift(initial, current) == pytest.approx((0.0, 0.0), abs=1e-5)
    current[3:] *= -1
    assert get_pose_drift(initial, current) == pytest.approx((0.0, 0.0), abs=1e-5)


def test_pose_drift_measures_translation_and_rotation():
    half_sqrt = math.sqrt(0.5)
    initial = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    current = torch.tensor([4.0, 6.0, 3.0, 0.0, 0.0, half_sqrt, half_sqrt])
    assert get_pose_drift(initial, current) == pytest.approx((5.0, 90.0))


def test_pose_drift_takes_independent_maxima_across_environments_and_links():
    initial = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(2, 2, 1)
    current = initial.clone()
    current[0, 1, :3] = torch.tensor([0.006, 0.008, 0.0])
    current[1, 0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert get_pose_drift(initial, current) == pytest.approx((0.01, 180.0))


@pytest.mark.parametrize(
    ("pose_name", "component", "value"),
    [("initial", 0, float("nan")), ("current", 6, float("inf"))],
    ids=["nan-initial-position", "inf-current-quaternion"],
)
def test_pose_drift_rejects_nonfinite_input(pose_name, component, value):
    initial = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(2, 1)
    current = initial.clone()
    poses = {"initial": initial, "current": current}
    poses[pose_name][1, component] = value
    assert get_pose_drift(initial, current) is None
