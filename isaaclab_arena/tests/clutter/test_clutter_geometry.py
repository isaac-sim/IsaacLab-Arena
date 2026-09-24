# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Release constraints and rest acceptance independent of simulation."""

import math
import torch

import pytest


def test_resting_containment_uses_rotated_bounds():
    from isaaclab_arena.offline_placement.clutter_geometry import get_placement_region
    from isaaclab_arena.offline_placement.clutter_validation import check_resting_poses
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


def test_motion_restarts_the_required_quiet_window():
    from isaaclab_arena.offline_placement.clutter_validation import SettleTracker
    from isaaclab_arena.offline_placement.clutter_validators import RestValidator

    tracker = SettleTracker(RestValidator(required_quiet_windows=2))
    rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    positions = torch.zeros((1, 3))
    assert not tracker.update(positions, rotations)
    assert not tracker.update(positions, rotations)
    positions[0, 2] = 0.1
    assert not tracker.update(positions, rotations)
    assert not tracker.update(positions, rotations)
    assert tracker.update(positions, rotations)
