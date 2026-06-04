# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for CameraViewTrajectory (no Isaac Sim)."""

import pytest

from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory


def test_static_camera_validates():
    cam = CameraViewTrajectory(position=(1.36, 0.0, 1.0), target=(0.0, 0.0, 0.0), focal_length_mm=14.0)
    cam.validate_trajectory_length(10)  # static (tuple) coords -> no error


def test_dynamic_camera_validates_with_matching_length():
    coords = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)]
    cam = CameraViewTrajectory(position=coords, target=(0.0, 0.0, 0.0), focal_length_mm=24.0)
    cam.validate_trajectory_length(3)


def test_dynamic_trajectory_length_mismatch_raises():
    cam = CameraViewTrajectory(
        position=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0)],
        target=(0.0, 0.0, 0.0),
        focal_length_mm=24.0,
    )
    with pytest.raises(ValueError):
        cam.validate_trajectory_length(5)  # position has 2 entries, not 5


def test_positive_focal_length_required():
    with pytest.raises(ValueError):
        CameraViewTrajectory(position=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0), focal_length_mm=0.0)
