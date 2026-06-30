# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for the front-hemisphere camera sampler (no Isaac Sim)."""

import numpy as np

import pytest

from isaaclab_arena_datagen.utils.camera_utils import sample_front_hemisphere_cameras


def test_count_radius_and_target():
    center = (0.2, 0.0, 0.3)
    radius = 1.5
    cams = sample_front_hemisphere_cameras(10, radius=radius, center=center, focal_length_mm=14.0, seed=0)
    assert len(cams) == 10
    for cam in cams:
        assert cam.target == center  # every camera looks at the center
        assert cam.focal_length_mm == 14.0
        dist = np.linalg.norm(np.array(cam.position) - np.array(center))
        assert abs(dist - radius) < 1e-6  # equal radius from the robot


def test_front_hemisphere_and_above_floor():
    center = (0.0, 0.0, 0.3)
    front = (1.0, 0.0, 0.0)
    cams = sample_front_hemisphere_cameras(20, radius=1.2, center=center, front_dir=front, min_height=0.1, seed=1)
    c = np.array(center)
    f = np.array(front)
    for cam in cams:
        p = np.array(cam.position)
        assert float(np.dot(p - c, f)) >= -1e-6  # on the front side
        assert p[2] >= 0.1 - 1e-6  # above the floor


def test_seed_reproducible_and_default_random():
    a = sample_front_hemisphere_cameras(5, radius=1.0, seed=42)
    b = sample_front_hemisphere_cameras(5, radius=1.0, seed=42)
    assert [c.position for c in a] == [c.position for c in b]  # same seed -> same layout

    # Default (no seed) should re-randomise (extremely unlikely to collide).
    c = sample_front_hemisphere_cameras(5, radius=1.0)
    d = sample_front_hemisphere_cameras(5, radius=1.0)
    assert [x.position for x in c] != [x.position for x in d]


def test_invalid_args():
    with pytest.raises(ValueError):
        sample_front_hemisphere_cameras(0, radius=1.0)
    with pytest.raises(ValueError):
        sample_front_hemisphere_cameras(3, radius=0.0)
    # min_height above the whole hemisphere -> cannot sample
    with pytest.raises(ValueError):
        sample_front_hemisphere_cameras(3, radius=1.0, center=(0.0, 0.0, 0.0), min_height=5.0)
