# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def test_slow_carry_counts_as_displaced():
    from isaaclab_arena.metrics.object_displaced import compute_object_displaced_rate
    from isaaclab_arena.metrics.object_peak_speed_exceeded import compute_object_peak_speed_exceeded_rate

    # 0.3 m carried over 300 steps at 1 mm/step: slow in speed terms, clearly displaced.
    num_steps = 300
    slow_carry_w = np.zeros((num_steps, 3))
    slow_carry_w[:, 0] = np.linspace(0.0, 0.3, num_steps)
    assert compute_object_displaced_rate([slow_carry_w], displacement_threshold_m=0.02) == 1.0

    # The same trajectory as velocity (0.01 m/s) stays under the 0.5 m/s speed threshold.
    slow_velocity_w = np.zeros((num_steps, 3))
    slow_velocity_w[:, 0] = 0.01
    assert compute_object_peak_speed_exceeded_rate([slow_velocity_w], object_velocity_threshold=0.5) == 0.0


def test_stationary_and_jitter_are_not_displaced():
    from isaaclab_arena.metrics.object_displaced import compute_object_displaced_rate

    stationary_w = np.tile(np.array([0.5, 0.1, 0.9]), (100, 1))
    jitter_w = stationary_w + np.random.default_rng(0).uniform(-1e-3, 1e-3, stationary_w.shape)
    assert compute_object_displaced_rate([stationary_w, jitter_w], displacement_threshold_m=0.02) == 0.0


def test_rate_is_fraction_of_episodes():
    from isaaclab_arena.metrics.object_displaced import compute_object_displaced_rate

    still_w = np.zeros((10, 3))
    moved_w = np.zeros((10, 3))
    moved_w[-1, 2] = 0.1
    assert compute_object_displaced_rate([still_w, moved_w], displacement_threshold_m=0.02) == 0.5
    assert compute_object_displaced_rate([], displacement_threshold_m=0.02) == 0.0
