# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EnvStepTimerWrapper. No Isaac Sim or GPU required."""

import gymnasium as gym

from isaaclab_arena.utils.env_step_timer import EnvStepTimerWrapper
from isaaclab_arena.utils.timer import get_timer_stats, reset_timer_stats


class _StubEnv(gym.Env):
    """Minimal env that returns a fixed step result."""

    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        super().__init__()
        self.num_steps = 0

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        self.num_steps += 1
        return {}, 0.0, False, False, {}


def test_records_one_measurement_per_step():
    """Each step through the wrapper adds one measurement under the configured name."""
    reset_timer_stats()
    env = EnvStepTimerWrapper(_StubEnv(), timer_name="test/inner_step")

    for _ in range(3):
        env.step(None)

    stats = get_timer_stats()
    assert stats["test/inner_step"].count == 3
    assert env.env.num_steps == 3


def test_step_result_is_passed_through():
    """The wrapper returns the wrapped env's step result unchanged."""
    reset_timer_stats()
    env = EnvStepTimerWrapper(_StubEnv(), timer_name="test/inner_step")

    assert env.step(None) == ({}, 0.0, False, False, {})


def test_reset_is_not_timed():
    """Only step is measured, so a reset leaves the registry untouched."""
    reset_timer_stats()
    env = EnvStepTimerWrapper(_StubEnv(), timer_name="test/inner_step")

    env.reset()

    assert "test/inner_step" not in get_timer_stats()
