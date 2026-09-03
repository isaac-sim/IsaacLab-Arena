# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper that times the env it wraps, so the cost of the wrappers above it can be separated."""

from __future__ import annotations

import gymnasium as gym

from isaaclab_arena.utils.timer import Timer


class EnvStepTimerWrapper(gym.Wrapper):
    """Record the wall time of the wrapped env's step under a timer name.

    The timer covers everything below this wrapper and nothing above it, so placing it under a
    stack of wrappers attributes the difference against an outer measurement to those wrappers.
    """

    def __init__(self, env: gym.Env, timer_name: str) -> None:
        """Wrap an env and record each of its steps under the given timer name.

        Args:
            env: The env whose step is timed.
            timer_name: Registry key the measurements accumulate under.
        """
        super().__init__(env)
        self.timer_name = timer_name

    def step(self, action):
        """Step the wrapped env, recording how long it took."""
        with Timer(self.timer_name):
            return self.env.step(action)
