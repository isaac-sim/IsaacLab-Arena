# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for TaskBase episode-length defaults."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_episode_length_defaults(simulation_app) -> bool:
    from isaaclab_arena.tasks.task_base import TaskBase

    class _StubTask(TaskBase):
        """Mirrors the real subclasses: declares ``episode_length_s: float | None = None`` and forwards it."""

        def __init__(self, episode_length_s: float | None = None):
            super().__init__(episode_length_s=episode_length_s)

        def get_scene_cfg(self):
            return None

        def get_termination_cfg(self):
            return None

        def get_events_cfg(self):
            return None

        def get_mimic_env_cfg(self, arm_mode):
            return None

        def get_metrics(self):
            return []

    # Regression: a bare ``episode_length_s: float = 20.0`` param default is bypassed by subclasses
    # forwarding None, which would leave episode_length_s = None and break episode-length setup.
    assert _StubTask().get_episode_length_s() == 20.0
    assert _StubTask(episode_length_s=None).get_episode_length_s() == 20.0
    assert _StubTask(episode_length_s=5.0).get_episode_length_s() == 5.0
    return True


def test_episode_length_defaults():
    result = run_simulation_app_function(_test_episode_length_defaults)
    assert result
