# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data-only unit tests for TaskBase defaults (no SimulationApp required)."""

from __future__ import annotations

from isaaclab_arena.tasks.task_base import TaskBase


class _StubTask(TaskBase):
    """Minimal concrete task that mirrors the real subclasses' constructor pattern.

    Like PickAndPlaceTask et al., it declares ``episode_length_s: float | None = None`` and forwards
    it verbatim to ``super().__init__`` — the case that must still coalesce to the 20s default.
    """

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


def test_episode_length_defaults_to_20s_when_subclass_forwards_none():
    # Regression guard: a bare `episode_length_s: float = 20.0` param default would be bypassed by
    # subclasses forwarding None, leaving episode_length_s = None and breaking episode-length setup.
    assert _StubTask().get_episode_length_s() == 20.0
    assert _StubTask(episode_length_s=None).get_episode_length_s() == 20.0


def test_explicit_episode_length_is_preserved():
    assert _StubTask(episode_length_s=5.0).get_episode_length_s() == 5.0
