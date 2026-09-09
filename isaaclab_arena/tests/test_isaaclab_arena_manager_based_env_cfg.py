# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for IsaacLabArenaManagerBasedRLEnvCfg defaults.

Importing isaaclab_arena_manager_based_env_cfg pulls in isaaclab.envs -> isaaclab.managers,
which needs a running SimulationApp (omni.timeline is only importable once Kit has
booted) -- see isaaclab_arena/tests/test_task_registry.py for the established
_test_/test_ + run_function_with_persistent_simulation_app pattern this mirrors.
"""

from __future__ import annotations

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_default_reruns_after_reset_to_flush_stale_camera_frames(simulation_app):
    """A positive default avoids RTX sensors reading the previous episode's last frame."""
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )

    assert IsaacLabArenaManagerBasedRLEnvCfg().num_rerenders_on_reset == 5
    return True


def test_default_reruns_after_reset_to_flush_stale_camera_frames():
    result = run_function_with_persistent_simulation_app(
        _test_default_reruns_after_reset_to_flush_stale_camera_frames
    )
    assert result
