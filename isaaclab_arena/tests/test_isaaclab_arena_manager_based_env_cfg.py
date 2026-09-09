# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for IsaacLabArenaManagerBasedRLEnvCfg defaults (no Isaac Sim required)."""

from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
)


def test_default_reruns_after_reset_to_flush_stale_camera_frames():
    """A positive default avoids RTX sensors reading the previous episode's last frame."""
    assert IsaacLabArenaManagerBasedRLEnvCfg().num_rerenders_on_reset == 5
