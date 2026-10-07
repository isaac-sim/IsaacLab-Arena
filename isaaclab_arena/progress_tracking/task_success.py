# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Expose the environment's task success to termination managers and direct callers."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def task_success(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
    """Update task progress and return success for each environment.

    Args:
        env: An Arena environment with task progress configured.

    Returns:
        Boolean success for each environment after the latest control step.
    """
    env.update_task_progress()
    tracker = env.progress_tracker
    assert tracker is not None, "Task progress is not configured."
    return tracker.is_complete()
