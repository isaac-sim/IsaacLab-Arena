# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Connect Arena-owned task progress to Isaac Lab's termination evaluation."""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria


@configclass
class TaskSuccessCfg:
    """Configure the progress tracker owned by an Arena environment."""

    success_criteria: list[CompletionCriteria] = MISSING
    """Ordered completion criteria whose completion determines success."""

    subtasks_are_sequential: bool = False
    """Whether later subtasks wait for all criteria in the current subtask."""

    desired_subtask_success_state: list[bool | None] | None = None
    """Required completion state for each subtask, or None for normal completion."""


def task_success(env) -> torch.Tensor:
    """Advance Arena-owned task progress and return success for each environment."""
    progress_tracker = env.progress_tracker
    assert progress_tracker is not None, "Arena must initialize task progress before evaluating success."
    progress_tracker.step(env, step_index=env.episode_length_buf)
    return progress_tracker.is_complete()
