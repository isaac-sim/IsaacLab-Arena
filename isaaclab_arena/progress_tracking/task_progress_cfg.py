# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg


@dataclass
class TaskProgressCfg:
    """Task progress settings retained independently of automatic termination."""

    success_criteria: list[CompletionCriteria] = field(default_factory=list)
    """Completion criteria used to determine task success."""

    subtasks_are_sequential: bool = False
    """Whether each subtask must complete before the next one starts."""

    desired_subtask_success_state: list[bool | None] | None = None
    """Required final subtask states, or None to use recorded completion alone."""

    tracked_events: list[TrackedEventCfg] = field(default_factory=list)
    """Independent recurring observations, excluded from success and progress scores."""
