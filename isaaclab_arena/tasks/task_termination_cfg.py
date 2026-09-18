# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective


@dataclass
class TaskTerminationCfg:
    """Declare when a task succeeds, fails, or runs out of time."""

    timeout_s: float | None
    """Episode time limit in seconds; None disables timeout termination."""

    success: list[ProgressObjective] = field(default_factory=list)
    """Objectives required for success by default; an empty list disables success termination."""

    failures: dict[str, TerminationTermCfg] = field(default_factory=dict)
    """Named failure conditions; any true condition ends the episode."""

    subtasks_are_sequential: bool = False
    """Whether ProgressTracker waits for each subtask's objectives before advancing the next subtask."""

    desired_subtask_success_state: list[bool | None] | None = None
    """Optional final subtask conditions; None entries exclude that subtask from the success check."""

    def __post_init__(self):
        if self.timeout_s is not None:
            assert (
                math.isfinite(self.timeout_s) and self.timeout_s > 0
            ), "timeout_s must be finite and positive, or None to disable timeout."
        assert isinstance(self.success, list) and all(
            isinstance(objective, ProgressObjective) for objective in self.success
        ), "success must be a list of ProgressObjective definitions."
        assert isinstance(self.failures, dict), "failures must map names to TerminationTermCfg definitions."
        for name, failure in self.failures.items():
            assert name not in {"success", "time_out"}, f"Failure name '{name}' is reserved."
            assert isinstance(failure, TerminationTermCfg), f"Failure '{name}' must be a TerminationTermCfg."
            assert not failure.time_out, f"Failure '{name}' cannot be a timeout; use timeout_s instead."
