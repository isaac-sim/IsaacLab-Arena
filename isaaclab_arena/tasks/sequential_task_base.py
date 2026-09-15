# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase


class SequentialTaskBase(CompositeTaskBase):
    """A composed task whose children complete in order through the progress tracker.

    The next child starts on the following control step. Completing the final
    child completes the task on that same step once its current-state requirements hold.
    """

    subtasks_are_sequential: bool = True
