# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Null task for environments that have no task."""

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg


@register_task
class NoTask(TaskBase):
    """Null object for environments without a task."""

    name = "no_task"

    def get_scene_cfg(self):
        """Return None; the task adds nothing to the scene."""
        pass

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Return a termination configuration without a timeout or success criteria."""
        return TaskTerminationCfg(timeout_s=None)

    def get_events_cfg(self):
        """Return None; the task defines no events."""
        pass

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Return None; the task has no Mimic environment configuration.

        Args:
            arm_mode: Unused.
        """
        pass

    def get_metrics(self):
        """Return None; the task defines no metrics."""
        pass

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return a fixed viewer camera pose."""
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))
