# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Composite task in which a Franka robot completes a first subtask and then closes a door."""

import numpy as np

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class FrankaPutAndCloseDoorTask(CompositeTaskBase):
    """Composite task whose two subtasks must both succeed, in order.

    The viewer looks at the door.
    """

    def __init__(
        self,
        openable_object,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
    ):
        """Initialize the task.

        Args:
            openable_object: The door-like object the viewer camera looks at.
            subtasks: The two subtasks, completed in order.
            episode_length_s: Episode time limit in seconds; the sum of the subtask timeouts when None.
        """
        desired_subtask_success_state = [True, True]
        super().__init__(
            subtasks=subtasks,
            episode_length_s=episode_length_s,
            desired_subtask_success_state=desired_subtask_success_state,
            subtasks_are_sequential=True,
        )
        self.openable_object = openable_object

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return a viewer camera configuration that looks at the openable object."""
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))

    def get_prompt(self) -> str:
        """Return None; the task has no prompt."""
        return None  # noqa: R501

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Return the combined Isaac Lab Mimic environment configuration of the subtasks.

        Args:
            arm_mode: Which arm or arms the Mimic subtasks are generated for.
        """
        cfg = super().get_mimic_env_cfg(arm_mode)
        cfg.name = "franka_put_and_close_door_task"
        return cfg
