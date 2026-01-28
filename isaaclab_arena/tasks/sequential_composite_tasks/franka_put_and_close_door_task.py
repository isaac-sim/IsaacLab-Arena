# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class FrankaPutAndCloseDoorTask(SequentialTaskBase):

    def __init__(
        self,
        openable_object,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
    ):
        super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)
        self.openable_object = openable_object

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))

    def get_prompt(self) -> str:
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        cfg = super().get_mimic_env_cfg(arm_mode)
        cfg.name = "franka_put_and_close_door_task"
        return cfg
