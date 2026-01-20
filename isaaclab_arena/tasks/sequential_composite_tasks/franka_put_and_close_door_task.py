# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
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

    def get_metrics(self) -> list[MetricBase]:
        return None

    def get_prompt(self) -> str:
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        mimic_env_cfg = FrankaPutAndCloseDoorTaskMimicEnvCfg()
        mimic_env_cfg.subtask_configs = self.combine_mimic_subtask_configs(arm_mode)
        return mimic_env_cfg


@configclass
class FrankaPutAndCloseDoorTaskMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka put and close door task.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "franka_put_and_close_door_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

            


