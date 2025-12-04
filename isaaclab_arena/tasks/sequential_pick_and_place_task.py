# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.terminations import object_on_destination
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class SequentialPickAndPlaceTask(SequentialTaskBase):

    def __init__(
        self,
        pick_up_object: Asset,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
    ):
        super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

        self.pick_up_object = pick_up_object

    #     self.termination_cfg = self.make_termination_cfg()


    # def make_termination_cfg(self):
    #     success = TerminationTermCfg(
    #         func=self.sequential_task_success_func,
    #         params={
    #             "task_instance": self,
    #         },
    #     )
    #     return TerminationsCfg(
    #         success=success,
    #     )

    # def get_termination_cfg(self):
    #     return self.termination_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )

    def get_metrics(self) -> list[MetricBase]:
        return None

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_prompt(self) -> str:
        return None

# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

#     success: TerminationTermCfg = MISSING

