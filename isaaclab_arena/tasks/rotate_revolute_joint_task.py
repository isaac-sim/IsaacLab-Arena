# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.revolute_joint_moved_rate import RevoluteJointMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class RotateRevoluteJointTask(TaskBase):
    def __init__(
        self,
        openable_object: Openable,
        target_joint_percentage_threshold: float,
        reset_joint_percentage: float,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.openable_object = openable_object
        self.target_joint_percentage_threshold = target_joint_percentage_threshold
        self.reset_joint_percentage = reset_joint_percentage
        self.task_description = (
            f"Rotate the {self.openable_object.name} joint to the target {target_joint_percentage_threshold} joint"
            " percentage."
            if task_description is None
            else task_description
        )
        self.events_cfg = RotateRevoluteJointEventCfg(
            self.openable_object, reset_openable_object_revolute_joint_percentage=self.reset_joint_percentage
        )
        self.scene_config = None
        self.termination_cfg = None
        self.mimic_env_cfg = None

    def get_scene_cfg(self):
        return self.scene_config

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function {self.get_mimic_env_cfg.__name__} not implemented yet.")

    def get_termination_cfg(self):
        raise NotImplementedError("Function {self.get_termination_cfg.__name__} not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            RevoluteJointMovedRateMetric(
                self.openable_object,
                reset_joint_percentage=self.reset_joint_percentage,
            ),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))


@configclass
class RotateRevoluteJointEventCfg:
    """Configuration for Open Door."""

    reset_openable_object_revolute_joint_percentage: EventTermCfg = MISSING

    reset_openable_object_pose: EventTermCfg = MISSING

    def __init__(self, openable_object: Openable, reset_openable_object_revolute_joint_percentage: float | None):
        assert isinstance(openable_object, Openable), "Object pose must be an instance of Openable"
        params = {}
        if reset_openable_object_revolute_joint_percentage is not None:
            params["percentage"] = reset_openable_object_revolute_joint_percentage
        self.reset_openable_object_revolute_joint_percentage = EventTermCfg(
            func=openable_object.rotate_revolute_joint,
            mode="reset",
            params=params,
        )
        initial_pose = openable_object.get_initial_pose()
        if initial_pose is not None:
            self.reset_openable_object_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(openable_object.name),
                },
            )
