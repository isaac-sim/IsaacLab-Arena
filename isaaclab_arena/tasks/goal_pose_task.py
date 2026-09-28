# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from functools import partial

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import goal_pose_task_termination
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class GoalPoseTask(TaskBase):
    def __init__(
        self,
        object: Asset,
        episode_length_s: float | None = None,
        target_x_range: tuple[float, float] | None = None,
        target_y_range: tuple[float, float] | None = None,
        target_z_range: tuple[float, float] | None = None,
        target_orientation_xyzw: tuple[float, float, float, float] | None = None,
        target_orientation_tolerance_rad: float | None = None,
    ):
        """
        Args:
            object: The object asset for the goal pose task.
            episode_length_s: Episode length in seconds.
            target_x_range: Success zone x-range [min, max] in meters.
            target_y_range: Success zone y-range [min, max] in meters.
            target_z_range: Success zone z-range [min, max] in meters.
            target_orientation_xyzw: Target quaternion [x, y, z, w].
            target_orientation_tolerance_rad: Angular tolerance in radians (default: 0.1).
        """
        super().__init__(episode_length_s=episode_length_s)
        self.object = object
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range
        self.target_z_range = target_z_range
        self.target_orientation_xyzw = target_orientation_xyzw
        self.target_orientation_tolerance_rad = target_orientation_tolerance_rad
        # this is needed to revise the default env_spacing in arena_env_builder: priority task > embodiment > scene > default
        self.scene_config = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
        self.events_cfg = None

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self) -> TaskTerminationCfg:
        params: dict = {"object_cfg": SceneEntityCfg(self.object.name)}
        if self.target_x_range is not None:
            params["target_x_range"] = self.target_x_range
        if self.target_y_range is not None:
            params["target_y_range"] = self.target_y_range
        if self.target_z_range is not None:
            params["target_z_range"] = self.target_z_range
        if self.target_orientation_xyzw is not None:
            params["target_orientation_xyzw"] = self.target_orientation_xyzw
        if self.target_orientation_tolerance_rad is not None:
            params["target_orientation_tolerance_rad"] = self.target_orientation_tolerance_rad
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="reach_goal_pose",
                    predicate_sequence=[partial(goal_pose_task_termination, **params)],
                )
            ],
        )

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            ObjectMovedRateMetric(self.object),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.object, offset=np.array([1.5, 1.5, 1.5]))
