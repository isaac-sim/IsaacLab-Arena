# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import goal_pose_task_termination
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class GoalPoseTask(TaskBase):
    def __init__(
        self,
        object: Asset,
        episode_length_s: float | None = None,
        target_x_range: tuple[float, float] | None = None,
        target_y_range: tuple[float, float] | None = None,
        target_z_range: tuple[float, float] | None = None,
        target_orientation_wxyz: tuple[float, float, float, float] | None = None,
        target_orientation_tolerance_rad: float | None = None,
    ):
        """
        Args:
            object: The object asset for the goal pose task.
            episode_length_s: Episode length in seconds.
            target_x_range: Success zone x-range [min, max] in meters.
            target_y_range: Success zone y-range [min, max] in meters.
            target_z_range: Success zone z-range [min, max] in meters.
            target_orientation_wxyz: Target quaternion [w, x, y, z].
            target_orientation_tolerance_rad: Angular tolerance in radians (default: 0.1).
        """
        super().__init__(episode_length_s=episode_length_s)
        self.object = object
        # this is needed to revise the default env_spacing in arena_env_builder: priority task > embodiment > scene > default
        self.scene_config = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
        self.events_cfg = None
        self.termination_cfg = self.make_termination_cfg(
            target_x_range=target_x_range,
            target_y_range=target_y_range,
            target_z_range=target_z_range,
            target_orientation_wxyz=target_orientation_wxyz,
            target_orientation_tolerance_rad=target_orientation_tolerance_rad,
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(
        self,
        target_x_range: tuple[float, float] | None = None,
        target_y_range: tuple[float, float] | None = None,
        target_z_range: tuple[float, float] | None = None,
        target_orientation_wxyz: tuple[float, float, float, float] | None = None,
        target_orientation_tolerance_rad: float | None = None,
    ):
        params: dict = {"object_cfg": SceneEntityCfg(self.object.name)}
        if target_x_range is not None:
            params["target_x_range"] = target_x_range
        if target_y_range is not None:
            params["target_y_range"] = target_y_range
        if target_z_range is not None:
            params["target_z_range"] = target_z_range
        if target_orientation_wxyz is not None:
            params["target_orientation_wxyz"] = target_orientation_wxyz
        if target_orientation_tolerance_rad is not None:
            params["target_orientation_tolerance_rad"] = target_orientation_tolerance_rad

        success = TerminationTermCfg(
            func=goal_pose_task_termination,
            params=params,
        )
        return TerminationsCfg(success=success)

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


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
