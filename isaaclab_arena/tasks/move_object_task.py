# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import object_displaced
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class MoveObjectTask(TaskBase):
    """Task where the robot must push/move a movable articulated object (e.g. a cart)."""

    def __init__(
        self,
        movable_object: Asset,
        background_scene: Asset,
        displacement_threshold: float = 0.5,
        episode_length_s: float = 10.0,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.movable_object = movable_object
        self.background_scene = background_scene
        self.displacement_threshold = displacement_threshold

        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = self._make_termination_cfg()
        self.task_description = (
            f"Push the {movable_object.name} at least {displacement_threshold:.1f}m from its start"
            if task_description is None
            else task_description
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Mimic is not yet supported for MoveObjectTask.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.movable_object,
            offset=np.array([-2.0, -2.0, 2.0]),
        )

    def _make_termination_cfg(self):
        success = TerminationTermCfg(
            func=object_displaced,
            params={
                "object_cfg": SceneEntityCfg(self.movable_object.name),
                "displacement_threshold": self.displacement_threshold,
            },
        )
        return MoveObjectTerminationsCfg(success=success)


@configclass
class MoveObjectTerminationsCfg:
    """Termination terms for the move-object task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
