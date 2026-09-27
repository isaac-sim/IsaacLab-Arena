# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING
from functools import partial

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class TurnKnobTask(TaskBase):
    def __init__(
        self,
        turnable_object: Turnable,
        target_level: int,
        reset_level: int = -1,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)

        assert isinstance(turnable_object, Turnable), "Object must be an instance of Turnable"
        self.turnable_object = turnable_object
        self.target_level = target_level
        self.reset_level = reset_level
        self.scene_cfg = None
        self.events_cfg = TurnKnobEventCfg(self.turnable_object, reset_level=self.reset_level)
        self.task_description = (
            f"Turn the {turnable_object.name} to level {target_level}."
            if task_description is None
            else task_description
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        params = {}
        if self.target_level is not None:
            params["target_level"] = self.target_level
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="turn_knob",
                    predicate_sequence=[partial(self.turnable_object.is_at_level, **params)],
                )
            ],
        )

    def get_scene_cfg(self):
        return self.scene_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.turnable_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )

    def get_metrics(self) -> list[MetricBase]:
        # TODO(xinjieyao, 2026.01.05): Add turning level tracking metrics for the task.
        return [
            SuccessRateMetric(),
        ]

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")


@configclass
class TurnKnobEventCfg:
    """Configuration for Turn Knob."""

    reset_knob_state: EventTermCfg = MISSING

    def __init__(self, turnable_object: Turnable, reset_level: int = -1):
        assert isinstance(turnable_object, Turnable), "Object pose must be an instance of Turnable"
        params = {}
        if reset_level is not None:
            params["target_level"] = reset_level
        self.reset_knob_state = EventTermCfg(
            func=turnable_object.turn_to_level,
            mode="reset",
            params=params,
        )
