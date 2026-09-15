# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test task that succeeds after objects remain settled for consecutive steps."""

from dataclasses import MISSING
from functools import partial

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.object_settling import ObjectsSettledForConsecutiveSteps
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import termination_term_result


class ObjectsSettledTask(TaskBase):
    """Succeed when every configured object remains settled for consecutive steps."""

    def __init__(self, object_names: list[str], consecutive_steps: int = 5):
        super().__init__(episode_length_s=10.0, task_description="Wait for all objects to settle")
        self.object_names = object_names
        self.consecutive_steps = consecutive_steps

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=ObjectsSettledForConsecutiveSteps,
            params={
                "object_names": self.object_names,
                "consecutive_steps": self.consecutive_steps,
            },
        )
        return ObjectsSettledTerminationsCfg(success=success)

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return []

    def get_progress_objectives(self) -> list[ProgressObjective]:
        return [
            ProgressObjective(
                name="objects_settled",
                predicate_groups=partial(termination_term_result, term_name="success"),
                description="All objects completed the consecutive settling window.",
            )
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.8, -1.8, 1.4), lookat=(0.25, 0.0, 0.3))


@configclass
class ObjectsSettledTerminationsCfg:
    """Termination terms for the objects-settled test task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
