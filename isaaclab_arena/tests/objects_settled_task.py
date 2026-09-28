# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test task that succeeds after objects remain settled for consecutive steps."""

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg


class ObjectsSettledTask(TaskBase):
    """Succeed when every configured object remains settled for consecutive steps."""

    def __init__(self, object_names: list[str], consecutive_steps: int = 5):
        super().__init__(episode_length_s=10.0, task_description="Wait for all objects to settle")
        self.object_names = object_names
        self.consecutive_steps = consecutive_steps

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        success = TrueForConsecutiveStepsCfg(
            predicate=TerminationTermCfg(
                func=objects_below_velocity_thresholds,
                params={"object_names": self.object_names},
            ),
            required_steps=self.consecutive_steps,
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="objects_settled",
                    predicate_sequence=[success],
                    description="All objects completed the consecutive settling window.",
                )
            ],
        )

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.8, -1.8, 1.4), lookat=(0.25, 0.0, 0.3))
