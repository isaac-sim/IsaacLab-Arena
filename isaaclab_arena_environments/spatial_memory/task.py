# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena task that wraps a ``SpatialMemoryTaskLogic``: sensors, layout reset, success/failure terms and metrics."""

from __future__ import annotations

import torch
from functools import partial

from isaaclab.managers import EventTermCfg, TerminationTermCfg

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena_environments.spatial_memory.collision import make_contact_sensor_cfgs
from isaaclab_arena_environments.spatial_memory.metrics import get_suite_metrics
from isaaclab_arena_environments.spatial_memory.observations import ObservationVariant, make_task_observation_cfg
from isaaclab_arena_environments.spatial_memory.runtime import NOT_SET, reset_layout, update_runtime
from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic


def collision_failure(env, logic: SpatialMemoryTaskLogic) -> torch.Tensor:
    """Failure term: the robot or a carried item touched scene geometry the task does not need."""
    runtime = update_runtime(env, logic)
    return runtime.collided.clone()


def searched_again_failure(env, logic: SpatialMemoryTaskLogic) -> torch.Tensor:
    """Failure term: the memorised scene was re-observed beyond the allowed glimpse before the first grasp."""
    runtime = update_runtime(env, logic)
    return runtime.searched_again.clone()


def task_specific_failure(env, logic: SpatialMemoryTaskLogic) -> torch.Tensor:
    """Failure term: task-specific failures such as a dropped item or a displaced neighbour."""
    runtime = update_runtime(env, logic)
    return logic.extra_failures(env, runtime)


def task_success(env, logic: SpatialMemoryTaskLogic) -> torch.Tensor:
    """Success predicate: the task condition holds and the episode has not collided."""
    runtime = update_runtime(env, logic)
    success = logic.success(env, runtime) & ~runtime.collided
    if logic.forbids_searching_again:
        success &= ~runtime.searched_again
    first = success & (runtime.success_step == NOT_SET)
    runtime.success_step[first] = env.episode_length_buf[first]
    return success


class SpatialMemoryTask(TaskBase):
    """Arena task for one spatial-memory logic.

    Args:
        logic: The task logic.
        observation_variant: Observation variant that decides which task observation groups are added.
    """

    def __init__(self, logic: SpatialMemoryTaskLogic, observation_variant: ObservationVariant):
        super().__init__(episode_length_s=logic.episode_length_s, task_description=logic.task_description)
        self.logic = logic
        self.observation_variant = ObservationVariant(observation_variant)

    def get_scene_cfg(self):
        sensors = make_contact_sensor_cfgs(self.logic)
        fields = [(name, type(cfg), cfg) for name, cfg in sensors.items()]
        return make_configclass("SpatialMemorySceneCfg", fields)()

    def get_events_cfg(self):
        layout_event = EventTermCfg(func=reset_layout, mode="reset", params={"logic": self.logic})
        return make_configclass("SpatialMemoryEventsCfg", [("spatial_memory_layout", EventTermCfg, layout_event)])()

    def get_termination_cfg(self) -> TaskTerminationCfg:
        failures = {
            "collision": TerminationTermCfg(func=collision_failure, params={"logic": self.logic}),
            "task_failure": TerminationTermCfg(func=task_specific_failure, params={"logic": self.logic}),
        }
        if self.logic.forbids_searching_again:
            failures["searched_again"] = TerminationTermCfg(func=searched_again_failure, params={"logic": self.logic})
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name=self.logic.name,
                    predicate_sequence=[partial(task_success, logic=self.logic)],
                )
            ],
            failures=failures,
        )

    def get_observation_cfg(self):
        return make_task_observation_cfg(self.logic, self.observation_variant)

    def get_viewer_cfg(self):
        from isaaclab.envs.common import ViewerCfg

        # A close view of the robot and its workspace, from behind its right shoulder.
        return ViewerCfg(eye=(-0.75, -1.15, 1.1), lookat=(0.45, 0.1, 0.15), origin_type="env", env_index=0)

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("Spatial-memory tasks do not define a Mimic configuration.")

    def get_metrics(self) -> list[MetricBase]:
        return get_suite_metrics()
