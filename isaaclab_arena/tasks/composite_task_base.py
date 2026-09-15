# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import numpy as np
import warnings
from functools import partial
from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import (
    check_configclass_field_duplicates,
    combine_configclass_instances,
    transform_configclass_instance,
)


class SubtaskSuccessStateRecorder(RecorderTerm):
    """Records the subtask success state just before the environment is reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.objective_name = cfg.objective_name

    def record_post_step(self):
        return self.name, self._env._progress_tracker.get_child_completion(self.objective_name)


@configclass
class SubtaskSuccessStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = SubtaskSuccessStateRecorder
    name: str = "subtask_success_rate"
    objective_name: str = "task"


def compute_subtask_success_rate(recorded_metric_data: list[np.ndarray]) -> list:
    """Computes per-subtask success rates.

    Args:
        recorded_metric_data: List of arrays, each shape (num_subtasks,) with bool values.

    Returns:
        List of success rates for each subtask.
    """
    num_demos = len(recorded_metric_data)
    if num_demos == 0:
        return [0.0]

    num_subtasks = recorded_metric_data[0].shape[1]
    subtask_successes = np.zeros(num_subtasks, dtype=float)

    for ep in range(num_demos):
        ep_subtask_success_result = np.any(recorded_metric_data[ep], axis=0).astype(float)
        subtask_successes += ep_subtask_success_result
    subtask_success_rates = subtask_successes / num_demos

    return subtask_success_rates.tolist()


class SubtaskSuccessRateMetric(MetricBase):
    """Computes the per-subtask success rates.

    Returns a dict with success rate for each subtask.
    """

    name = "subtask_success_rate"
    recorder_term_name = "subtask_success_rate"

    def __init__(self, objective_name: str = "task"):
        super().__init__()
        self.objective_name = objective_name

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the subtask success state metric."""
        return SubtaskSuccessStateRecorderCfg(name=self.recorder_term_name, objective_name=self.objective_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        """Return the metric term configuration for the subtask success rate metric."""
        return MetricTermCfg(
            compute_metric_func=compute_subtask_success_rate,
            params={},
            recorder_term_name=self.recorder_term_name,
        )


class CompositeTaskBase(TaskBase):
    """
    A base class for composite tasks composed of multiple subtasks.
    Completion ordering of subtasks does not matter.


    Args:
        subtasks: List of TaskBase instances representing the subtasks that compose this composite task.
        episode_length_s: Maximum duration of a single episode in seconds. Defaults to the sum of the
            subtasks' configured timeouts, giving one overall budget for completing the task.
        task_description: (Optional) Natural-language summary of the overall composite task.
        desired_subtask_success_state: (Optional) Precise success state for each subtask during the final time step.
            Checks each subtask's current outcome after every subtask has completed its progress.
            A leaf objective uses its final predicates. Composed objectives, including tasks with
            multiple root objectives, retain completion history and explicit current-state requirements.
            None entries omit only the current-outcome check.
    """

    subtasks_are_sequential: bool = False
    """Whether each subtask waits for the preceding subtask to complete."""

    def __init__(
        self,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
        task_description: str | None = None,
        desired_subtask_success_state: list[bool | None] | None = None,
    ):
        assert len(subtasks) > 0, "Composite task requires at least one subtask"
        # Default task length is the summation of the lengths of the subtasks.
        if episode_length_s is None:
            episode_length_s = self._sum_subtask_episode_lengths_s(subtasks)
        super().__init__(episode_length_s, task_description)
        self.subtasks = subtasks

        if desired_subtask_success_state is not None:
            assert len(desired_subtask_success_state) == len(
                subtasks
            ), "Desired subtask success state must be the same length as the number of subtasks"
            assert all(
                s is None or isinstance(s, bool) for s in desired_subtask_success_state
            ), "Desired subtask success state entries must each be True, False, or None"
        self.desired_subtask_success_state = desired_subtask_success_state

    @staticmethod
    def _sum_subtask_episode_lengths_s(subtasks: list[TaskBase]) -> float:
        """Return the sum of the subtasks' configured timeouts, in seconds."""
        return sum(subtask.get_termination_cfg().timeout_s for subtask in subtasks)

    def get_viewer_cfg(self) -> ViewerCfg:
        """Use the first subtask's viewport framing (e.g. pick-and-place look-at-object)."""
        return self.subtasks[0].get_viewer_cfg()

    def apply_reachability_constraints(self) -> None:
        """Apply RequiresReachability relations to each subtask's reachability targets."""
        for subtask in self.subtasks:
            subtask.apply_reachability_constraints()

    @staticmethod
    def _add_suffix_configclass_transform(fields: list[tuple], suffix: str) -> list[tuple]:
        "Config transformation to add a suffix to all field names."
        return [(f"{name}{suffix}", ftype, value) for name, ftype, value in fields]

    def get_scene_cfg(self) -> Any:
        "Make combined scene cfg from all subtasks."
        # Check for duplicate fields across subtask scene configs and warn if found
        duplicates = check_configclass_field_duplicates(*(subtask.get_scene_cfg() for subtask in self.subtasks))
        if duplicates:
            warnings.warn(
                f"\n[WARNING] Duplicate scene config fields found across subtasks: {duplicates}. "
                "Duplicates will be ignored.\n",
                UserWarning,
            )

        scene_cfg = combine_configclass_instances("SceneCfg", *(subtask.get_scene_cfg() for subtask in self.subtasks))
        return scene_cfg

    def get_events_cfg(self) -> Any:
        "Make combined events cfg from all subtasks."
        # Collect events_cfgs from subtasks with renamed fields to avoid collisions
        renamed_events_cfgs = []
        for i, subtask in enumerate(self.subtasks):
            subtask_events_cfg = subtask.get_events_cfg()
            if subtask_events_cfg is None:
                continue
            renamed_cfg = transform_configclass_instance(
                subtask_events_cfg, partial(self._add_suffix_configclass_transform, suffix=f"_subtask_{i}")
            )
            assert renamed_cfg is not None, f"Renaming dropped subtask {i}'s events cfg"
            renamed_events_cfgs.append(renamed_cfg)

        events_cfg = combine_configclass_instances("EventsCfg", *renamed_events_cfgs)

        return events_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Compose subtask success, independent failures, and one overall timeout.

        Every child must complete its progress. A None desired state omits only
        that child's current-state constraint. Multiple success objectives from
        one subtask form a composed child whose completion history is retained.
        """
        children = []
        failures = {}
        for subtask_index, subtask in enumerate(self.subtasks):
            subtask_termination = subtask.get_termination_cfg()
            assert isinstance(subtask_termination, TaskTerminationCfg), "Subtasks must return TaskTerminationCfg."
            assert subtask_termination.success, f"Subtask {subtask_index} must define success objectives."
            namespace = f"subtask_{subtask_index}"
            namespaced_objectives = [
                self._namespace_progress_objective(objective, namespace) for objective in subtask_termination.success
            ]
            if len(namespaced_objectives) == 1:
                children.append(namespaced_objectives[0])
            else:
                children.append(ProgressObjective(name=namespace, children=namespaced_objectives))
            for failure_name, failure_term in subtask_termination.failures.items():
                failures[f"{failure_name}_subtask_{subtask_index}"] = failure_term

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="task",
                    children=children,
                    sequential=self.subtasks_are_sequential,
                    desired_child_states=self.desired_subtask_success_state,
                )
            ],
            failures=failures,
        )

    def _combine_subtask_metrics(self, subtask_idxs: list[int]) -> list[MetricBase]:
        """Combine metrics from subtasks with the given ids.

        Per-subtask "success_rate" metrics are intentionally collapsed into a single shared entry as
        the composite task should only have one success rate metric.
        Individual per-subtask success is reported separately via SubtaskSuccessRateMetric (added in get_metrics).
        """
        combined_metrics = []

        for subtask_idx in subtask_idxs:
            subtask_metrics = self.subtasks[subtask_idx].get_metrics()
            for metric in subtask_metrics:
                metric = copy.copy(metric)
                if isinstance(metric, SubtaskSuccessRateMetric):
                    metric.objective_name = f"subtask_{subtask_idx}/{metric.objective_name}"
                if metric.name != "success_rate":
                    metric.name = f"{metric.name}_subtask_{subtask_idx}"
                    metric.recorder_term_name = f"{metric.recorder_term_name}_subtask_{subtask_idx}"
                    combined_metrics.append(metric)
                else:
                    if not any(m.name == "success_rate" for m in combined_metrics):
                        combined_metrics.append(metric)

        return combined_metrics

    def get_metrics(self) -> list[MetricBase]:
        "Get metrics for the composite task."
        subtask_metrics = self._combine_subtask_metrics([i for i in range(len(self.subtasks))])
        # Add the composite task's own metric for per-subtask success rates
        subtask_metrics.append(SubtaskSuccessRateMetric())

        return subtask_metrics

    @staticmethod
    def _namespace_progress_objective(objective: ProgressObjective, namespace: str) -> ProgressObjective:
        """Prefix every objective in a child task without changing its predicate definitions."""
        children = None
        if objective.children is not None:
            children = [
                CompositeTaskBase._namespace_progress_objective(child, namespace) for child in objective.children
            ]
        return dataclasses.replace(objective, name=f"{namespace}/{objective.name}", children=children)

    def _validate_consistent_mimic_eef_names(self, arm_mode: ArmMode) -> set[str]:
        "Check that all subtasks have the same Mimic eef_names."
        mimic_eef_names = set(self.subtasks[0].get_mimic_env_cfg(arm_mode).subtask_configs.keys())
        for i, subtask in enumerate(self.subtasks[1:], start=1):
            subtask_eef_names_set = set(subtask.get_mimic_env_cfg(arm_mode).subtask_configs.keys())
            if subtask_eef_names_set != mimic_eef_names:
                raise ValueError(
                    f"All subtasks must have the same Mimic eef_names.\nSubtask 0 has eef_names: {mimic_eef_names}, but"
                    f" subtask {i} has eef_names: {subtask_eef_names_set}."
                )
        return mimic_eef_names

    def combine_mimic_subtask_configs(self, arm_mode: ArmMode) -> dict[str, list[SubTaskConfig]]:
        "Combine the Mimic subtask configs for all subtasks."
        mimic_eef_names = self._validate_consistent_mimic_eef_names(arm_mode)

        combined_mimic_subtask_configs = {eef_name: [] for eef_name in mimic_eef_names}

        # Combine the "Mimic subtask" cfgs from all subtasks
        for i, subtask in enumerate(self.subtasks):
            # Get the Mimic env cfg for the subtask
            mimic_env_cfg = subtask.get_mimic_env_cfg(arm_mode)
            for eef_name in mimic_eef_names:
                # For each eef, get the "Mimic subtask" cfgs for the subtask, update the term signal name,
                # and add it to the combined "Mimic subtask" list
                for mimic_subtask in mimic_env_cfg.subtask_configs[eef_name]:
                    if not mimic_subtask.subtask_term_signal:
                        # The last Mimic subtasks may not have an explicit term signal name
                        # so give it a default name if it doesn't already have one.
                        mimic_subtask.subtask_term_signal = f"subtask_{i}_{eef_name}_last_mimic_subtask"
                    else:
                        mimic_subtask.subtask_term_signal = (
                            f"subtask_{i}_{eef_name}_{mimic_subtask.subtask_term_signal}"
                        )
                    combined_mimic_subtask_configs[eef_name].append(mimic_subtask)

        return combined_mimic_subtask_configs

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> MimicEnvCfg:
        "Get the Mimic environment configuration for the sequential task."
        mimic_env_cfg = MimicEnvCfg()

        # Assign all default config values to mimic_env_cfg.datagen_config
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(mimic_env_cfg.datagen_config, key, value)

        mimic_env_cfg.subtask_configs = self.combine_mimic_subtask_configs(arm_mode)
        return mimic_env_cfg
