# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import numpy as np
import torch
import warnings
from dataclasses import MISSING
from functools import partial
from typing import Any

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.subtask_windowed_metric import (
    WINDOW_SCOPE_FROM_EPISODE_START,
    SubtaskWindowedMetric,
)
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import (
    check_configclass_field_duplicates,
    combine_configclass_instances,
    transform_configclass_instance,
)


@configclass
class CompositeTaskEventsCfg:
    reset_subtask_success_state: EventTermCfg = MISSING


@configclass
class TerminationsCfg:
    success: TerminationTermCfg = MISSING


class SubtaskSuccessStateRecorder(RecorderTerm):
    """Records the subtask success state just before the environment is reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name

    def record_post_step(self):
        # Return subtask success state as a torch tensor
        subtask_ever_succeeded = torch.tensor(self._env._subtask_ever_succeeded, device=self._env.device)
        return self.name, subtask_ever_succeeded.clone()


@configclass
class SubtaskSuccessStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = SubtaskSuccessStateRecorder
    name: str = "subtask_success_rate"


class SubtaskSuccessRateMetric(MetricBase):
    """Computes the per-subtask success rates.

    Returns a dict with success rate for each subtask.
    """

    name = "subtask_success_rate"
    recorder_term_name = "subtask_success_rate"

    def __init__(self):
        super().__init__()

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the subtask success state metric."""
        return SubtaskSuccessStateRecorderCfg(name=self.recorder_term_name)

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> list:
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


class CompositeTaskBase(TaskBase):
    """
    A base class for composite tasks composed of multiple subtasks.
    Completion ordering of subtasks does not matter.

    Args:
        subtasks: List of TaskBase instances representing the subtasks that compose this composite task.
        episode_length_s: Maximum duration of a single episode in seconds. If None, no time limit is enforced.
        desired_subtask_success_state: (Optional) Precise success state for each subtask during the final time step.
            Can be used to enforce a specific current state for each subtask at the end of the episode.
        window_subtask_metrics: If True, each per-subtask metric is also reported under a ``_windowed`` suffix
            evaluated only over the subtask's own window (see ``SUBTASK_WINDOW_SCOPE``).
    """

    # Scope used to derive each subtask's evaluation window when
    # ``window_subtask_metrics`` is enabled. Subclasses override this — sequential
    # tasks use the previous-subtask-completion as the start of the window.
    SUBTASK_WINDOW_SCOPE: str = WINDOW_SCOPE_FROM_EPISODE_START

    def __init__(
        self,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
        desired_subtask_success_state: list[bool | None] | None = None,
        window_subtask_metrics: bool = False,
    ):
        super().__init__(episode_length_s)
        assert len(subtasks) > 0, "Composite task requires at least one subtask"
        self.subtasks = subtasks

        if desired_subtask_success_state is not None:
            assert len(desired_subtask_success_state) == len(
                subtasks
            ), "Desired subtask success state must be the same length as the number of subtasks"
            assert all(
                s is None or isinstance(s, bool) for s in desired_subtask_success_state
            ), "Desired subtask success state entries must each be True, False, or None"
        self.desired_subtask_success_state = desired_subtask_success_state
        self.window_subtask_metrics = window_subtask_metrics

    @staticmethod
    def _add_suffix_configclass_transform(fields: list[tuple], suffix: str) -> list[tuple]:
        "Config transformation to add a suffix to all field names."
        return [(f"{name}{suffix}", ftype, value) for name, ftype, value in fields]

    @staticmethod
    def _remove_configclass_transform(fields: list[tuple], exclude_fields: set[str]) -> list[tuple]:
        "Config transformation to remove all fields in an exclude set."
        return [(name, ftype, value) for name, ftype, value in fields if name not in exclude_fields]

    @staticmethod
    def _evaluate_subtask_successes(
        env,
        subtasks: list[TaskBase],
        subtask_indices,
    ) -> list[list[bool]]:
        """Evaluate the success function of selected subtasks across all envs.

        Args:
            env: The environment instance.
            subtasks: Full list of subtasks for this composite task.
            subtask_indices: Iterable of subtask indices to evaluate. Indices not in this
                iterable are left as False in the returned matrix.

        Returns:
            A (num_envs x len(subtasks)) list of bools, where entry [env_idx][subtask_idx]
            is True if that subtask's success function returned True this step.
        """
        subtask_currently_succeeding = [[False for _ in subtasks] for _ in range(env.num_envs)]
        for subtask_idx in subtask_indices:
            subtask_success_func = subtasks[subtask_idx].get_termination_cfg().success.func
            subtask_success_params = subtasks[subtask_idx].get_termination_cfg().success.params
            results = subtask_success_func(env, **subtask_success_params)
            for env_idx in range(env.num_envs):
                if results[env_idx]:
                    subtask_currently_succeeding[env_idx][subtask_idx] = True
        return subtask_currently_succeeding

    @staticmethod
    def composite_task_success_func(
        env,
        subtasks: list[TaskBase],
        desired_subtask_success_state: list[bool | None] | None,
    ) -> torch.Tensor:
        """Composite task composite success function.

        Args:
            env: The environment instance.
            subtasks: List of subtasks that compose this composite task.
            desired_subtask_success_state: (Optional) Precise success state for each subtask during the final time step.
                Can be used to enforce a specific current state for each subtask at the end of the episode.

        Returns:
            A bool tensor of shape (num_envs,) indicating composite success per env.
        """
        # Initialize each env's subtask success state to False if not already initialized
        if not hasattr(env, "_subtask_ever_succeeded"):
            env._subtask_ever_succeeded = [[False for _ in subtasks] for _ in range(env.num_envs)]

        # Evaluate every subtask's success function (composite tasks have no ordering constraint).
        subtask_currently_succeeding = CompositeTaskBase._evaluate_subtask_successes(
            env, subtasks, range(len(subtasks))
        )
        for env_idx in range(env.num_envs):
            for subtask_idx in range(len(subtasks)):
                if subtask_currently_succeeding[env_idx][subtask_idx]:
                    env._subtask_ever_succeeded[env_idx][subtask_idx] = True

        # Compute composite task success state for each env.
        # Entries in `desired_subtask_success_state` set to None are "don't cares" and
        # may be any state. For each subtask it must (a) have been evaluated as True
        # at some point and (b) currently match the desired value.
        if desired_subtask_success_state is not None:
            per_env_success = []
            for env_idx in range(env.num_envs):
                env_success = True
                for i, desired in enumerate(desired_subtask_success_state):
                    if desired is None:
                        continue
                    # Check that both the subtask has ever succeeded and currently matches the desired success state.
                    ever_succeeded = env._subtask_ever_succeeded[env_idx][i]
                    currently_matches = subtask_currently_succeeding[env_idx][i] == desired
                    if not (ever_succeeded and currently_matches):
                        env_success = False
                        break
                per_env_success.append(env_success)
        else:
            per_env_success = [all(env_successes) for env_successes in env._subtask_ever_succeeded]

        success_tensor = torch.tensor(per_env_success, dtype=torch.bool, device=env.device)

        env.extras["subtask_success_state"] = copy.copy(env._subtask_ever_succeeded)

        return success_tensor

    @staticmethod
    def reset_subtask_success_state(
        env,
        env_ids,
        subtasks: list[TaskBase],
    ) -> None:
        "Reset subtask success vector for each environment."
        # Initialize each env's subtask success state to False
        if not hasattr(env, "_subtask_ever_succeeded"):
            env._subtask_ever_succeeded = [[False for _ in subtasks] for _ in range(env.num_envs)]
        else:
            for env_id in env_ids:
                env._subtask_ever_succeeded[env_id] = [False for _ in subtasks]

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

    def _make_composite_task_events_cfg(self) -> Any:
        "Make event to reset subtask success state."
        reset_subtask_success_state = EventTermCfg(
            func=self.reset_subtask_success_state,
            mode="reset",
            params={
                "subtasks": self.subtasks,
            },
        )

        return CompositeTaskEventsCfg(
            reset_subtask_success_state=reset_subtask_success_state,
        )

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

        # Add reset subtask success state event to the combined events cfgs
        events_cfg = combine_configclass_instances(
            "EventsCfg", *renamed_events_cfgs, self._make_composite_task_events_cfg()
        )

        return events_cfg

    def _make_composite_task_termination_cfg(self) -> Any:
        "Make composite success check termination term."
        success = TerminationTermCfg(
            func=self.composite_task_success_func,
            params={
                "subtasks": self.subtasks,
                "desired_subtask_success_state": self.desired_subtask_success_state,
            },
        )

        return TerminationsCfg(
            success=success,
        )

    def get_termination_cfg(self) -> Any:
        "Make combined termination cfg from all subtasks."
        # Collect termination cfgs from subtasks with 'success' field removed
        subtask_termination_cfgs = []
        for subtask in self.subtasks:
            termination_cfg = subtask.get_termination_cfg()
            cleaned_cfg = transform_configclass_instance(
                termination_cfg, partial(self._remove_configclass_transform, exclude_fields={"success"})
            )
            # cleaned_cfg is None when the subtask's only termination field was 'success'
            if cleaned_cfg is not None:
                subtask_termination_cfgs.append(cleaned_cfg)

        # Combine subtask terminations with the composite sequential task success
        combined_termination_cfg = combine_configclass_instances(
            "TerminationsCfg", *subtask_termination_cfgs, self._make_composite_task_termination_cfg()
        )

        return combined_termination_cfg

    def _combine_subtask_metrics(self, subtask_idxs: list[int]) -> list[MetricBase]:
        """Combine metrics from subtasks with the given ids.

        Per-subtask "success_rate" metrics are intentionally collapsed into a single shared entry —
        per-subtask success is reported separately via SubtaskSuccessRateMetric (added in get_metrics).
        When ``self.window_subtask_metrics`` is True, each non-success-rate metric is additionally
        reported under a ``_windowed`` suffix evaluated over the subtask's own window.
        """
        combined_metrics = []

        for subtask_idx in subtask_idxs:
            subtask_metrics = self.subtasks[subtask_idx].get_metrics()
            for metric in subtask_metrics:
                if metric.name != "success_rate":
                    metric.name = f"{metric.name}_subtask_{subtask_idx}"
                    metric.recorder_term_name = f"{metric.recorder_term_name}_subtask_{subtask_idx}"
                    renamed_metric = copy.copy(metric)
                    # Episode-wide reduction is always reported.
                    combined_metrics.append(renamed_metric)
                    # When opting into windowing, also report the subtask-scoped reduction
                    # under a ``_windowed`` suffix so both views are available side-by-side.
                    # The wrapper shares ``recorder_term_name`` with the unwrapped metric;
                    # dedup happens in ``metrics_to_recorder_manager_cfg``.
                    if self.window_subtask_metrics:
                        combined_metrics.append(
                            SubtaskWindowedMetric(
                                inner_metric=copy.copy(renamed_metric),
                                subtask_idx=subtask_idx,
                                scope=self.SUBTASK_WINDOW_SCOPE,
                                name=f"{renamed_metric.name}_windowed",
                            )
                        )
                else:
                    if not any(m.name == "success_rate" for m in combined_metrics):
                        combined_metrics.append(copy.copy(metric))

        return combined_metrics

    def get_metrics(self) -> list[MetricBase]:
        "Get metrics for the composite task."
        subtask_metrics = self._combine_subtask_metrics([i for i in range(len(self.subtasks))])
        # Add the composite task's own metric for per-subtask success rates
        subtask_metrics.append(SubtaskSuccessRateMetric())

        return subtask_metrics

    def get_own_fine_grained_subtasks(self) -> list[FineGrainedSubtask]:
        """Composite-level fine-grained recipes (e.g. cross-child conditions).

        These are added on top of whatever recipes the children declare. They
        carry ``parent_subtask_idx = None`` and therefore have no activation
        gating — they remain active for the full episode.
        """
        return []

    def get_fine_grained_subtasks(self) -> list[FineGrainedSubtask]:
        """Concatenate children's fine-grained recipes with namespace prefixes.

        Each child's recipe gets a new name (``f"subtask_{i}/{original_name}"``)
        and a ``parent_subtask_idx = i`` tag. For ``SequentialTaskBase`` the
        state machine reads ``parent_subtask_idx`` to gate advancement against
        ``env._current_subtask_idx`` so a child's recipes only advance while
        that child is the active parent subtask. For unordered
        ``CompositeTaskBase`` (no ``_current_subtask_idx``) the gate is a no-op
        and every recipe is always active.

        Composite-level recipes from ``get_own_fine_grained_subtasks`` are
        appended afterward and are not gated.
        """
        out: list[FineGrainedSubtask] = []
        for i, child in enumerate(self.subtasks):
            for fgs in child.get_fine_grained_subtasks():
                out.append(
                    dataclasses.replace(
                        fgs,
                        name=f"subtask_{i}/{fgs.name}",
                        parent_subtask_idx=i,
                    )
                )
        out.extend(self.get_own_fine_grained_subtasks())
        return out

    def _validate_consistent_mimic_eef_names(self, arm_mode: ArmMode) -> set[str]:
        "Check that all subtasks have the same Mimic eef_names; return that set."
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
