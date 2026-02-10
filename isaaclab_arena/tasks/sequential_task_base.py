# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
from dataclasses import MISSING
from functools import partial

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import (
    check_configclass_field_duplicates,
    combine_configclass_instances,
    transform_configclass_instance,
)


@configclass
class SequentialTaskEventsCfg:
    reset_subtask_success_state: EventTermCfg = MISSING


@configclass
class TerminationsCfg:
    success: TerminationTermCfg = MISSING


class SubtaskSuccessStateRecorder(RecorderTerm):
    """Records the subtask success state just before the environment is reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name

    def record_post_reset(self, env_ids):
        # Return subtask success state as a torch tensor
        subtask_success_state = torch.tensor(self._env._subtask_success_state, device=self._env.device)
        return self.name, subtask_success_state.clone()

    def record_pre_reset(self, env_ids):
        # Skip recording before subtask_success_state exists
        if not hasattr(self._env, "_subtask_success_state"):
            return None, None
        # Return subtask success state as a torch tensor
        subtask_success_state = torch.tensor(self._env._subtask_success_state, device=self._env.device)
        return self.name, subtask_success_state.clone()

    def record_post_step(self):
        # Return subtask success state as a torch tensor
        subtask_success_state = torch.tensor(self._env._subtask_success_state, device=self._env.device)
        return self.name, subtask_success_state.clone()


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


class SequentialTaskBase(TaskBase):
    """
    A base class for composite tasks composed sequentially from multiple subtasks.
    The sequential task takes a list of TaskBase instances (subtasks),
    and automatically collects configs to form a composite task.

    The sequential task satisfies the following properties:
        - Made up of atomic tasks that must be completed in order.
        - Once a subtask is complete once (success = True), it's success state can go back to False
          without affecting the completeness of the overall sequential task.
    """

    def __init__(
        self,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
        desired_subtask_success_state: list[bool] | None = None,
    ):
        super().__init__(episode_length_s)
        assert len(subtasks) > 0, "SequentialTaskBase requires at least one subtask"
        self.subtasks = subtasks

        if desired_subtask_success_state is not None:
            assert len(desired_subtask_success_state) == len(
                subtasks
            ), "Desired subtask success state must be the same length as the number of subtasks"
        self.desired_subtask_success_state = desired_subtask_success_state

    @staticmethod
    def add_suffix_configclass_transform(fields: list[tuple], suffix: str) -> list[tuple]:
        "Config transformation to add a suffix to all field names."
        return [(f"{name}{suffix}", ftype, value) for name, ftype, value in fields]

    @staticmethod
    def remove_configclass_transform(fields: list[tuple], exclude_fields: set[str]) -> list[tuple]:
        "Config transformation to remove all fields in an exclude set."
        return [(name, ftype, value) for name, ftype, value in fields if name not in exclude_fields]

    @staticmethod
    def sequential_task_success_func(
        env,
        subtasks: list[TaskBase],
        desired_subtask_success_state: list[bool] | None,
    ) -> torch.Tensor:
        "Sequential task composite success function."
        # Initialize each env's subtask success state to False if not already initialized
        if not hasattr(env, "_subtask_success_state"):
            env._subtask_success_state = [[False for _ in subtasks] for _ in range(env.num_envs)]
        # Initialize each env's current subtask index (state machine) to 0 if not already initialized
        if not hasattr(env, "_current_subtask_idx"):
            env._current_subtask_idx = [0 for _ in range(env.num_envs)]

        # Check success of current subtask for each env
        for env_idx in range(env.num_envs):
            current_subtask_idx = env._current_subtask_idx[env_idx]
            current_subtask_success_func = subtasks[current_subtask_idx].get_termination_cfg().success.func
            current_subtask_success_params = subtasks[current_subtask_idx].get_termination_cfg().success.params
            result = current_subtask_success_func(env, **current_subtask_success_params)[env_idx]

            if result:
                env._subtask_success_state[env_idx][current_subtask_idx] = True
                if current_subtask_idx < len(subtasks) - 1:
                    env._current_subtask_idx[env_idx] += 1

        # Compute composite task success state for each env
        if desired_subtask_success_state:
            per_env_success = [
                all([a == b for a, b in zip(env_successes, desired_subtask_success_state)])
                for env_successes in env._subtask_success_state
            ]
        else:
            per_env_success = [all(env_successes) for env_successes in env._subtask_success_state]

        success_tensor = torch.tensor(per_env_success, dtype=torch.bool, device=env.device)

        env.extras["subtask_success_state"] = copy.copy(env._subtask_success_state)

        return success_tensor

    @staticmethod
    def reset_subtask_success_state(
        env,
        env_ids,
        subtasks: list[TaskBase],
    ) -> None:
        "Reset subtask success vector and state machine for each environment."
        # Initialize each env's subtask success state to False
        if not hasattr(env, "_subtask_success_state"):
            env._subtask_success_state = [[False for _ in subtasks] for _ in range(env.num_envs)]
        else:
            for env_id in env_ids:
                env._subtask_success_state[env_id] = [False for _ in subtasks]

        # Initialize each env's current subtask index (state machine) to 0
        if not hasattr(env, "_current_subtask_idx"):
            env._current_subtask_idx = [0 for _ in range(env.num_envs)]
        else:
            for env_id in env_ids:
                env._current_subtask_idx[env_id] = 0

    def get_scene_cfg(self) -> configclass:
        "Make combined scene cfg from all subtasks."
        # Check for duplicate fields across subtask scene configs and warn if found
        duplicates = check_configclass_field_duplicates(*(subtask.get_scene_cfg() for subtask in self.subtasks))
        if duplicates:
            import warnings

            warnings.warn(
                f"\n[WARNING] Duplicate scene config fields found across subtasks: {duplicates}. "
                "Duplicates will be ignored.\n",
                UserWarning,
            )

        scene_cfg = combine_configclass_instances("SceneCfg", *(subtask.get_scene_cfg() for subtask in self.subtasks))
        return scene_cfg

    def make_sequential_task_events_cfg(self) -> configclass:
        "Make event to reset subtask success state."
        reset_subtask_success_state = EventTermCfg(
            func=self.reset_subtask_success_state,
            mode="reset",
            params={
                "subtasks": self.subtasks,
            },
        )

        return SequentialTaskEventsCfg(
            reset_subtask_success_state=reset_subtask_success_state,
        )

    def get_events_cfg(self) -> configclass:
        "Make combined events cfg from all subtasks."
        # Collect events_cfgs from subtasks with renamed fields to avoid collisions
        renamed_events_cfgs = []
        for i, subtask in enumerate(self.subtasks):
            subtask_events_cfg = subtask.get_events_cfg()
            renamed_cfg = transform_configclass_instance(
                subtask_events_cfg, partial(self.add_suffix_configclass_transform, suffix=f"_subtask_{i}")
            )
            if renamed_cfg is not None:
                renamed_events_cfgs.append(renamed_cfg)

        # Add reset subtask success state event to the combined events cfgs
        events_cfg = combine_configclass_instances(
            "EventsCfg", *renamed_events_cfgs, self.make_sequential_task_events_cfg()
        )

        return events_cfg

    def make_sequential_task_termination_cfg(self) -> configclass:
        "Make composite success check termination term."
        success = TerminationTermCfg(
            func=self.sequential_task_success_func,
            params={
                "subtasks": self.subtasks,
                "desired_subtask_success_state": self.desired_subtask_success_state,
            },
        )

        return TerminationsCfg(
            success=success,
        )

    def get_termination_cfg(self) -> configclass:
        "Make combined termination cfg from all subtasks."
        # Collect termination cfgs from subtasks with 'success' field removed
        subtask_termination_cfgs = []
        for subtask in self.subtasks:
            termination_cfg = subtask.get_termination_cfg()
            cleaned_cfg = transform_configclass_instance(
                termination_cfg, partial(self.remove_configclass_transform, exclude_fields={"success"})
            )
            if cleaned_cfg is not None:
                subtask_termination_cfgs.append(cleaned_cfg)

        # Combine subtask terminations with the composite sequential task success
        combined_termination_cfg = combine_configclass_instances(
            "TerminationsCfg", *subtask_termination_cfgs, self.make_sequential_task_termination_cfg()
        )

        return combined_termination_cfg

    def combine_subtask_metrics(self, subtask_idxs: list[int]) -> list[MetricBase]:
        "Combine metrics from subtasks with the given ids."
        combined_metrics = []

        for subtask_idx in subtask_idxs:
            subtask_metrics = self.subtasks[subtask_idx].get_metrics()
            for metric in subtask_metrics:
                if metric.name != "success_rate":
                    metric.name = f"{metric.name}_subtask_{subtask_idx}"
                    metric.recorder_term_name = f"{metric.recorder_term_name}_subtask_{subtask_idx}"
                    combined_metrics.append(copy.copy(metric))
                else:
                    if not any(m.name == "success_rate" for m in combined_metrics):
                        combined_metrics.append(copy.copy(metric))

        return combined_metrics

    def get_metrics(self) -> list[MetricBase]:
        "Get metrics for the sequential task."
        subtask_metrics = self.combine_subtask_metrics([i for i in range(len(self.subtasks))])
        # Add the sequential task's own metric for per-subtask success rates
        subtask_metrics.append(SubtaskSuccessRateMetric())

        return subtask_metrics

    def combine_mimic_subtask_configs(self, arm_mode: ArmMode) -> dict[str, list[SubTaskConfig]]:
        "Combine the Mimic subtask configs for all subtasks."
        # Check that all subtasks have the same Mimic eef_names
        mimic_eef_names = set(self.subtasks[0].get_mimic_env_cfg(arm_mode).subtask_configs.keys())

        for subtask in self.subtasks[1:]:
            subtask_eef_names_set = set(subtask.get_mimic_env_cfg(arm_mode).subtask_configs.keys())
            if subtask_eef_names_set != mimic_eef_names:
                raise ValueError(
                    f"All subtasks must have the same Mimic eef_names.\nSubtask 0 has eef_names: {mimic_eef_names}, but"
                    f" subtask {self.subtasks.index(subtask)} has eef_names: {subtask_eef_names_set}."
                )

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
