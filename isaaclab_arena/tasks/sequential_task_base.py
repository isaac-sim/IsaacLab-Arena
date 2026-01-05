# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
from dataclasses import MISSING
from functools import partial

from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

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

    # TODO: peterd - add functions to process Mimic and Metrics configs.

    def __init__(self, subtasks: list[TaskBase], episode_length_s: float | None = None):
        super().__init__(episode_length_s)
        assert len(subtasks) > 0, "SequentialTaskBase requires at least one subtask"
        self.subtasks = subtasks

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
