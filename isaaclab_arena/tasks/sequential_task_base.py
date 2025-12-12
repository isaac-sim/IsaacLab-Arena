# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass


@configclass
class SequentialTaskEventsCfg:
    reset_subtask_success_state: EventTermCfg = MISSING

@configclass
class TerminationsCfg:
    success: TerminationTermCfg = MISSING

class SequentialTaskBase(TaskBase):
    """
    A base class for tasks composed sequentially from multiple subtasks.
    The sequential task takes a list of TaskBase instances (subtasks),
    and collects configs to form a composite task.
    """

    def __init__(self, subtasks: list[TaskBase], episode_length_s: float | None = None):
        super().__init__(episode_length_s)
        assert len(subtasks) > 0, "SequentialTaskBase requires at least one subtask"
        self.subtasks = subtasks

    @staticmethod
    def _add_suffix_to_configclass_fields(cfg_instance: Any, suffix: str) -> Any:
        """Create a new configclass instance with all field names appended with a suffix."""
        if cfg_instance is None:
            return None
            
        fields = dataclasses.fields(cfg_instance)
        new_fields = []
        field_values = {}

        # Rename the fields with suffix
        for field in fields:
            new_name = f"{field.name}{suffix}"
            value = getattr(cfg_instance, field.name)
            new_fields.append((new_name, field.type, value))
            field_values[new_name] = value
        
        # Create a new configclass with renamed fields
        new_cfg_class = make_configclass(type(cfg_instance).__name__, new_fields)
        return new_cfg_class(**field_values)

    @staticmethod
    def _remove_configclass_fields(cfg_instance: Any, exclude_fields: set[str]) -> Any:
        "Remove all fields from the configclass instance that are in the exclude_fields set."
        if cfg_instance is None:
            return None
        
        fields = dataclasses.fields(cfg_instance)
        new_fields = []
        field_values = {}
        
        # Remove the fields that are in the exclude_fields set
        for field in fields:
            if field.name in exclude_fields:
                continue
            value = getattr(cfg_instance, field.name)
            new_fields.append((field.name, field.type, value))
            field_values[field.name] = value
        
        if not new_fields:
            return None
        
        # Create a new configclass without the excluded fields
        new_cfg_class = make_configclass(type(cfg_instance).__name__, new_fields)
        return new_cfg_class(**field_values)

    @staticmethod
    def sequential_task_success_func(
        env,
        task_instance: "SequentialTaskBase",
    ) -> torch.Tensor:
        "Sequential task compositesuccess function."
        # Check success of current subtask for each env
        for env_idx in range(env.num_envs):
            current_subtask_idx = env._current_subtask_idx[env_idx]
            current_subtask_success_func = task_instance.subtasks[current_subtask_idx].get_termination_cfg().success.func
            current_subtask_success_params = task_instance.subtasks[current_subtask_idx].get_termination_cfg().success.params
            result = current_subtask_success_func(env, **current_subtask_success_params)[env_idx]

            if result:
                env._subtask_success_state[env_idx][current_subtask_idx] = True
                if current_subtask_idx < len(task_instance.subtasks) - 1:
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
        task_instance: "SequentialTaskBase",
    ) -> None:
        "Reset subtask success vector and state machine for each environment."
        # Initialize each env's subtask success state to False
        if not hasattr(env, "_subtask_success_state"):
            env._subtask_success_state = [[False for _ in task_instance.subtasks] for _ in range(env.num_envs)]
        else:
            env._subtask_success_state[env_ids] = [False for _ in task_instance.subtasks]
        
        # Initialize each env's current subtask index (state machine) to 0
        if not hasattr(env, "_current_subtask_idx"):
            env._current_subtask_idx = [0 for _ in range(env.num_envs)]
        else:
            env._current_subtask_idx[env_ids] = 0
        
    def get_scene_cfg(self) -> configclass:
        "Make combined scene cfg from all subtasks."
        scene_cfg = combine_configclass_instances(
            "SceneCfg", *(subtask.get_scene_cfg() for subtask in self.subtasks)
        )

        return scene_cfg

    def make_sequential_task_events_cfg(self) -> configclass:
        "Make event to reset subtask success state."
        reset_subtask_success_state = EventTermCfg(
            func=self.reset_subtask_success_state,
            mode="reset",
            params={
                "task_instance": self,
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
            renamed_cfg = self._add_suffix_to_configclass_fields(subtask_events_cfg, f"_subtask_{i}")
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
                "task_instance": self,
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
            # Remove the 'success' field from the subtask termination cfg
            cleaned_cfg = self._remove_configclass_fields(termination_cfg, {"success"})
            if cleaned_cfg is not None:
                subtask_termination_cfgs.append(cleaned_cfg)

        # Combine subtask terminations with the composite sequential task success
        combined_termination_cfg = combine_configclass_instances(
            "TerminationsCfg", *subtask_termination_cfgs, self.make_sequential_task_termination_cfg()
        )

        return combined_termination_cfg









