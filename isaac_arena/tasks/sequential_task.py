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

from typing import Any, List

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaac_arena.metrics.metric_base import MetricBase
from isaac_arena.tasks.task_base import TaskBase


@configclass
class NoTerminationCfg:
    """Termination configuration that never terminates (for intermediate tasks)."""

    # Only timeout termination, no success termination for intermediate tasks
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)


class CompositeTask(TaskBase):
    """A task that chains multiple tasks in sequence.

    Only the final task in the sequence will trigger environment termination.
    Intermediate tasks are monitored for success to enable transitions.
    """

    def __init__(self, tasks: list[TaskBase]):
        """Initialize sequential task with a list of tasks to execute in order.

        Args:
            tasks: List of tasks to execute sequentially. Only the final task
                  will trigger environment termination.
        """
        super().__init__()
        if not tasks:
            raise ValueError("Sequential task requires at least one task")

        self.tasks = tasks
        self.current_task_index = 0
        self.task_completion_states = [False] * len(tasks)

        # Cache configurations to avoid recomputation
        # TODO(cvolk): Combine multiple scene configs similarly to compile_env()
        # TODO(cvolk): How to check if feasable task combination?
        self._scene_cfg = None
        self._events_cfg = None
        self._metrics = None
        self._mimic_env_cfg = None

    def get_scene_cfg(self) -> Any:
        """Combine scene configurations from all tasks."""
        return self._scene_cfg

    # TODO(cvolk): Type annotation
    # TODO(cvolk): How do we detect subtermination success?
    def get_termination_cfg(self) -> Any:
        """Only return termination config for the FINAL task.

        Intermediate tasks use NoTerminationCfg to prevent environment termination.
        """
        return tasks[-1].get_termination_cfg()

    def get_events_cfg(self) -> Any:
        """Get events configuration from current task."""
        return self._events_cfg

    def get_current_task(self) -> TaskBase:
        """Get the currently active task."""
        return self.tasks[self.current_task_index]

    def is_final_task(self) -> bool:
        """Check if we're on the final task in the sequence."""
        return self.current_task_index == len(self.tasks) - 1

    def get_completed_tasks(self) -> list[TaskBase]:
        """Get list of completed tasks."""
        return [task for i, task in enumerate(self.tasks) if self.task_completion_states[i]]

    def get_remaining_tasks(self) -> list[TaskBase]:
        """Get list of remaining tasks (including current)."""
        return self.tasks[self.current_task_index :]

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        """Get mimic environment configuration from current task."""
        return self._mimic_env_cfg

    def get_metrics(self) -> list[MetricBase]:
        """Combine metrics from all tasks in the sequence."""
        return self._metrics

    def check_current_task_success(self, env) -> bool:
        """Check if current task succeeded (for internal transition logic).

        Args:
            env: The environment instance to check against

        Returns:
            bool: True if current task has succeeded
        """
        current_task = self.get_current_task()
        termination_cfg = current_task.get_termination_cfg()

        # Check if termination config has a success condition
        if hasattr(termination_cfg, "success"):
            success_condition = termination_cfg.success
            try:
                return bool(success_condition.func(env, **success_condition.params)[0])
            except Exception as e:
                print(f"Warning: Error checking task success: {e}")
                return False

        return False

    # TODO(cvolk): How is this triggered when compiled through IsaacLab
    def try_advance_task(self, env) -> bool:
        """Try to advance to next task if current one is complete.

        Args:
            env: The environment instance to check against

        Returns:
            bool: True if task was advanced, False otherwise
        """
        if not self.is_final_task() and self.check_current_task_success(env):
            # Mark current task as completed
            self.task_completion_states[self.current_task_index] = True

            # Advance to next task
            self.current_task_index += 1

            # Clear cached configurations since we switched tasks
            self._scene_cfg = None
            self._events_cfg = None
            self._mimic_env_cfg = None

            print(f"Sequential Task: Advanced to task {self.current_task_index + 1}/{len(self.tasks)}")
            return True  # Task advanced

        return False  # No advancement

    def reset_sequence(self):
        """Reset the task sequence to the beginning."""
        self.current_task_index = 0
        self.task_completion_states = [False] * len(self.tasks)

        # Clear cached configurations
        self._scene_cfg = None
        self._events_cfg = None
        self._mimic_env_cfg = None

        print("Sequential Task: Reset to beginning of sequence")
