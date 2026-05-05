# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import torch

from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
from isaaclab_arena.tasks.task_base import TaskBase


class SequentialTaskBase(CompositeTaskBase):
    """
    A class for composite tasks composed sequentially from multiple subtasks.
    The sequential task takes a list of TaskBase instances (subtasks),
    and automatically collects configs to form a composite task.

    The sequential task satisfies the following properties:
        - Made up of atomic tasks that must be completed in order.
        - Once a subtask is complete once (success = True), it's success state can go back to False
          without affecting the completeness of the overall sequential task.
    """

    @staticmethod
    def composite_task_success_func(
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

        current_subtask_success_state = [[False for _ in subtasks] for _ in range(env.num_envs)]

        # Check success of subtask for each env
        for env_idx in range(env.num_envs):
            if desired_subtask_success_state:
                # Compute the success state for all subtasks
                for subtask_idx in range(len(subtasks)):
                    subtask_success_func = subtasks[subtask_idx].get_termination_cfg().success.func
                    subtask_success_params = subtasks[subtask_idx].get_termination_cfg().success.params
                    result = subtask_success_func(env, **subtask_success_params)[env_idx]
                    if result:
                        current_subtask_success_state[env_idx][subtask_idx] = True

            # Compute the success state for the current subtask
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
                all(env._subtask_success_state[env_idx])
                and current_subtask_success_state[env_idx] == desired_subtask_success_state
                for env_idx in range(env.num_envs)
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
