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
        desired_subtask_success_state: list[bool | None] | None,
    ) -> torch.Tensor:
        """Sequential task composite success function.

        Args:
            env: The environment instance.
            subtasks: List of subtasks that compose this sequential task.
            desired_subtask_success_state: (Optional) Precise success state for each subtask during the final time step.
                Can be used to enforce a specific current state for each subtask at the end of the episode.

        Returns:
            A bool tensor of shape (num_envs,) indicating composite success per env.
        """
        # Initialize each env's subtask success state to False if not already initialized
        if not hasattr(env, "_subtask_ever_succeeded"):
            env._subtask_ever_succeeded = [[False for _ in subtasks] for _ in range(env.num_envs)]
        # Initialize each env's current subtask index (state machine) to 0 if not already initialized
        if not hasattr(env, "_current_subtask_idx"):
            env._current_subtask_idx = [0 for _ in range(env.num_envs)]

        # Determine which subtasks need their success function evaluated.
        if desired_subtask_success_state is not None:
            subtasks_to_evaluate = range(len(subtasks))
        else:
            subtasks_to_evaluate = sorted(set(env._current_subtask_idx))

        subtask_currently_succeeding = CompositeTaskBase._evaluate_subtask_successes(
            env, subtasks, subtasks_to_evaluate
        )

        # Advance the state machine per env using the precomputed active-subtask result.
        for env_idx in range(env.num_envs):
            current_subtask_idx = env._current_subtask_idx[env_idx]
            if subtask_currently_succeeding[env_idx][current_subtask_idx]:
                env._subtask_ever_succeeded[env_idx][current_subtask_idx] = True
                if current_subtask_idx < len(subtasks) - 1:
                    env._current_subtask_idx[env_idx] += 1

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
        "Reset subtask success vector and state machine for each environment."
        # Initialize each env's subtask success state to False
        if not hasattr(env, "_subtask_ever_succeeded"):
            env._subtask_ever_succeeded = [[False for _ in subtasks] for _ in range(env.num_envs)]
        else:
            for env_id in env_ids:
                env._subtask_ever_succeeded[env_id] = [False for _ in subtasks]

        # Initialize each env's current subtask index (state machine) to 0
        if not hasattr(env, "_current_subtask_idx"):
            env._current_subtask_idx = [0 for _ in range(env.num_envs)]
        else:
            for env_id in env_ids:
                env._current_subtask_idx[env_id] = 0
