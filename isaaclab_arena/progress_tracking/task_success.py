# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Connect task progress to Isaac Lab's termination evaluation and reset lifecycle."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
from isaaclab_arena.tasks.predicates.object_settling import reset_rest_pose_recorder


class TaskSuccessTerm(ManagerTermBase):
    """Determine task success using ProgressTracker.

    ArenaEnvBuilder registers this term with Isaac Lab's TerminationManager.
    TaskSuccessTerm creates and owns ProgressTracker. TerminationManager
    calls this term to update progress and check the task's success
    requirements. On episode resets, TerminationManager calls
    this term's reset() to clear progress for the restarting environments.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        # Isaac Lab validates required __call__ parameters before constructing this term.
        success_objectives: list[ProgressObjective] = cfg.params["success_objectives"]
        assert success_objectives, "Task success requires at least one success objective."
        assert env.progress_tracker is None, "Only one root term may own task progress."
        self._progress_tracker = ProgressTracker(
            success_objectives,
            num_envs=env.num_envs,
            device=env.device,
            env=env,
            subtasks_are_sequential=cfg.params.get("subtasks_are_sequential", False),
            desired_subtask_success_state=cfg.params.get("desired_subtask_success_state"),
        )
        self._environment_ids = torch.arange(env.num_envs, device=env.device)
        env._progress_tracker = self._progress_tracker

    def __call__(
        self,
        env,
        success_objectives: list[ProgressObjective],
        subtasks_are_sequential: bool = False,
        desired_subtask_success_state: list[bool | None] | None = None,
    ) -> torch.Tensor:
        """Update ProgressTracker and return whether the task's success requirements are met in each environment."""
        self._progress_tracker.step(env, step_index=env.episode_length_buf)
        return self._progress_tracker.is_complete()

    def reset(self, env_ids=None) -> None:
        """Clear progress and initial resting positions for the restarting environments."""
        selected_env_ids = self._environment_ids if env_ids is None else self._environment_ids[env_ids]
        self._progress_tracker.reset(selected_env_ids)
        # TODO(cvolk): Consider a shared Arena reset hook in IsaacLabArenaManagerBasedRLEnv.
        # Revisit this if ObjectInitialRestPoseRecorder is used independently of task success.
        reset_rest_pose_recorder(self._env, selected_env_ids)
