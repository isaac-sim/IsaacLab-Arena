# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lab demo tools retain Arena's temporal success and partial-reset lifecycle."""

import torch
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_external_success(simulation_app, mode):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationManager, TerminationTermCfg
    from isaaclab.managers.recorder_manager import (
        DatasetExportMode,
        RecorderManagerBaseCfg,
        RecorderTerm,
        RecorderTermCfg,
    )

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.task_progress_cfg import TaskProgressCfg
    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tasks.terminations import task_success_from_progress

    predicate_calls = []
    recorded_success = []
    terminal_success = []

    def predicate(env):
        predicate_calls.append(env.episode_length_buf.clone())
        return torch.ones(env.num_envs, dtype=torch.bool)

    class SuccessObserver(RecorderTerm):
        def record_post_step(self):
            recorded_success.append(self._env.progress_tracker.is_complete().tolist())
            return None, None

    # Recording replaces success; replay removes all termination and recorder terms.
    success_cfg = TerminationTermCfg(func=task_success_from_progress)
    if mode == "native":
        termination_cfg = {"success": success_cfg}
    elif mode == "record":
        termination_cfg = {"success": TerminationTermCfg(func=lambda env: torch.zeros(2, dtype=torch.bool))}
    else:
        termination_cfg = {}
    is_replay = mode in ("replay", "replay_disabled_recorder")
    recorder_cfg = {} if is_replay else RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    if mode == "replay_disabled_recorder":
        recorder_cfg["disabled_recorder"] = None
    if not is_replay:
        recorder_cfg.success_observer = RecorderTermCfg(class_type=SuccessObserver)
    env = IsaacLabArenaManagerBasedRLEnv.__new__(IsaacLabArenaManagerBasedRLEnv)
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=2),
        sim=SimpleNamespace(device="cpu"),
        terminations=termination_cfg,
        recorders=recorder_cfg,
        metrics=None,
        episode_recorders=None,
        task_progress=TaskProgressCfg(
            success_criteria=[
                CompletionCriteria(
                    name="stable",
                    predicate_sequence=[TrueForConsecutiveStepsCfg(predicate=predicate, required_steps=3)],
                )
            ]
        ),
    )
    env.sim = SimpleNamespace(is_playing=lambda: True, device="cpu")
    env.scene = SimpleNamespace(num_envs=2)
    env.extras = {}
    env._arena_world = None
    env._progress_tracker = None
    env._task_progress_step = -1
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 0
    env._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(num_envs=2, device="cpu")
    env._active_episode_mask = torch.ones(2, dtype=torch.bool)
    env._reset_env_ids = torch.empty(0, dtype=torch.long)
    env._episode_indices = {0: 0, 1: 0}
    env._episode_limit = None
    env._started_episode_count = 2
    env._completed_episode_count = 0

    def load_lab_managers(self):
        assert self.cfg.recorders is None
        self.termination_manager = TerminationManager(self.cfg.terminations, self)

    with (
        patch.object(ManagerBasedRLEnv, "load_managers", load_lab_managers),
        patch("isaaclab_arena.environments.isaaclab_arena_manager_based_env.ArenaWorld"),
    ):
        env.load_managers()
    recorder = env.recorder_manager
    assert env.progress_tracker is not None
    assert "progress_tracking" in recorder.active_terms
    assert env.cfg.recorders.dataset_export_mode == DatasetExportMode.EXPORT_NONE
    assert not success_cfg.func(env, **success_cfg.params).any(), "A query before stepping must not advance progress"
    assert not predicate_calls

    for step in range(1, 4):
        env.episode_length_buf += 1
        env.common_step_counter += 1
        terminated = env.termination_manager.compute()
        assert terminated.tolist() == ([step == 3] * 2 if mode == "native" else [False, False])
        recorder.record_post_step()
        # Replay queries success only at episode end; its recorder hook must advance earlier frames.
        if not is_replay or step == 3:
            for _ in range(3):
                assert success_cfg.func(env, **success_cfg.params).tolist() == [step == 3] * 2
        assert len(predicate_calls) == step, "Queries and the progress hook must not count extra frames"
        assert [state.all_complete for state in env.extras["progress_tracking"]["states"]] == [step == 3] * 2
        if not is_replay:
            assert recorded_success[-1] == [step == 3] * 2, "Progress must update before recorder terms"

    # Terminal episode recording must see completed progress before the environment clears it.
    env.episode_recorder_manager.record_pre_reset = lambda ids: terminal_success.append(
        env.progress_tracker.is_complete()[ids].tolist()
    )

    def reset_lab_environment(self, env_ids):
        self.episode_length_buf[env_ids] = 0
        self.termination_manager.reset(env_ids)

    with patch.object(ManagerBasedRLEnv, "_reset_idx", reset_lab_environment):
        env._reset_idx([0])
    assert terminal_success == [[True]]
    assert task_success_from_progress(env).tolist() == [False, True]
    assert len(predicate_calls) == 3, "A query after a partial reset must not advance surviving episodes"
    assert env.completed_episode_count == 1
    for step in range(1, 4):
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env.termination_manager.compute()
        recorder.record_post_step()
        assert task_success_from_progress(env).tolist() == [step == 3, True]
        assert len(predicate_calls) == 3 + step
    recorder.close()
    return True


@pytest.mark.parametrize("mode", ["native", "record", "replay", "replay_disabled_recorder"])
def test_external_success(mode):
    assert run_function_with_persistent_simulation_app(partial(_test_external_success, mode=mode))
