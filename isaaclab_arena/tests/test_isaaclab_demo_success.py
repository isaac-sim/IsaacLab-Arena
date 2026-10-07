# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lab demo tools retain Arena's temporal success and partial-reset lifecycle."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_external_success(simulation_app, mode):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.task_success import (
        ExternalTaskSuccessRecorderCfg,
        TaskSuccessTerm,
        external_task_success,
    )
    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate_calls = []

    def predicate(env):
        predicate_calls.append(env.episode_length_buf.clone())
        return torch.ones(env.num_envs, dtype=torch.bool)

    env = IsaacLabArenaManagerBasedRLEnv.__new__(IsaacLabArenaManagerBasedRLEnv)
    env.cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"))
    env.sim = SimpleNamespace(is_playing=lambda: True, device="cpu")
    env.scene = SimpleNamespace(num_envs=2)
    env.extras = {}
    env._progress_tracker = None
    env._external_success_step = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 0
    env._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(num_envs=2, device="cpu")
    env._active_episode_mask = torch.ones(2, dtype=torch.bool)
    env._reset_env_ids = torch.empty(0, dtype=torch.long)
    env._episode_indices = {0: 0, 1: 0}
    env._episode_limit = None
    env._started_episode_count = 2
    env._completed_episode_count = 0
    env.episode_recorder_manager = SimpleNamespace(record_pre_reset=lambda _ids: None)

    success_cfg = TerminationTermCfg(
        func=TaskSuccessTerm,
        params={
            "success_criteria": [
                CompletionCriteria(
                    name="stable",
                    predicate_sequence=[TrueForConsecutiveStepsCfg(predicate=predicate, required_steps=3)],
                )
            ]
        },
    )
    env._external_success_manager = TerminationManager({"success": success_cfg}, env)
    # Recording replaces success; replay removes the entire termination config.
    if mode == "native":
        termination_cfg = {"success": success_cfg.replace(func=external_task_success, params={})}
    elif mode == "record":
        termination_cfg = {"success": TerminationTermCfg(func=lambda env: torch.zeros(2, dtype=torch.bool))}
    else:
        termination_cfg = {}
    env.termination_manager = TerminationManager(termination_cfg, env)
    recorder_cfg = ExternalTaskSuccessRecorderCfg()
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    assert env.progress_tracker is not None
    assert not external_task_success(env).any(), "A query before stepping must not advance progress"
    assert not predicate_calls

    for step in range(1, 4):
        env.episode_length_buf += 1
        env.common_step_counter += 1
        terminated = env.termination_manager.compute()
        assert terminated.tolist() == ([step == 3] * 2 if mode == "native" else [False, False])
        assert recorder.record_post_step() == (None, None)
        for _ in range(3):
            assert external_task_success(env).tolist() == [step == 3] * 2
        assert len(predicate_calls) == step, "Queries and the progress hook must not count extra frames"
        assert [state.all_complete for state in env.extras["progress_tracking"]["states"]] == [step == 3] * 2

    # Exercise Arena's actual reset override. Lab's reset resets only episode counters here.
    def reset_lab_environment(self, env_ids):
        self.episode_length_buf[env_ids] = 0

    with patch.object(ManagerBasedRLEnv, "_reset_idx", reset_lab_environment):
        env._reset_idx([0])
    assert external_task_success(env).tolist() == [False, True]
    assert len(predicate_calls) == 3, "A query after a partial reset must not advance surviving episodes"
    assert env.completed_episode_count == 1
    for step in range(1, 4):
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env.termination_manager.compute()
        recorder.record_post_step()
        assert external_task_success(env).tolist() == [step == 3, True]
        assert len(predicate_calls) == 3 + step
    return True


@pytest.mark.parametrize("mode", ["native", "record", "replay"])
def test_external_success(mode):
    from functools import partial

    assert run_function_with_persistent_simulation_app(partial(_test_external_success, mode=mode))
