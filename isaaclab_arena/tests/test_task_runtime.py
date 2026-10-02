# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime hooks preserve terminal evidence and subset reset ordering."""

from types import SimpleNamespace
from unittest.mock import patch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_runtime_lifecycle(_simulation_app):
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.tasks.task_runtime import TaskRuntime, TaskRuntimeCfg
    from isaaclab_arena.tests.test_task_success_from_progress import _make_environment_and_manager

    calls = []

    class Runtime(TaskRuntime):
        def __init__(self, env, initial):
            self.env = env
            self.values = list(initial)

        def prepare_reset(self, env_ids):
            calls.append(("prepare", tuple(env_ids)))

        def reset(self, env_ids):
            calls.append(("finalize", tuple(env_ids)))
            for env_id in env_ids:
                self.values[env_id] = 0

        def update(self):
            self.env.predicate_results["ready"][:] = True

    env = object.__new__(IsaacLabArenaManagerBasedRLEnv)
    env._first_reset = True
    env._episode_counts = {}
    env._task_runtime = TaskRuntimeCfg(Runtime, {"initial": [7, 9]}).build(env)
    env.episode_recorder_manager = SimpleNamespace(
        record_pre_reset=lambda ids: calls.append(("record", tuple(env.task_runtime.values)))
    )

    def reset_events(base, ids):
        calls.append(("events", tuple(base.get_episode_index(index) for index in ids)))
        assert calls[-2][0] == "prepare", "Constraints must be released before reset/variation events."

    with patch.object(ManagerBasedRLEnv, "_reset_idx", reset_events):
        env._reset_idx([0, 1])
        assert calls == [("prepare", (0, 1)), ("events", (0, 0)), ("finalize", (0, 1))]
        env.task_runtime.values[:] = [3, 5]
        calls.clear()
        env._reset_idx([1])
        assert calls == [("record", (3, 5)), ("prepare", (1,)), ("events", (1,)), ("finalize", (1,))]
        assert env.task_runtime.values == [3, 0]
        assert [env.get_episode_index(index) for index in (0, 1)] == [0, 1]

    progress_env, manager, _recorder = _make_environment_and_manager(["ready"])
    progress_env.task_runtime = Runtime(progress_env, [0, 0])
    progress_env.predicate_results["ready"][:] = False
    manager.compute()
    assert torch.all(manager.get_term("success")), "Runtime measurements must precede success predicates."
    return True


def test_runtime_lifecycle():
    assert run_function_with_persistent_simulation_app(_test_runtime_lifecycle)
