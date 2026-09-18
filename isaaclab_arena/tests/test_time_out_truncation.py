# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that ArenaEnvBuilder installs episode timeouts as truncations."""

from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_builder_timeout_is_truncation(simulation_app, timeout_s):
    import math
    import torch

    from isaaclab.envs.mdp import time_out
    from isaaclab.managers import TerminationManager

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class TimeoutOnlyTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=timeout_s)

    description = IsaacLabArenaEnvironment(name="timeout_truncation", scene=Scene(), task=TimeoutOnlyTask())
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=3, solve_relations=False, device="cpu"))
    env_cfg, _ = builder.compose_manager_cfg()
    if timeout_s is None:
        assert "time_out" not in env_cfg.terminations.to_dict()
    else:
        assert env_cfg.episode_length_s == timeout_s
        assert env_cfg.terminations.time_out.func is time_out
        assert env_cfg.terminations.time_out.time_out is True

    max_episode_length = math.ceil(env_cfg.episode_length_s / (env_cfg.decimation * env_cfg.sim.dt))
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={},
        episode_length_buf=torch.tensor([max_episode_length - 1, max_episode_length, max_episode_length + 1]),
        max_episode_length=max_episode_length,
    )
    manager = TerminationManager(env_cfg.terminations, env)
    expected_timeouts = torch.tensor([False, timeout_s is not None, timeout_s is not None])
    torch.testing.assert_close(manager.compute(), expected_timeouts)
    torch.testing.assert_close(manager.time_outs, expected_timeouts)
    assert not manager.terminated.any()
    return True


@pytest.mark.parametrize("timeout_s", [None, 1.0, 12.0])
def test_builder_timeout_is_truncation(timeout_s):
    assert run_function_with_persistent_simulation_app(_test_builder_timeout_is_truncation, timeout_s=timeout_s)
