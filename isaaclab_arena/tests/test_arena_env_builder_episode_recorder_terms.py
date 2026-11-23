# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how ArenaEnvBuilder combines built-in, task, and environment episode recorder terms."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _build_episode_recorders_cfg(task_terms, environment_terms):
    """Build an environment whose task and environment add the given terms; return its episode recorder cfg."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    class _RecordingTask(NoTask):
        def get_episode_recorder_terms(self):
            return task_terms

    environment = IsaacLabArenaEnvironment(
        name="episode_recorder_terms", scene=Scene(), task=_RecordingTask(), episode_recorder_terms=environment_terms
    )
    builder = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cpu"))
    env_cfg, _ = builder.build_env_cfg()
    return env_cfg.episode_recorders


def _record(env, env_id):
    return {"value": 1}


def _test_episode_recorder_terms_are_combined_in_order(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    cfg = _build_episode_recorders_cfg(
        {"task_term": EpisodeRecorderTermCfg(func=_record)},
        {"environment_term": EpisodeRecorderTermCfg(func=_record)},
    )
    assert list(vars(cfg)) == ["placement", "variations", "progress", "task_term", "environment_term"]
    return True


def test_episode_recorder_terms_are_combined_in_order():
    assert run_function_with_persistent_simulation_app(_test_episode_recorder_terms_are_combined_in_order)


def _test_episode_recorder_term_names_must_be_unique(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    with pytest.raises(AssertionError, match="'motion' is already used by a built-in or task term"):
        _build_episode_recorders_cfg(
            {"motion": EpisodeRecorderTermCfg(func=_record)}, {"motion": EpisodeRecorderTermCfg(func=_record)}
        )
    with pytest.raises(AssertionError, match="'progress' is already used by a built-in or task term"):
        _build_episode_recorders_cfg({"progress": EpisodeRecorderTermCfg(func=_record)}, {})
    return True


def test_episode_recorder_term_names_must_be_unique():
    assert run_function_with_persistent_simulation_app(_test_episode_recorder_term_names_must_be_unique)
