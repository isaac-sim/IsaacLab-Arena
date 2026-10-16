# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how the builder combines built-in, task, and environment episode recorder terms."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _builder():
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    environment = IsaacLabArenaEnvironment(name="episode_recorder_terms", scene=Scene())
    return ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cpu"))


def _record(env, env_id):
    return {"value": 1}


def _test_episode_recorder_terms_are_combined_in_order(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    cfg = _builder()._build_episode_recorders_cfg(
        {"task_term": EpisodeRecorderTermCfg(func=_record)},
        {"environment_term": EpisodeRecorderTermCfg(func=_record)},
    )
    assert list(vars(cfg)) == ["placement", "variations", "progress", "task_term", "environment_term"]
    return True


def test_episode_recorder_terms_are_combined_in_order():
    assert run_function_with_persistent_simulation_app(_test_episode_recorder_terms_are_combined_in_order)


def _test_episode_recorder_term_names_must_be_unique(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    builder = _builder()
    with pytest.raises(AssertionError, match="'motion' is used more than once"):
        builder._build_episode_recorders_cfg(
            {"motion": EpisodeRecorderTermCfg(func=_record)}, {"motion": EpisodeRecorderTermCfg(func=_record)}
        )
    with pytest.raises(AssertionError, match="'progress' is used more than once"):
        builder._build_episode_recorders_cfg({"progress": EpisodeRecorderTermCfg(func=_record)}, {})
    return True


def test_episode_recorder_term_names_must_be_unique():
    assert run_function_with_persistent_simulation_app(_test_episode_recorder_term_names_must_be_unique)
