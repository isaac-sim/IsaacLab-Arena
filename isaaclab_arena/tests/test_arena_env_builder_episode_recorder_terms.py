# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how ArenaEnvBuilder combines built-in and environment episode recorder terms."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _build_episode_recorders_cfg(environment_terms):
    """Build an environment that adds the given terms; return its episode recorder cfg."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    environment = IsaacLabArenaEnvironment(
        name="episode_recorder_terms", scene=Scene(), episode_recorder_terms=environment_terms
    )
    builder = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cpu"))
    env_cfg, _ = builder.build_env_cfg()
    return env_cfg.episode_recorders


def _record(env, env_id):
    return {"value": 1}


def _test_environment_terms_follow_the_built_in_terms(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    cfg = _build_episode_recorders_cfg({"environment_term": EpisodeRecorderTermCfg(func=_record)})
    assert list(vars(cfg)) == ["placement", "variations", "progress", "environment_term"]
    return True


def test_environment_terms_follow_the_built_in_terms():
    assert run_function_with_persistent_simulation_app(_test_environment_terms_follow_the_built_in_terms)


def _test_environment_term_must_not_use_a_built_in_name(simulation_app):
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    with pytest.raises(AssertionError, match="'progress' is used by a built-in term"):
        _build_episode_recorders_cfg({"progress": EpisodeRecorderTermCfg(func=_record)})
    return True


def test_environment_term_must_not_use_a_built_in_name():
    assert run_function_with_persistent_simulation_app(_test_environment_term_must_not_use_a_built_in_name)
