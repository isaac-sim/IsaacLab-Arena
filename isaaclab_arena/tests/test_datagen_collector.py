# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for DatagenCollectorBase and its CallbackRecorderTerm adapter.

Importing isaaclab_arena.evaluation.datagen_collector needs a running SimulationApp
(it imports CallbackRecorderTermHandlers, which imports isaaclab.managers) -- see
isaaclab_arena/tests/test_task_registry.py for the established _test_/test_ +
run_function_with_persistent_simulation_app pattern this mirrors.
"""

from __future__ import annotations

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _build_fake_collector():
    """Build a fresh fake DatagenCollectorBase, deferring the import until Kit is running."""
    from isaaclab_arena.evaluation.datagen_collector import DatagenCollectorBase

    class _FakeCollector(DatagenCollectorBase):
        def __init__(self) -> None:
            self.steps: list[object] = []
            self.episode_ends: list[tuple[object, int, str]] = []
            self.finalized: list[object] = []
            self.closed: list[object] = []

        def on_step(self, env):
            self.steps.append(env)

        def on_episode_end(self, env, env_id, outcome="timeout"):
            self.episode_ends.append((env, env_id, outcome))

        def finalize(self, env=None):
            self.finalized.append(env)

        def close(self, env=None):
            self.closed.append(env)

    return _FakeCollector()


class _FakeTerminationManager:
    active_terms: list[str] = []

    def get_term(self, name):
        raise AssertionError("no termination terms configured for this fake env")


class _FakeEnv:
    def __init__(self) -> None:
        self.termination_manager = _FakeTerminationManager()


def _test_on_post_step_forwards_to_collector_on_step(simulation_app):
    from isaaclab_arena.evaluation.datagen_collector import build_datagen_callback_handlers

    collector = _build_fake_collector()
    handlers = build_datagen_callback_handlers(collector)
    env = _FakeEnv()

    handlers.on_post_step(env)

    assert collector.steps == [env]
    return True


def test_on_post_step_forwards_to_collector_on_step():
    assert run_function_with_persistent_simulation_app(_test_on_post_step_forwards_to_collector_on_step)


def _test_on_pre_reset_calls_on_episode_end_per_env_id_with_classified_outcome(simulation_app):
    from isaaclab_arena.evaluation.datagen_collector import build_datagen_callback_handlers

    collector = _build_fake_collector()
    handlers = build_datagen_callback_handlers(collector)
    env = _FakeEnv()  # no active termination terms -> classify_outcome returns "failure"

    handlers.on_pre_reset(env, [0, 3])

    assert collector.episode_ends == [(env, 0, "failure"), (env, 3, "failure")]
    return True


def test_on_pre_reset_calls_on_episode_end_per_env_id_with_classified_outcome():
    assert run_function_with_persistent_simulation_app(
        _test_on_pre_reset_calls_on_episode_end_per_env_id_with_classified_outcome
    )


def _test_on_close_forwards_env_via_closure_not_file_path(simulation_app):
    """on_close's build_handlers closure captures env at call time and ignores file_path."""
    from isaaclab_arena.evaluation.datagen_collector import build_datagen_callback_handlers

    collector = _build_fake_collector()
    env = _FakeEnv()

    def build_handlers(built_env):
        return build_datagen_callback_handlers(collector, env=built_env)

    handlers = build_handlers(env)
    handlers.on_close("/tmp/whatever.hdf5")

    assert collector.closed == [env]
    return True


def test_on_close_forwards_env_via_closure_not_file_path():
    assert run_function_with_persistent_simulation_app(_test_on_close_forwards_env_via_closure_not_file_path)
