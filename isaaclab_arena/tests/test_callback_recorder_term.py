# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for CallbackRecorderTerm.

Importing isaaclab_arena.recording.callback_recorder_term pulls in isaaclab.managers
via RecorderTerm, which needs a running SimulationApp (omni.timeline is only importable
once Kit has booted) -- see isaaclab_arena/tests/test_task_registry.py for the established
_test_/test_ + run_function_with_persistent_simulation_app pattern this mirrors.
"""

from __future__ import annotations

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _FakeEnv:
    """Minimal stand-in for a ManagerBasedEnv; CallbackRecorderTerm never inspects it."""


def _test_build_handlers_called_once_at_construction_with_env(simulation_app):
    from isaaclab_arena.recording.callback_recorder_term import (
        CallbackRecorderTerm,
        CallbackRecorderTermCfg,
        CallbackRecorderTermHandlers,
    )

    calls: list[object] = []

    def build_handlers(env):
        calls.append(env)
        return CallbackRecorderTermHandlers()

    env = _FakeEnv()
    CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=build_handlers), env)

    assert calls == [env]
    return True


def test_build_handlers_called_once_at_construction_with_env():
    result = run_function_with_persistent_simulation_app(_test_build_handlers_called_once_at_construction_with_env)
    assert result


def _test_record_post_step_forwards_to_on_post_step_and_returns_none_none(simulation_app):
    from isaaclab_arena.recording.callback_recorder_term import (
        CallbackRecorderTerm,
        CallbackRecorderTermCfg,
        CallbackRecorderTermHandlers,
    )

    calls: list[object] = []
    handlers = CallbackRecorderTermHandlers(on_post_step=calls.append)
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    result = term.record_post_step()

    assert calls == [env]
    assert result == (None, None)
    return True


def test_record_post_step_forwards_to_on_post_step_and_returns_none_none():
    result = run_function_with_persistent_simulation_app(
        _test_record_post_step_forwards_to_on_post_step_and_returns_none_none
    )
    assert result


def _test_record_pre_reset_forwards_env_and_env_ids(simulation_app):
    from isaaclab_arena.recording.callback_recorder_term import (
        CallbackRecorderTerm,
        CallbackRecorderTermCfg,
        CallbackRecorderTermHandlers,
    )

    calls: list[tuple[object, object]] = []
    handlers = CallbackRecorderTermHandlers(on_pre_reset=lambda env, env_ids: calls.append((env, env_ids)))
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    result = term.record_pre_reset([0, 2])

    assert calls == [(env, [0, 2])]
    assert result == (None, None)
    return True


def test_record_pre_reset_forwards_env_and_env_ids():
    result = run_function_with_persistent_simulation_app(_test_record_pre_reset_forwards_env_and_env_ids)
    assert result


def _test_close_forwards_to_on_close(simulation_app):
    from isaaclab_arena.recording.callback_recorder_term import (
        CallbackRecorderTerm,
        CallbackRecorderTermCfg,
        CallbackRecorderTermHandlers,
    )

    calls: list[str | None] = []
    handlers = CallbackRecorderTermHandlers(on_close=calls.append)
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    term.close("/tmp/dataset.hdf5")

    assert calls == ["/tmp/dataset.hdf5"]
    return True


def test_close_forwards_to_on_close():
    result = run_function_with_persistent_simulation_app(_test_close_forwards_to_on_close)
    assert result


def _test_none_handlers_are_no_ops(simulation_app):
    from isaaclab_arena.recording.callback_recorder_term import (
        CallbackRecorderTerm,
        CallbackRecorderTermCfg,
        CallbackRecorderTermHandlers,
    )

    handlers = CallbackRecorderTermHandlers()
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    assert term.record_post_step() == (None, None)
    assert term.record_pre_reset([0]) == (None, None)
    term.close(None)  # must not raise
    return True


def test_none_handlers_are_no_ops():
    result = run_function_with_persistent_simulation_app(_test_none_handlers_are_no_ops)
    assert result
