# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for CallbackRecorderTerm (no Isaac Sim required -- ManagerTermBase.__init__ only
stores cfg/env, so a plain object stands in for the env)."""

from __future__ import annotations

from isaaclab_arena.recording.callback_recorder_term import (
    CallbackRecorderTerm,
    CallbackRecorderTermCfg,
    CallbackRecorderTermHandlers,
)


class _FakeEnv:
    """Minimal stand-in for a ManagerBasedEnv; CallbackRecorderTerm never inspects it."""


def test_build_handlers_called_once_at_construction_with_env():
    calls: list[object] = []

    def build_handlers(env):
        calls.append(env)
        return CallbackRecorderTermHandlers()

    env = _FakeEnv()
    CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=build_handlers), env)

    assert calls == [env]


def test_record_post_step_forwards_to_on_post_step_and_returns_none_none():
    calls: list[object] = []
    handlers = CallbackRecorderTermHandlers(on_post_step=calls.append)
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    result = term.record_post_step()

    assert calls == [env]
    assert result == (None, None)


def test_record_pre_reset_forwards_env_and_env_ids():
    calls: list[tuple[object, object]] = []
    handlers = CallbackRecorderTermHandlers(on_pre_reset=lambda env, env_ids: calls.append((env, env_ids)))
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    result = term.record_pre_reset([0, 2])

    assert calls == [(env, [0, 2])]
    assert result == (None, None)


def test_close_forwards_to_on_close():
    calls: list[str | None] = []
    handlers = CallbackRecorderTermHandlers(on_close=calls.append)
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    term.close("/tmp/dataset.hdf5")

    assert calls == ["/tmp/dataset.hdf5"]


def test_none_handlers_are_no_ops():
    handlers = CallbackRecorderTermHandlers()
    env = _FakeEnv()
    term = CallbackRecorderTerm(CallbackRecorderTermCfg(build_handlers=lambda _env: handlers), env)

    assert term.record_post_step() == (None, None)
    assert term.record_pre_reset([0]) == (None, None)
    term.close(None)  # must not raise
