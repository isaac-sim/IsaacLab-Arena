# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

import pytest
import torch

from isaaclab_arena.evaluation.policy_runner import _cleanup_policy_and_env, rollout_policy
from isaaclab_arena.remote_policy.policy_client import TransportTimeoutError


class _FakeEnv:
    def __init__(self, *, step_exc: Exception | None = None) -> None:
        self.step_exc = step_exc
        self.closed = False
        self.cfg = types.SimpleNamespace(
            isaaclab_arena_env=types.SimpleNamespace(
                task=types.SimpleNamespace(get_task_description=lambda: "dummy task")
            )
        )

    def reset(self):
        return {}, {}

    def step(self, actions):
        if self.step_exc is not None:
            raise self.step_exc
        return {}, 0.0, torch.tensor([False]), torch.tensor([False]), {}

    def close(self) -> None:
        self.closed = True


class _TimeoutOnResetPolicy:
    is_remote = False

    def reset(self, env_ids=None):
        raise TransportTimeoutError(
            "ucx recv timeout",
            must_reconnect=True,
            source="ucx_recv",
        )

    def set_task_description(self, task_description):
        return task_description

    def get_action(self, env, obs):
        return torch.zeros((1, 1), dtype=torch.float32)


class _ValueErrorOnResetPolicy:
    is_remote = False

    def reset(self, env_ids=None):
        raise ValueError("policy reset failed")

    def set_task_description(self, task_description):
        return task_description

    def get_action(self, env, obs):
        return torch.zeros((1, 1), dtype=torch.float32)


class _StepFailPolicy:
    is_remote = False

    def reset(self, env_ids=None):
        return None

    def set_task_description(self, task_description):
        return task_description

    def get_action(self, env, obs):
        return torch.zeros((1, 1), dtype=torch.float32)


class _RemotePolicyStub:
    is_remote = True

    def __init__(self, *, raise_on_shutdown: bool = False) -> None:
        self.raise_on_shutdown = raise_on_shutdown
        self.shutdown_calls: list[bool] = []

    def shutdown_remote(self, kill_server: bool = False) -> None:
        self.shutdown_calls.append(kill_server)
        if self.raise_on_shutdown:
            raise RuntimeError("shutdown failed")


class _FakePbar:
    def __init__(self) -> None:
        self.closed = False

    def update(self, n: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_rollout_policy_preserves_transport_timeout_before_pbar_creation() -> None:
    env = _FakeEnv()
    policy = _TimeoutOnResetPolicy()

    with pytest.raises(TransportTimeoutError) as exc_info:
        rollout_policy(env, policy, num_steps=1, num_episodes=None)

    assert exc_info.value.must_reconnect is True
    assert exc_info.value.source == "ucx_recv"


def test_rollout_policy_wraps_generic_exception_without_unboundlocalerror() -> None:
    env = _FakeEnv()
    policy = _ValueErrorOnResetPolicy()

    with pytest.raises(RuntimeError, match="policy reset failed") as exc_info:
        rollout_policy(env, policy, num_steps=1, num_episodes=None)

    assert "UnboundLocalError" not in str(exc_info.value)


def test_rollout_policy_closes_progress_bar_on_runtime_error(monkeypatch) -> None:
    env = _FakeEnv(step_exc=RuntimeError("step failed"))
    policy = _StepFailPolicy()
    pbar = _FakePbar()

    monkeypatch.setattr("isaaclab_arena.evaluation.policy_runner.tqdm.tqdm", lambda *args, **kwargs: pbar)

    with pytest.raises(RuntimeError, match="step failed"):
        rollout_policy(env, policy, num_steps=1, num_episodes=None)

    assert pbar.closed is True


def test_cleanup_policy_and_env_runs_both_paths() -> None:
    env = _FakeEnv()
    policy = _RemotePolicyStub()

    _cleanup_policy_and_env(policy, env, kill_server=True)

    assert policy.shutdown_calls == [True]
    assert env.closed is True


def test_cleanup_policy_and_env_still_closes_env_when_shutdown_fails() -> None:
    env = _FakeEnv()
    policy = _RemotePolicyStub(raise_on_shutdown=True)

    _cleanup_policy_and_env(policy, env, kill_server=False)

    assert policy.shutdown_calls == [False]
    assert env.closed is True
