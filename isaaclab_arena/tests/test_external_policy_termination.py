# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnv,
    external_policy_termination,
)
from isaaclab_arena.evaluation.policy_runner import _forward_policy_episode_terminations


class _StubArenaEnv:
    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._external_policy_termination_buf = torch.zeros(num_envs, dtype=torch.bool)

    @property
    def external_policy_termination_buf(self) -> torch.Tensor:
        return self._external_policy_termination_buf

    def request_external_policy_termination(self, termination_mask: torch.Tensor) -> None:
        IsaacLabArenaManagerBasedRLEnv.request_external_policy_termination(self, termination_mask)


class _StubGymEnv:
    def __init__(self, num_envs: int = 2):
        self.unwrapped = _StubArenaEnv(num_envs)


class _StubPolicy:
    def __init__(self, termination_mask):
        self._termination_mask = termination_mask

    def get_episode_termination_mask(self):
        return self._termination_mask


def test_policy_termination_mask_is_forwarded_and_latched():
    env = _StubGymEnv()

    _forward_policy_episode_terminations(env, _StubPolicy(torch.tensor([True, False])))
    _forward_policy_episode_terminations(env, _StubPolicy(torch.tensor([False, True])))

    assert external_policy_termination(env.unwrapped).tolist() == [True, True]


def test_policy_without_termination_mask_is_noop():
    env = _StubGymEnv()

    _forward_policy_episode_terminations(env, _StubPolicy(None))

    assert not env.unwrapped.external_policy_termination_buf.any()


@pytest.mark.parametrize(
    "termination_mask, message",
    [
        ([True, False], "torch.Tensor"),
        (torch.tensor([1, 0]), "torch.bool"),
        (torch.tensor([[True, False]]), "shape"),
        (torch.tensor([True]), "shape"),
    ],
)
def test_invalid_policy_termination_mask_is_rejected(termination_mask, message):
    env = _StubGymEnv()

    with pytest.raises(AssertionError, match=message):
        _forward_policy_episode_terminations(env, _StubPolicy(termination_mask))


def test_environment_rejects_invalid_termination_mask_shape():
    env = _StubArenaEnv()

    with pytest.raises(AssertionError, match="shape"):
        env.request_external_policy_termination(torch.tensor([True]))


def test_reset_records_policy_termination_before_clearing_completed_envs(monkeypatch):
    events = []
    env = object.__new__(IsaacLabArenaManagerBasedRLEnv)
    env._is_closed = True
    env._first_reset = False
    env._episode_counts = {}
    env._external_policy_termination_buf = torch.tensor([True, True])

    class _Recorder:
        def record_pre_reset(self, env_ids):
            events.append(("record", env._external_policy_termination_buf.clone()))

    env.episode_recorder_manager = _Recorder()
    monkeypatch.setattr(
        IsaacLabArenaManagerBasedRLEnv.__mro__[1],
        "_reset_idx",
        lambda self, env_ids: events.append(("reset", self._external_policy_termination_buf.clone())),
    )
    monkeypatch.setattr(IsaacLabArenaManagerBasedRLEnv, "_snapshot_initial_object_poses", lambda self, env_ids: None)

    IsaacLabArenaManagerBasedRLEnv._reset_idx(env, torch.tensor([0]))

    assert events[0][0] == "record"
    assert events[0][1].tolist() == [True, True]
    assert events[1][0] == "reset"
    assert events[1][1].tolist() == [False, True]
