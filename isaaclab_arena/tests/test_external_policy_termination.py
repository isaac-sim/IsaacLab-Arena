# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest


def _test_external_policy_termination_contract(_simulation_app):
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
        IsaacLabArenaManagerBasedRLEnv,
        external_policy_termination,
    )
    from isaaclab_arena.evaluation.policy_runner import _forward_policy_episode_terminations

    class StubArenaEnv:
        def __init__(self, num_envs: int = 2):
            self.num_envs = num_envs
            self.device = torch.device("cpu")
            self._external_policy_termination_buf = torch.zeros(num_envs, dtype=torch.bool)

        @property
        def external_policy_termination_buf(self) -> torch.Tensor:
            return self._external_policy_termination_buf

        def request_external_policy_termination(self, termination_mask: torch.Tensor) -> None:
            IsaacLabArenaManagerBasedRLEnv.request_external_policy_termination(self, termination_mask)

    class StubGymEnv:
        def __init__(self, num_envs: int = 2):
            self.unwrapped = StubArenaEnv(num_envs)

    class StubPolicy:
        def __init__(self, termination_mask):
            self._termination_mask = termination_mask

        def get_episode_termination_mask(self):
            return self._termination_mask

    env = StubGymEnv()
    _forward_policy_episode_terminations(env, StubPolicy(torch.tensor([True, False])))
    _forward_policy_episode_terminations(env, StubPolicy(torch.tensor([False, True])))
    assert external_policy_termination(env.unwrapped).tolist() == [True, True]

    noop_env = StubGymEnv()
    _forward_policy_episode_terminations(noop_env, StubPolicy(None))
    assert not noop_env.unwrapped.external_policy_termination_buf.any()

    invalid_masks = [
        ([True, False], "torch.Tensor"),
        (torch.tensor([1, 0]), "torch.bool"),
        (torch.tensor([[True, False]]), "shape"),
        (torch.tensor([True]), "shape"),
    ]
    for termination_mask, message in invalid_masks:
        with pytest.raises(AssertionError, match=message):
            _forward_policy_episode_terminations(env, StubPolicy(termination_mask))

    unsupported_env = type(
        "UnsupportedGymEnv",
        (),
        {"unwrapped": type("UnsupportedEnv", (), {"num_envs": 2})()},
    )()
    with pytest.raises(RuntimeError, match="does not support"):
        _forward_policy_episode_terminations(unsupported_env, StubPolicy(torch.tensor([True, False])))

    with pytest.raises(AssertionError, match="shape"):
        env.unwrapped.request_external_policy_termination(torch.tensor([True]))

    events = []
    reset_env = object.__new__(IsaacLabArenaManagerBasedRLEnv)
    reset_env._is_closed = True
    reset_env._first_reset = False
    reset_env._episode_counts = {}
    reset_env._external_policy_termination_buf = torch.tensor([True, True])

    class Recorder:
        def record_pre_reset(self, env_ids):
            events.append(("record", reset_env._external_policy_termination_buf.clone()))

    reset_env.episode_recorder_manager = Recorder()
    original_reset = IsaacLabArenaManagerBasedRLEnv.__mro__[1]._reset_idx
    try:
        IsaacLabArenaManagerBasedRLEnv.__mro__[1]._reset_idx = lambda self, env_ids: events.append(
            ("reset", self._external_policy_termination_buf.clone())
        )
        IsaacLabArenaManagerBasedRLEnv._reset_idx(reset_env, torch.tensor([0]))
    finally:
        IsaacLabArenaManagerBasedRLEnv.__mro__[1]._reset_idx = original_reset

    assert events[0][0] == "record"
    assert events[0][1].tolist() == [True, True]
    assert events[1][0] == "reset"
    assert events[1][1].tolist() == [False, True]
    return True


def test_external_policy_termination_contract():
    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_external_policy_termination_contract, headless=True)
