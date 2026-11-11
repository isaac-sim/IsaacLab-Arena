# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for holding the last action in environments without an active episode."""

import torch
from dataclasses import dataclass
from types import SimpleNamespace

from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class _CountingPolicyCfg(PolicyCfg):
    pass


class _CountingPolicy(PolicyBase[_CountingPolicyCfg]):
    """Return the step count in every row and record the mask each step receives."""

    def __init__(self, config: _CountingPolicyCfg):
        super().__init__(config)
        self.active_env_masks = []

    def _get_action(self, env, observation, active_env_mask) -> torch.Tensor:
        self.active_env_masks.append(active_env_mask.tolist())
        return torch.full((env.unwrapped.num_envs, 2), float(len(self.active_env_masks)))


def _env(*active):
    return SimpleNamespace(unwrapped=SimpleNamespace(num_envs=len(active), active_episode_mask=torch.tensor(active)))


def test_inactive_environments_hold_their_last_action():
    policy = _CountingPolicy(_CountingPolicyCfg())

    # Env 1 never had an episode, so it still needs one action.
    torch.testing.assert_close(policy.get_action(_env(True, False), None), torch.full((2, 2), 1.0))
    torch.testing.assert_close(policy.get_action(_env(True, False), None), torch.tensor([[2.0, 2.0], [1.0, 1.0]]))
    torch.testing.assert_close(policy.get_action(_env(False, False), None), torch.tensor([[2.0, 2.0], [1.0, 1.0]]))

    assert policy.active_env_masks == [[True, True], [True, False], [False, False]]


def test_reset_discards_the_actions_to_hold():
    policy = _CountingPolicy(_CountingPolicyCfg())
    policy.get_action(_env(True, True), None)

    policy.reset(torch.tensor([1]))
    torch.testing.assert_close(policy.get_action(_env(False, False), None), torch.tensor([[1.0, 1.0], [2.0, 2.0]]))

    policy.reset()
    torch.testing.assert_close(policy.get_action(_env(False, False), None), torch.full((2, 2), 3.0))
    assert policy.active_env_masks[1:] == [[False, True], [True, True]]


def test_holding_can_be_turned_off():
    policy = _CountingPolicy(_CountingPolicyCfg(hold_action_while_inactive=False))
    policy.get_action(_env(True, True), None)

    torch.testing.assert_close(policy.get_action(_env(False, True), None), torch.full((2, 2), 2.0))
    assert policy.active_env_masks == [[True, True], [True, True]]


def test_environments_without_episode_tracking_are_always_active():
    policy = _CountingPolicy(_CountingPolicyCfg())
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2))
    policy.get_action(env, None)

    torch.testing.assert_close(policy.get_action(env, None), torch.full((2, 2), 2.0))
    assert policy.active_env_masks == [[True, True], [True, True]]
