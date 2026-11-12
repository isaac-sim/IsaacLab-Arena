# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the environments whose actions a policy computes."""

import torch
from dataclasses import dataclass
from types import SimpleNamespace

from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class _RecordingPolicyCfg(PolicyCfg):
    pass


class _RecordingPolicy(PolicyBase[_RecordingPolicyCfg]):
    """Return zero actions and record the mask each step receives."""

    def __init__(self, config: _RecordingPolicyCfg):
        super().__init__(config)
        self.active_env_masks = []

    def _get_action(self, env, observation, active_env_mask) -> torch.Tensor:
        self.active_env_masks.append(active_env_mask.tolist())
        return torch.zeros((env.unwrapped.num_envs, 2))


def test_finished_environments_are_inactive():
    policy = _RecordingPolicy(_RecordingPolicyCfg())
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(num_envs=3, finished_episode_mask=torch.tensor([False, True, False]))
    )

    policy.get_action(env, None)

    assert policy.active_env_masks == [[True, False, True]]


def test_environments_without_episode_tracking_are_all_active():
    policy = _RecordingPolicy(_RecordingPolicyCfg())

    policy.get_action(SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2)), None)

    assert policy.active_env_masks == [[True, True]]
