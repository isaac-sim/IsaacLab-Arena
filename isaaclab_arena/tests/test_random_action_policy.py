# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch
from types import SimpleNamespace

from isaaclab_arena.policy.random_action_policy import RandomActionPolicy, RandomActionPolicyCfg


def _fake_env(shape: tuple[int, ...], low: float = -np.inf, high: float = np.inf):
    action_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    return SimpleNamespace(action_space=action_space, unwrapped=SimpleNamespace(device="cpu"))


def _gr1_observation(num_envs: int = 2) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "policy": {
            "left_eef_pos": torch.zeros(num_envs, 3),
            "left_eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1),
            "right_eef_pos": torch.ones(num_envs, 3),
            "right_eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1),
            "hand_joint_state": torch.zeros(num_envs, 22),
        }
    }


def test_random_action_preserves_gr1_quaternions_and_perturbs_pose():
    """Keep GR1 Pink rotations valid while randomizing positions and hands."""
    config = RandomActionPolicyCfg(random_seed=7, eef_position_noise=0.02, hand_joint_noise=0.1)
    actions = RandomActionPolicy(config).get_action(_fake_env((2, 36)), _gr1_observation())

    assert actions.shape == (2, 36)
    torch.testing.assert_close(actions[:, 3:7], _gr1_observation()["policy"]["left_eef_quat"])
    torch.testing.assert_close(actions[:, 10:14], _gr1_observation()["policy"]["right_eef_quat"])
    assert torch.all(actions[:, 0:3].abs() <= 0.02)
    assert torch.all((actions[:, 7:10] - 1.0).abs() <= 0.02)
    assert torch.all(actions[:, 14:].abs() <= 0.1)


def test_random_action_is_reproducible_and_respects_box_bounds():
    """Use the configured seed and clamp actions to finite environment bounds."""
    config = RandomActionPolicyCfg(random_seed=3, action_scale=0.5)
    env = _fake_env((2, 4), low=-0.1, high=0.1)

    first = RandomActionPolicy(config).get_action(env, {})
    second = RandomActionPolicy(config).get_action(env, {})

    torch.testing.assert_close(first, second)
    assert torch.all(first >= -0.1)
    assert torch.all(first <= 0.1)
