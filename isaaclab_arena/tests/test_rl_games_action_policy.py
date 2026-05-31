# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch


class _DummyUnwrappedEnv:
    def __init__(self):
        self.obs_buf = {"policy": torch.ones(1, 3)}


class _DummyEnv:
    def __init__(self):
        self.unwrapped = _DummyUnwrappedEnv()
        self.num_resets = 0

    def reset(self, *args, **kwargs):
        self.num_resets += 1
        return {"policy": torch.zeros(1, 3)}, {"reset": True}


def test_rl_games_wrapper_init_reuses_current_observation_buffer(monkeypatch):
    from isaaclab_arena.policy import rl_games_action_policy

    env = _DummyEnv()
    original_reset = env.reset
    observed = {}

    def _mock_base_init(self, env, *args, **kwargs):
        obs, info = env.reset()
        observed["obs"] = obs
        observed["info"] = info
        self.env = env

    monkeypatch.setattr(rl_games_action_policy.RlGamesVecEnvWrapper, "__init__", _mock_base_init)
    rl_games_action_policy._RlGamesInferenceEnvWrapper(env, "cuda:0", torch.inf, torch.inf)

    assert env.reset == original_reset
    assert env.num_resets == 0
    assert observed["info"] == {}
    assert torch.equal(observed["obs"]["policy"], env.unwrapped.obs_buf["policy"])
