# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import io
import torch
from types import SimpleNamespace

from isaaclab_arena.evaluation import policy_runner
from isaaclab_arena.policy.policy_base import PolicyCfg

NUM_ENVS = 3
CONTROL_STEP_DT = 1.0 / 15.0


class _RolloutEnvironment:
    """Return scheduled completion events without simulating episode scheduling."""

    def __init__(self, step_results):
        self.cfg = SimpleNamespace(metrics=None)
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.step_dt = CONTROL_STEP_DT
        self.step_results = iter(step_results)
        self.configured_episode_limits = []
        self.reset_called = False
        self.steps_completed = 0
        self.completed_episode_count = 0
        self.reset_env_ids = torch.empty(0, dtype=torch.long)
        self.applied_actions = []

    def configure_episode_limit(self, num_episodes):
        assert not self.reset_called
        self.configured_episode_limits.append(num_episodes)

    def reset(self):
        self.reset_called = True
        return {"step": 0}, {}

    def get_language_instruction(self):
        return "Complete the task."

    def step(self, actions):
        self.applied_actions.append(actions.clone())
        completed_count, reset_env_ids, terminated_env_ids, truncated_env_ids = next(self.step_results)
        self.steps_completed += 1
        self.completed_episode_count = completed_count
        self.reset_env_ids = torch.tensor(reset_env_ids, dtype=torch.long)
        terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        truncated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        terminated[terminated_env_ids] = True
        truncated[truncated_env_ids] = True
        return {"step": self.steps_completed}, None, terminated, truncated, {}


class _Policy:
    handles_inference_latency = False

    def __init__(self, inference_latency_ms: float = 0.0):
        self.config = PolicyCfg(inference_latency_ms=inference_latency_ms)
        self.reset_calls = []
        self.observed_steps = []

    def reset(self, env_ids=None):
        self.reset_calls.append(None if env_ids is None else env_ids.tolist())

    def set_task_description(self, task_description):
        self.task_description = task_description

    def get_action(self, env, observation):
        self.observed_steps.append(observation["step"])
        # The action carries the step it was computed from, so delays are visible on the env side.
        return torch.full((NUM_ENVS, 1), float(observation["step"]))


def _wrap_environment(environment):
    return SimpleNamespace(unwrapped=environment, reset=environment.reset, step=environment.step)


def _capture_progress(monkeypatch, total):
    progress_bar = policy_runner.tqdm.tqdm(total=total, file=io.StringIO())
    monkeypatch.setattr(policy_runner.tqdm, "tqdm", lambda **kwargs: progress_bar)
    return progress_bar


def test_episode_rollout_waits_for_every_completion_and_resets_only_replacements(monkeypatch):
    environment = _RolloutEnvironment([
        (2, [0], [0, 1], []),
        (2, [], [], []),
        (3, [], [0], []),
        (4, [], [], [2]),
    ])
    policy = _Policy()
    progress_bar = _capture_progress(monkeypatch, total=4)

    metrics = policy_runner.rollout_policy(_wrap_environment(environment), policy, num_steps=None, num_episodes=4)

    assert metrics is None
    assert environment.configured_episode_limits == [4]
    assert environment.steps_completed == 4
    assert policy.reset_calls == [None, [0]]
    assert policy.observed_steps == [0, 1, 2, 3]
    assert policy.task_description == "Complete the task."
    assert progress_bar.n == 4


def test_step_rollout_keeps_its_step_limit_and_resets_actual_reset_environments(monkeypatch):
    environment = _RolloutEnvironment([
        (1, [1], [1], []),
        (1, [], [], []),
    ])
    policy = _Policy()
    progress_bar = _capture_progress(monkeypatch, total=2)

    policy_runner.rollout_policy(_wrap_environment(environment), policy, num_steps=2, num_episodes=None)

    assert environment.configured_episode_limits == []
    assert environment.steps_completed == 2
    assert policy.reset_calls == [None, [1]]
    assert policy.observed_steps == [0, 1]
    assert progress_bar.n == 2


def test_per_step_policy_actions_are_delayed_by_the_configured_latency(monkeypatch):
    """Apply each action two control steps after its observation and hold until then, per env."""
    environment = _RolloutEnvironment([
        (0, [], [], []),
        (0, [], [], []),
        (0, [], [], []),
        (1, [1], [1], []),  # env 1 restarts and holds again
        (1, [], [], []),
        (1, [], [], []),
    ])
    hold_action = torch.full((NUM_ENVS, 1), -1.0)
    monkeypatch.setattr(policy_runner, "hold_action_from_action_manager", lambda env: hold_action)
    policy = _Policy(inference_latency_ms=100.0)  # 1.5 control steps at 15 Hz, rounded up to 2
    _capture_progress(monkeypatch, total=6)

    policy_runner.rollout_policy(_wrap_environment(environment), policy, num_steps=6, num_episodes=None)

    applied = torch.cat(environment.applied_actions, dim=1)  # (num_envs, steps)
    # Env 0: holds for two steps, then applies the action computed two steps earlier.
    assert applied[0].tolist() == [-1.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    # Env 1: reset after the step-3 action, so it holds for two steps and replays from step 4's observation.
    assert applied[1].tolist() == [-1.0, -1.0, 0.0, 1.0, -1.0, -1.0]
    assert policy.reset_calls == [None, [1]]


def test_zero_latency_leaves_per_step_policy_actions_unchanged(monkeypatch):
    """Keep today's behavior: with no latency the action from this step's observation is applied this step."""
    environment = _RolloutEnvironment([(0, [], [], []), (0, [], [], [])])
    monkeypatch.setattr(policy_runner, "hold_action_from_action_manager", lambda env: None)
    _capture_progress(monkeypatch, total=2)

    policy_runner.rollout_policy(_wrap_environment(environment), _Policy(), num_steps=2, num_episodes=None)

    applied = torch.cat(environment.applied_actions, dim=1)
    assert applied[0].tolist() == [0.0, 1.0]
