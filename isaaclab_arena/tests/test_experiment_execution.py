# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import experiment_execution
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, ExperimentStatus, RolloutCfg
from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    pass


@dataclass
class _PolicyCfg(PolicyCfg):
    pass


class _Policy:
    def has_length(self):
        return False


class _EpisodeRecorder:
    def set_job_name(self, name):
        self.name = name

    def set_output_path(self, path):
        self.path = path


def _environment():
    return SimpleNamespace(unwrapped=SimpleNamespace(episode_recorder=_EpisodeRecorder()))


def _experiment(**overrides):
    values = {
        "name": "test_experiment",
        "environment": _EnvironmentCfg(),
        "policy": _PolicyCfg(),
        "rollout": RolloutCfg(num_episodes=5),
        "num_rebuilds": 2,
    }
    values.update(overrides)
    return ArenaExperimentCfg(**values)


def test_build_and_run_experiment_splits_episode_budget_without_mutating_config(monkeypatch, tmp_path):
    experiment = _experiment()
    rollout_limits = []
    received_experiment_cfgs = []

    def make_environment(cfg, render_mode):
        received_experiment_cfgs.append(cfg)
        return _environment()

    monkeypatch.setattr(experiment_execution, "_build_environment_from_cfg", make_environment)
    monkeypatch.setattr(experiment_execution, "_build_policy_from_cfg", lambda cfg: _Policy())
    monkeypatch.setattr(experiment_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(experiment_execution, "close_experiment_resources", lambda policy, env: None)

    def record_rollout(env, policy, num_steps, num_episodes):
        rollout_limits.append((num_steps, num_episodes))

    monkeypatch.setattr(experiment_execution, "rollout_policy", record_rollout)

    result = experiment_execution.build_and_run_experiment(
        experiment,
        output_dir=tmp_path,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert rollout_limits == [(None, 3), (None, 2)]
    assert received_experiment_cfgs == [experiment, experiment]
    assert experiment.rollout == RolloutCfg(num_episodes=5)


def test_build_and_run_experiment_raises_and_closes_resources(monkeypatch, tmp_path):
    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        experiment_execution,
        "_build_environment_from_cfg",
        lambda cfg, render_mode: environment,
    )
    monkeypatch.setattr(experiment_execution, "_build_policy_from_cfg", lambda cfg: policy)
    monkeypatch.setattr(experiment_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(
        experiment_execution,
        "close_experiment_resources",
        lambda closed_policy, closed_environment: closed_resources.append((closed_policy, closed_environment)),
    )
    monkeypatch.setattr(
        experiment_execution,
        "rollout_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )

    with pytest.raises(RuntimeError, match="rollout failed"):
        experiment_execution.build_and_run_experiment(
            _experiment(rollout=RolloutCfg(num_steps=2), num_rebuilds=1),
            output_dir=tmp_path,
        )

    assert closed_resources == [(policy, environment)]


def test_build_and_run_experiment_requires_a_limit_for_an_unbounded_policy(monkeypatch, tmp_path):
    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        experiment_execution,
        "_build_environment_from_cfg",
        lambda cfg, render_mode: environment,
    )
    monkeypatch.setattr(experiment_execution, "_build_policy_from_cfg", lambda cfg: policy)
    monkeypatch.setattr(
        experiment_execution,
        "close_experiment_resources",
        lambda closed_policy, closed_environment: closed_resources.append((closed_policy, closed_environment)),
    )

    with pytest.raises(AssertionError, match="must configure num_steps or num_episodes"):
        experiment_execution.build_and_run_experiment(
            _experiment(rollout=RolloutCfg(), num_rebuilds=1),
            output_dir=tmp_path,
        )

    assert closed_resources == [(policy, environment)]
