# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import experiment_runner
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


def test_run_experiment_splits_episode_budget_without_mutating_config(monkeypatch, tmp_path):
    experiment = _experiment()
    rollout_limits = []
    received_builder_factories = []

    def custom_builder_factory(cfg):
        pass

    def make_environment(cfg, render_mode, arena_builder_factory):
        received_builder_factories.append(arena_builder_factory)
        return _environment()

    monkeypatch.setattr(experiment_runner, "_make_environment", make_environment)
    monkeypatch.setattr(experiment_runner, "_make_policy", lambda cfg: _Policy())
    monkeypatch.setattr(experiment_runner, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(experiment_runner, "_close_experiment_resources", lambda policy, env: None)

    def record_rollout(env, policy, num_steps, num_episodes):
        rollout_limits.append((num_steps, num_episodes))

    monkeypatch.setattr(experiment_runner, "rollout_policy", record_rollout)

    result = experiment_runner.run_experiment(
        experiment,
        output_dir=tmp_path,
        arena_builder_factory=custom_builder_factory,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert rollout_limits == [(None, 3), (None, 2)]
    assert received_builder_factories == [custom_builder_factory, custom_builder_factory]
    assert experiment.rollout == RolloutCfg(num_episodes=5)


def test_run_experiment_returns_failure_and_closes_resources(monkeypatch, tmp_path):
    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        experiment_runner,
        "_make_environment",
        lambda cfg, render_mode, arena_builder_factory: environment,
    )
    monkeypatch.setattr(experiment_runner, "_make_policy", lambda cfg: policy)
    monkeypatch.setattr(experiment_runner, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(
        experiment_runner,
        "_close_experiment_resources",
        lambda closed_policy, closed_environment: closed_resources.append((closed_policy, closed_environment)),
    )
    monkeypatch.setattr(
        experiment_runner,
        "rollout_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )

    result = experiment_runner.run_experiment(
        _experiment(rollout=RolloutCfg(num_steps=2), num_rebuilds=1),
        output_dir=tmp_path,
        arena_builder_factory=lambda cfg: None,
    )

    assert result.status is ExperimentStatus.FAILED
    assert "rollout failed" in result.error
    assert closed_resources == [(policy, environment)]
