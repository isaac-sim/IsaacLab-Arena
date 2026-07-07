# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import experiment_execution
from isaaclab_arena.evaluation.arena_experiment import (
    ArenaExperimentCfg,
    ArenaExperimentPlan,
    ExperimentStatus,
    RolloutCfg,
)
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
    received_builder_factories = []

    def custom_builder_factory(cfg):
        pass

    def make_environment(cfg, render_mode, arena_builder_factory):
        received_builder_factories.append(arena_builder_factory)
        return _environment()

    monkeypatch.setattr(experiment_execution, "_build_environment", make_environment)
    monkeypatch.setattr(experiment_execution, "_build_policy", lambda cfg: _Policy())
    monkeypatch.setattr(experiment_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(experiment_execution, "close_experiment_resources", lambda policy, env: None)

    def record_rollout(env, policy, num_steps, num_episodes):
        rollout_limits.append((num_steps, num_episodes))

    monkeypatch.setattr(experiment_execution, "rollout_policy", record_rollout)

    result = experiment_execution.build_and_run_experiment(
        ArenaExperimentPlan(
            experiment_cfg=experiment,
            arena_builder_factory=custom_builder_factory,
        ),
        output_dir=tmp_path,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert rollout_limits == [(None, 3), (None, 2)]
    assert received_builder_factories == [custom_builder_factory, custom_builder_factory]
    assert experiment.rollout == RolloutCfg(num_episodes=5)


def test_build_and_run_experiment_raises_and_closes_resources(monkeypatch, tmp_path):
    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        experiment_execution,
        "_build_environment",
        lambda cfg, render_mode, arena_builder_factory: environment,
    )
    monkeypatch.setattr(experiment_execution, "_build_policy", lambda cfg: policy)
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
            ArenaExperimentPlan(
                experiment_cfg=_experiment(rollout=RolloutCfg(num_steps=2), num_rebuilds=1),
                arena_builder_factory=lambda cfg: None,
            ),
            output_dir=tmp_path,
        )

    assert closed_resources == [(policy, environment)]


def test_build_and_run_experiment_uses_legacy_step_fallback(monkeypatch, tmp_path):
    rollout_limits = []

    monkeypatch.setattr(
        experiment_execution,
        "_build_environment",
        lambda cfg, render_mode, arena_builder_factory: _environment(),
    )
    monkeypatch.setattr(experiment_execution, "_build_policy", lambda cfg: _Policy())
    monkeypatch.setattr(experiment_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(experiment_execution, "close_experiment_resources", lambda policy, env: None)
    monkeypatch.setattr(
        experiment_execution,
        "rollout_policy",
        lambda env, policy, num_steps, num_episodes: rollout_limits.append((num_steps, num_episodes)),
    )

    result = experiment_execution.build_and_run_experiment(
        ArenaExperimentPlan(
            experiment_cfg=_experiment(rollout=RolloutCfg(), num_rebuilds=1),
            arena_builder_factory=lambda cfg: None,
        ),
        output_dir=tmp_path,
        fallback_num_steps=17,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert rollout_limits == [(17, None)]
