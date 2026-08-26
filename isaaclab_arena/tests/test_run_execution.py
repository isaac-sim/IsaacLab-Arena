# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import run_execution
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg, RunStatus
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


@dataclass
class _RecorderManagerCfg:
    dataset_filename: str = "environment_default_filename"
    dataset_export_dir_path: str = "environment_default_export_dir"


class _ArenaEnvBuilder:
    """Stands in for the real builder so the dataset naming can be checked without a SimulationApp."""

    def __init__(self, recorders):
        self.env_cfg = SimpleNamespace(recorders=recorders)
        self.made_with = None

    def make_registered(self, env_cfg=None, env_kwargs=None, render_mode=None):
        self.made_with = SimpleNamespace(env_cfg=env_cfg, env_kwargs=env_kwargs, render_mode=render_mode)
        return _environment()


def _builder(monkeypatch, recorders):
    builder = _ArenaEnvBuilder(recorders)
    monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", lambda cfg: builder)
    return builder


def _run(**overrides):
    values = {
        "name": "test_run",
        "environment": _EnvironmentCfg(),
        "policy": _PolicyCfg(),
        "rollout_limit": RolloutLimitCfg(num_episodes=5),
        "num_rebuilds": 2,
    }
    values.update(overrides)
    return ArenaRunCfg(**values)


def _experiment(*run_cfgs: ArenaRunCfg) -> ArenaExperimentCfg:
    return ArenaExperimentCfg(runs={run_cfg.name: run_cfg for run_cfg in run_cfgs})


def test_build_and_run_splits_episode_budget_without_mutating_config(monkeypatch, tmp_path):
    run = _run()
    rollout_limits = []
    received_run_cfgs = []

    def make_environment(cfg, render_mode, **kwargs):
        received_run_cfgs.append(cfg)
        return _environment()

    monkeypatch.setattr(run_execution, "_build_environment_from_cfg", make_environment)
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: _Policy())
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(run_execution, "close_run_resources", lambda policy, env: None)

    def record_rollout(env, policy, num_steps, num_episodes):
        rollout_limits.append((num_steps, num_episodes))

    monkeypatch.setattr(run_execution, "rollout_policy", record_rollout)

    result = run_execution.build_and_run(
        run,
        output_dir=tmp_path,
    )

    base_seed = run.environment_builder.seed
    run_seed_0 = deepcopy(run)  # Rebuild 0 keeps the configured seed.
    run_seed_1 = deepcopy(run)
    run_seed_1.environment_builder.seed = base_seed + 1

    assert result.run_name == "test_run"
    assert result.status is RunStatus.COMPLETED
    assert rollout_limits == [(None, 3), (None, 2)]
    # Runs are the same except for their seeds.
    assert received_run_cfgs == [run_seed_0, run_seed_1]
    # The original config is never mutated.
    assert run.rollout_limit == RolloutLimitCfg(num_episodes=5)
    assert run.environment_builder.seed == base_seed


def test_seed_cfg_for_rebuild_offsets_seed_per_rebuild():
    run = _run(num_rebuilds=3)
    base_seed = run.environment_builder.seed

    assert run_execution._seed_cfg_for_rebuild(run, 0).environment_builder.seed == base_seed
    assert run_execution._seed_cfg_for_rebuild(run, 1).environment_builder.seed == base_seed + 1
    assert run_execution._seed_cfg_for_rebuild(run, 2).environment_builder.seed == base_seed + 2
    # The original config is never mutated.
    assert run.environment_builder.seed == base_seed


def test_build_and_run_raises_and_closes_resources(monkeypatch, tmp_path):
    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        run_execution,
        "_build_environment_from_cfg",
        lambda cfg, render_mode, **kwargs: environment,
    )
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: policy)
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(
        run_execution,
        "close_run_resources",
        lambda closed_policy, closed_environment: closed_resources.append((closed_policy, closed_environment)),
    )
    monkeypatch.setattr(
        run_execution,
        "rollout_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )

    with pytest.raises(RuntimeError, match="rollout failed"):
        run_execution.build_and_run(
            _run(rollout_limit=RolloutLimitCfg(num_steps=2), num_rebuilds=1),
            output_dir=tmp_path,
        )

    assert closed_resources == [(policy, environment)]


def test_build_and_run_requires_a_limit_for_an_unbounded_policy(monkeypatch, tmp_path):
    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        run_execution,
        "_build_environment_from_cfg",
        lambda cfg, render_mode, **kwargs: environment,
    )
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: policy)
    monkeypatch.setattr(
        run_execution,
        "close_run_resources",
        lambda closed_policy, closed_environment: closed_resources.append((closed_policy, closed_environment)),
    )

    with pytest.raises(AssertionError, match="must configure num_steps or num_episodes"):
        run_execution.build_and_run(
            _run(rollout_limit=RolloutLimitCfg(), num_rebuilds=1),
            output_dir=tmp_path,
        )

    assert closed_resources == [(policy, environment)]


def test_execute_experiment_runs_in_declaration_order(monkeypatch, tmp_path):
    received = []

    def build_and_run(run_cfg, output_dir, video_cfg):
        received.append((run_cfg.name, output_dir, video_cfg.video_base_dir))
        return ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED)

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    results = run_execution.execute_experiment(
        _experiment(_run(name="first"), _run(name="second")),
        output_dir=tmp_path,
    )

    assert [result.run_name for result in results] == ["first", "second"]
    assert received == [
        ("first", tmp_path / "first", str(tmp_path / "first")),
        ("second", tmp_path / "second", str(tmp_path / "second")),
    ]


def test_execute_experiment_records_failure_and_continues(monkeypatch, tmp_path):
    attempted = []

    def build_and_run(run_cfg, output_dir, video_cfg):
        attempted.append(run_cfg.name)
        if run_cfg.name == "failing":
            raise RuntimeError("rollout failed")
        return ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED)

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    results = run_execution.execute_experiment(
        _experiment(_run(name="failing"), _run(name="passing")),
        output_dir=tmp_path,
        continue_on_error=True,
    )

    assert attempted == ["failing", "passing"]
    assert [(result.run_name, result.status) for result in results] == [
        ("failing", RunStatus.FAILED),
        ("passing", RunStatus.COMPLETED),
    ]


def test_execute_experiment_stops_on_failure_by_default(monkeypatch, tmp_path):
    attempted = []

    def build_and_run(run_cfg, output_dir, video_cfg):
        attempted.append(run_cfg.name)
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    with pytest.raises(RuntimeError, match="rollout failed"):
        run_execution.execute_experiment(
            _experiment(_run(name="failing"), _run(name="not_attempted")),
            output_dir=tmp_path,
        )

    assert attempted == ["failing"]


def test_build_environment_names_the_dataset_per_rebuild(monkeypatch):
    builder = _ArenaEnvBuilder(_RecorderManagerCfg())
    captured = {}

    def capture_builder(cfg):
        captured["environment_builder"] = cfg.environment_builder
        return builder

    monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", capture_builder)

    run_execution._build_environment_from_cfg(_run(name="pick"), render_mode="rgb_array", rebuild_index=2)

    # Every rebuild recreates the dataset file, so the index has to distinguish them.
    assert captured["environment_builder"].recorder_dataset_filename == "dataset_pick_rebuild2"
    # Trajectory recording is off, so the builder keeps the environment's own export directory.
    assert captured["environment_builder"].record_trajectories is False
    assert captured["environment_builder"].recorder_dataset_export_dir_path is None
    assert builder.made_with.render_mode == "rgb_array"


def test_build_environment_exports_configured_trajectories_to_the_run_directory(monkeypatch, tmp_path):
    builder = _ArenaEnvBuilder(_RecorderManagerCfg())
    captured = {}
    run = _run(name="pick", environment_builder=ArenaEnvBuilderCfg(record_trajectories=True))

    def capture_builder(cfg):
        captured["environment_builder"] = cfg.environment_builder
        return builder

    monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", capture_builder)

    run_execution._build_environment_from_cfg(
        run,
        render_mode=None,
        output_dir=tmp_path,
        rebuild_index=1,
    )

    assert captured["environment_builder"].record_trajectories is True
    assert captured["environment_builder"].recorder_dataset_export_dir_path == str(tmp_path)
    assert captured["environment_builder"].recorder_dataset_filename == "dataset_pick_rebuild1"
    assert run.environment_builder.recorder_dataset_export_dir_path is None
    assert run.environment_builder.recorder_dataset_filename is None


def test_build_environment_tolerates_an_environment_without_recorders(monkeypatch):
    builder = _builder(monkeypatch, None)
    run = _run(environment_builder=ArenaEnvBuilderCfg(record_trajectories=True))

    env = run_execution._build_environment_from_cfg(run, render_mode=None)

    assert env is not None
    assert builder.made_with.render_mode is None
