# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for run_execution's Run/Experiment orchestration and datagen recorder-term wiring.

Importing run_execution needs a running SimulationApp (it imports RecorderManagerBaseCfg
from isaaclab.managers.recorder_manager) -- so each test case follows the established
_test_/test_ + run_function_with_persistent_simulation_app pattern from
isaaclab_arena/tests/test_task_registry.py. Helpers that only reference plain Arena
configs (no isaaclab imports) are defined once at module scope and shared across cases.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg, RunStatus
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


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


def _test_build_and_run_splits_episode_budget_without_mutating_config(simulation_app, monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import run_execution

    run = _run()
    rollout_limits = []
    received_run_cfgs = []

    def make_environment(cfg, render_mode, datagen_collector_factory=None):
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
    return True


def test_build_and_run_splits_episode_budget_without_mutating_config(monkeypatch, tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_build_and_run_splits_episode_budget_without_mutating_config, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_seed_cfg_for_rebuild_offsets_seed_per_rebuild(simulation_app):
    from isaaclab_arena.evaluation import run_execution

    run = _run(num_rebuilds=3)
    base_seed = run.environment_builder.seed

    assert run_execution._seed_cfg_for_rebuild(run, 0).environment_builder.seed == base_seed
    assert run_execution._seed_cfg_for_rebuild(run, 1).environment_builder.seed == base_seed + 1
    assert run_execution._seed_cfg_for_rebuild(run, 2).environment_builder.seed == base_seed + 2
    # The original config is never mutated.
    assert run.environment_builder.seed == base_seed
    return True


def test_seed_cfg_for_rebuild_offsets_seed_per_rebuild():
    assert run_function_with_persistent_simulation_app(_test_seed_cfg_for_rebuild_offsets_seed_per_rebuild)


def _test_build_and_run_raises_and_closes_resources(simulation_app, monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import run_execution

    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        run_execution,
        "_build_environment_from_cfg",
        lambda cfg, render_mode, datagen_collector_factory=None: environment,
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
    return True


def test_build_and_run_raises_and_closes_resources(monkeypatch, tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_build_and_run_raises_and_closes_resources, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_build_and_run_requires_a_limit_for_an_unbounded_policy(simulation_app, monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import run_execution

    closed_resources = []
    environment = _environment()
    policy = _Policy()

    monkeypatch.setattr(
        run_execution,
        "_build_environment_from_cfg",
        lambda cfg, render_mode, datagen_collector_factory=None: environment,
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
    return True


def test_build_and_run_requires_a_limit_for_an_unbounded_policy(monkeypatch, tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_build_and_run_requires_a_limit_for_an_unbounded_policy, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_execute_experiment_runs_in_declaration_order(simulation_app, monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import run_execution

    received = []

    def build_and_run(run_cfg, output_dir, video_cfg, datagen_collector_factory=None):
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
    return True


def test_execute_experiment_runs_in_declaration_order(monkeypatch, tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_execute_experiment_runs_in_declaration_order, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_execute_experiment_records_failure_and_continues(simulation_app, monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import run_execution

    attempted = []

    def build_and_run(run_cfg, output_dir, video_cfg, datagen_collector_factory=None):
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
    return True


def test_execute_experiment_records_failure_and_continues(monkeypatch, tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_execute_experiment_records_failure_and_continues, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_execute_experiment_stops_on_failure_by_default(simulation_app, monkeypatch, tmp_path):
    from isaaclab_arena.evaluation import run_execution

    attempted = []

    def build_and_run(run_cfg, output_dir, video_cfg, datagen_collector_factory=None):
        attempted.append(run_cfg.name)
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    with pytest.raises(RuntimeError, match="rollout failed"):
        run_execution.execute_experiment(
            _experiment(_run(name="failing"), _run(name="not_attempted")),
            output_dir=tmp_path,
        )

    assert attempted == ["failing"]
    return True


def test_execute_experiment_stops_on_failure_by_default(monkeypatch, tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_execute_experiment_stops_on_failure_by_default, monkeypatch=monkeypatch, tmp_path=tmp_path
    )


def _test_merges_datagen_term_into_none_recorders_cfg(simulation_app):
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.evaluation.run_execution import _with_datagen_recorder_term
    from isaaclab_arena.recording.callback_recorder_term import CallbackRecorderTermHandlers

    merged = _with_datagen_recorder_term(None, build_handlers=lambda env: CallbackRecorderTermHandlers())

    assert isinstance(merged, RecorderManagerBaseCfg)
    assert hasattr(merged, "datagen_callback")
    return True


def test_merges_datagen_term_into_none_recorders_cfg():
    assert run_function_with_persistent_simulation_app(_test_merges_datagen_term_into_none_recorders_cfg)


def _test_preserves_existing_recorder_terms_alongside_datagen_term(simulation_app):
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.evaluation.run_execution import _with_datagen_recorder_term
    from isaaclab_arena.recording.callback_recorder_term import CallbackRecorderTermHandlers
    from isaaclab_arena.utils.configclass import make_configclass

    ExistingRecordersCfg = make_configclass(
        "ExistingRecordersCfg", [("existing_term", object, "sentinel")], bases=(RecorderManagerBaseCfg,)
    )
    existing = ExistingRecordersCfg()
    # A pre-set value on a field RecorderManagerBaseCfg also declares (not just existing_term,
    # which only datagen_recorders_cfg lacks): merging must not let datagen_recorders_cfg's
    # inherited base-class default for this field silently overwrite it.
    existing.dataset_filename = "already_configured_dataset"

    merged = _with_datagen_recorder_term(existing, build_handlers=lambda env: CallbackRecorderTermHandlers())

    assert merged.existing_term == "sentinel"
    assert merged.dataset_filename == "already_configured_dataset"
    assert hasattr(merged, "datagen_callback")
    return True


def test_preserves_existing_recorder_terms_alongside_datagen_term():
    assert run_function_with_persistent_simulation_app(_test_preserves_existing_recorder_terms_alongside_datagen_term)
