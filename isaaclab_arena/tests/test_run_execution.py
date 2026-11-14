# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import run_execution
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME,
    ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME,
    ARENA_SINGLE_RUN_TIMINGS_FILENAME,
)
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg
from isaaclab_arena.evaluation.run_status import RunStatus
from isaaclab_arena.inference.backend import InferenceUsage
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.relations.placement_diagnostics import (
    AggregatedPlacementDiagnostics,
    BatchValidationDiagnostics,
    PlacementDiagnostics,
)
from isaaclab_arena.relations.pooled_object_placer import PlacementPoolFillError
from isaaclab_arena.relations.runtime.placement_events import PLACEMENT_RESET_EVENT_NAME
from isaaclab_arena.utils.timer import Timer


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    pass


@dataclass
class _PolicyCfg(PolicyCfg):
    pass


class _Policy:
    def has_length(self):
        return False

    def get_inference_usage(self):
        return None


class _FinitePolicy(_Policy):
    def has_length(self):
        return True


class _EpisodeRecorder:
    def set_job_name(self, name):
        self.name = name

    def set_output_path(self, path):
        self.path = path


class _EventManager:
    """Has a placement reset event, whose live pool reports placement_diagnostics, unless they are None."""

    def __init__(self, placement_diagnostics):
        self.placement_diagnostics = placement_diagnostics

    def get_term_cfg(self, term_name):
        if term_name != PLACEMENT_RESET_EVENT_NAME or self.placement_diagnostics is None:
            raise ValueError(f"Event term '{term_name}' not found.")
        pool = SimpleNamespace(last_fill_diagnostics=self.placement_diagnostics)
        return SimpleNamespace(params={"placement_sampler": SimpleNamespace(pool=pool)})


def _environment(placement_diagnostics=None):
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            episode_recorder=_EpisodeRecorder(),
            event_manager=_EventManager(placement_diagnostics),
        )
    )


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
    monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", lambda cfg, rebuild_index: builder)
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
    return ArenaExperimentCfg(runs={run_cfg.name: run_cfg for run_cfg in run_cfgs}, name="experiment")


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

    before_run_ns = time.time_ns()
    result = run_execution.build_and_run(
        run,
        output_dir=tmp_path,
    )
    after_run_ns = time.time_ns()

    base_seed = run.environment_builder.seed
    run_seed_0 = deepcopy(run)  # Rebuild 0 keeps the configured seed.
    run_seed_1 = deepcopy(run)
    run_seed_1.environment_builder.seed = base_seed + 1

    assert result.run_name == "test_run"
    assert result.status is RunStatus.COMPLETED
    assert before_run_ns <= result.start_timestamp_ns <= result.end_timestamp_ns <= after_run_ns
    assert rollout_limits == [(None, 3), (None, 2)]
    # Runs are the same except for their seeds.
    assert received_run_cfgs == [run_seed_0, run_seed_1]
    # The original config is never mutated.
    assert run.rollout_limit == RolloutLimitCfg(num_episodes=5)
    assert run.environment_builder.seed == base_seed


def test_build_and_run_gives_each_rebuild_its_listed_episodes(monkeypatch, tmp_path):
    run = _run(rollout_limit=RolloutLimitCfg(num_episodes_per_rebuild=[20, 5, 5]), num_rebuilds=3)
    rollout_limits = []
    received_run_cfgs = []

    def make_environment(cfg, render_mode, **kwargs):
        received_run_cfgs.append(cfg)
        return _environment()

    monkeypatch.setattr(run_execution, "_build_environment_from_cfg", make_environment)
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: _Policy())
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(run_execution, "close_run_resources", lambda policy, env: None)
    monkeypatch.setattr(
        run_execution,
        "rollout_policy",
        lambda env, policy, num_steps, num_episodes: rollout_limits.append((num_steps, num_episodes)),
    )

    run_execution.build_and_run(run, output_dir=tmp_path)

    assert rollout_limits == [(None, 20), (None, 5), (None, 5)]
    assert [rebuild_cfg.environment_builder.seed for rebuild_cfg in received_run_cfgs] == [
        run.environment_builder.seed + rebuild_index for rebuild_index in range(3)
    ]


@pytest.mark.parametrize(
    ("configured_episodes", "num_rebuilds", "expected_rollout_limits"),
    [
        (None, 1, [(None, 3)]),
        (None, 2, [(None, 3), (None, 3)]),
        (8, 1, [(None, 8)]),
        (8, 2, [(None, 4), (None, 4)]),
    ],
)
def test_build_and_run_resolves_variation_replay_budget(
    monkeypatch,
    tmp_path,
    configured_episodes,
    num_rebuilds,
    expected_rollout_limits,
):
    variation_samples_path = tmp_path / "variation_samples.jsonl"
    variation_samples_path.write_text("\n".join(['{"variations": {}}'] * 3) + "\n")
    run = _run(
        environment_builder=ArenaEnvBuilderCfg(recorded_variation_samples_path=str(variation_samples_path)),
        rollout_limit=RolloutLimitCfg(num_episodes=configured_episodes),
        num_rebuilds=num_rebuilds,
    )
    received_run_cfgs = []
    rollout_limits = []

    def make_environment(cfg, render_mode, **kwargs):
        received_run_cfgs.append(cfg)
        return _environment()

    monkeypatch.setattr(run_execution, "_build_environment_from_cfg", make_environment)
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: _Policy())
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(run_execution, "close_run_resources", lambda policy, env: None)
    monkeypatch.setattr(
        run_execution,
        "rollout_policy",
        lambda env, policy, num_steps, num_episodes: rollout_limits.append((num_steps, num_episodes)),
    )

    run_execution.build_and_run(run, output_dir=tmp_path)

    assert rollout_limits == expected_rollout_limits
    assert all(
        rebuild_cfg.environment_builder.recorded_variation_samples_path == str(variation_samples_path)
        for rebuild_cfg in received_run_cfgs
    )


@pytest.mark.parametrize(
    ("configured_episodes", "expected_rollout_limits"),
    [(None, [(None, 1), (None, 3)]), (5, [(None, 3), (None, 2)])],
)
def test_build_and_run_assigns_one_replay_file_per_rebuild(
    monkeypatch,
    tmp_path,
    configured_episodes,
    expected_rollout_limits,
):
    replay_paths = [tmp_path / "rebuild0.jsonl", tmp_path / "rebuild1.jsonl"]
    replay_paths[0].write_text('{"variations": {}}\n')
    replay_paths[1].write_text("\n".join(['{"variations": {}}'] * 3) + "\n")
    run = _run(
        rollout_limit=RolloutLimitCfg(num_episodes=configured_episodes),
        recorded_variation_samples_paths=[str(path) for path in replay_paths],
    )
    received_run_cfgs = []
    rollout_limits = []

    def make_environment(cfg, render_mode, **kwargs):
        received_run_cfgs.append(cfg)
        return _environment()

    monkeypatch.setattr(run_execution, "_build_environment_from_cfg", make_environment)
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: _Policy())
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(run_execution, "close_run_resources", lambda policy, env: None)
    monkeypatch.setattr(
        run_execution,
        "rollout_policy",
        lambda env, policy, num_steps, num_episodes: rollout_limits.append((num_steps, num_episodes)),
    )

    run_execution.build_and_run(run, output_dir=tmp_path)

    assert rollout_limits == expected_rollout_limits
    assert [rebuild_cfg.environment_builder.recorded_variation_samples_path for rebuild_cfg in received_run_cfgs] == [
        str(path) for path in replay_paths
    ]
    assert [rebuild_cfg.environment_builder.seed for rebuild_cfg in received_run_cfgs] == [
        run.environment_builder.seed,
        run.environment_builder.seed + 1,
    ]
    assert all(rebuild_cfg.recorded_variation_samples_paths is None for rebuild_cfg in received_run_cfgs)
    assert run.environment_builder.recorded_variation_samples_path is None


def test_recorded_variation_replay_rejects_finite_action_policy():
    run = _run(
        environment_builder=ArenaEnvBuilderCfg(recorded_variation_samples_path="unused.jsonl"),
        rollout_limit=RolloutLimitCfg(num_episodes=1),
        num_rebuilds=1,
    )

    with pytest.raises(AssertionError, match="finite-action policy"):
        run_execution._resolve_rollout_limit(run, _FinitePolicy(), num_episodes=1)


def test_cfg_for_rebuild_offsets_seed_per_rebuild():
    run = _run(num_rebuilds=3)
    base_seed = run.environment_builder.seed

    assert run_execution._cfg_for_rebuild(run, 0).environment_builder.seed == base_seed
    assert run_execution._cfg_for_rebuild(run, 1).environment_builder.seed == base_seed + 1
    assert run_execution._cfg_for_rebuild(run, 2).environment_builder.seed == base_seed + 2
    # The original config is never mutated.
    assert run.environment_builder.seed == base_seed


class _InferenceUsagePolicy(_Policy):
    """Reports how many policies were built before it as its request count, so each rebuild's usage differs."""

    num_built = 0

    def __init__(self):
        self.build_index = _InferenceUsagePolicy.num_built
        _InferenceUsagePolicy.num_built += 1

    def get_inference_usage(self):
        return InferenceUsage(endpoint="test", model="test-model", num_requests=self.build_index)


def _expected_inference_usage(num_rebuilds):
    return [
        asdict(InferenceUsage(endpoint="test", model="test-model", num_requests=build_index))
        for build_index in range(num_rebuilds)
    ]


def _patch_build_and_run(monkeypatch, policy_type, rollout_policy):
    _InferenceUsagePolicy.num_built = 0
    monkeypatch.setattr(run_execution, "_build_environment_from_cfg", lambda cfg, render_mode, **kwargs: _environment())
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: policy_type())
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(run_execution, "close_run_resources", lambda policy, env: None)
    monkeypatch.setattr(run_execution, "rollout_policy", rollout_policy)


@pytest.mark.parametrize("policy_type", [_Policy, _InferenceUsagePolicy])
def test_build_and_run_writes_inference_usage_of_every_rebuild(monkeypatch, tmp_path, policy_type):
    _patch_build_and_run(monkeypatch, policy_type, lambda env, policy, num_steps, num_episodes: None)

    run_execution.build_and_run(_run(), output_dir=tmp_path)

    usage_path = tmp_path / ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME
    if policy_type is _Policy:
        assert not usage_path.exists()
        return
    assert json.loads(usage_path.read_text(encoding="utf-8")) == _expected_inference_usage(num_rebuilds=2)


def test_build_and_run_writes_inference_usage_of_a_failed_rollout(monkeypatch, tmp_path):
    def rollout_policy(env, policy, num_steps, num_episodes):
        if policy.build_index == 1:
            raise RuntimeError("rollout failed")

    _patch_build_and_run(monkeypatch, _InferenceUsagePolicy, rollout_policy)

    with pytest.raises(RuntimeError, match="rollout failed"):
        run_execution.build_and_run(_run(), output_dir=tmp_path)

    usage_path = tmp_path / ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME
    assert json.loads(usage_path.read_text(encoding="utf-8")) == _expected_inference_usage(num_rebuilds=2)


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
        return ArenaRunResult(
            run_name=run_cfg.name,
            status=RunStatus.COMPLETED,
            start_timestamp_ns=time.time_ns(),
            end_timestamp_ns=time.time_ns(),
        )

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
        return ArenaRunResult(
            run_name=run_cfg.name,
            status=RunStatus.COMPLETED,
            start_timestamp_ns=time.time_ns(),
            end_timestamp_ns=time.time_ns(),
        )

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    before_experiment_ns = time.time_ns()
    results = run_execution.execute_experiment(
        _experiment(_run(name="failing"), _run(name="passing")),
        output_dir=tmp_path,
        continue_on_error=True,
    )
    after_experiment_ns = time.time_ns()

    assert attempted == ["failing", "passing"]
    assert [(result.run_name, result.status) for result in results] == [
        ("failing", RunStatus.FAILED),
        ("passing", RunStatus.COMPLETED),
    ]
    start_timestamps_ns = [result.start_timestamp_ns for result in results]
    end_timestamps_ns = [result.end_timestamp_ns for result in results]
    assert before_experiment_ns <= start_timestamps_ns[0] <= end_timestamps_ns[0] <= after_experiment_ns
    assert start_timestamps_ns[1] <= end_timestamps_ns[1] <= after_experiment_ns
    assert end_timestamps_ns[0] <= end_timestamps_ns[1]


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


@pytest.mark.parametrize("record_trajectories", [False, True])
def test_build_environment_exports_recorder_dataset_to_the_run_directory(monkeypatch, tmp_path, record_trajectories):
    builder = _ArenaEnvBuilder(_RecorderManagerCfg())
    captured = {}
    run = _run(name="pick", environment_builder=ArenaEnvBuilderCfg(record_trajectories=record_trajectories))

    def capture_builder(cfg, rebuild_index):
        captured["environment_builder"] = cfg.environment_builder
        captured["rebuild_index"] = rebuild_index
        return builder

    monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", capture_builder)

    run_execution._build_environment_from_cfg(
        run,
        render_mode="rgb_array",
        output_dir=tmp_path,
        rebuild_index=2,
    )

    assert captured["environment_builder"].record_trajectories is record_trajectories
    assert captured["environment_builder"].recorder_dataset_export_dir_path == str(tmp_path)
    assert captured["environment_builder"].recorder_dataset_filename == "dataset_pick_rebuild2"
    assert captured["rebuild_index"] == 2
    assert run.environment_builder.recorder_dataset_export_dir_path is None
    assert run.environment_builder.recorder_dataset_filename is None
    assert builder.made_with.render_mode == "rgb_array"


def test_build_environment_tolerates_an_environment_without_recorders(monkeypatch, tmp_path):
    builder = _builder(monkeypatch, None)
    run = _run(environment_builder=ArenaEnvBuilderCfg(record_trajectories=True))

    env = run_execution._build_environment_from_cfg(run, render_mode=None, output_dir=tmp_path)

    assert env is not None
    assert builder.made_with.render_mode is None


def test_execute_experiment_writes_one_timings_file_per_run(monkeypatch, tmp_path):
    """Each Run's timings cover that Run alone, matching what OSMO produces per process."""

    def build_and_run(run_cfg, output_dir, video_cfg):
        with Timer(run_cfg.name):
            pass
        return ArenaRunResult(
            run_name=run_cfg.name,
            status=RunStatus.COMPLETED,
            start_timestamp_ns=time.time_ns(),
            end_timestamp_ns=time.time_ns(),
        )

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    run_execution.execute_experiment(_experiment(_run(name="first"), _run(name="second")), output_dir=tmp_path)

    for run_name in ("first", "second"):
        records = json.loads(
            (tmp_path / run_name / ARENA_SINGLE_RUN_TIMINGS_FILENAME).read_text(encoding="utf-8"),
        )
        # The registry is cleared between Runs, so a Run's file holds only its own timer.
        assert [record["name"] for record in records] == [run_name]
        assert records[0]["app_name"] == "experiment_runner"


def test_execute_experiment_writes_no_timings_for_a_failed_run(monkeypatch, tmp_path):
    """A failed Run leaves no timings file, which is what the Experiment aggregation expects."""

    def build_and_run(run_cfg, output_dir, video_cfg):
        if run_cfg.name == "failing":
            raise RuntimeError("rollout failed")
        with Timer("step"):
            pass
        return ArenaRunResult(
            run_name=run_cfg.name,
            status=RunStatus.COMPLETED,
            start_timestamp_ns=time.time_ns(),
            end_timestamp_ns=time.time_ns(),
        )

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    run_execution.execute_experiment(
        _experiment(_run(name="failing"), _run(name="passing")),
        output_dir=tmp_path,
        continue_on_error=True,
    )

    assert not (tmp_path / "failing" / ARENA_SINGLE_RUN_TIMINGS_FILENAME).exists()
    assert (tmp_path / "passing" / ARENA_SINGLE_RUN_TIMINGS_FILENAME).is_file()


def test_execute_experiment_writes_the_placement_diagnostics_that_built_the_rebuilds_of_each_run(monkeypatch, tmp_path):
    """Each Run writes the diagnostics of the fills that built its rebuilds, although its rollouts can refill the
    pool. A Run whose rollout fails writes the fills it built, and a Run whose fill fails writes that fill too."""
    first_build = PlacementDiagnostics(2, [BatchValidationDiagnostics(4, 3, {})], fallback_envs=[1])
    second_build = PlacementDiagnostics(2, [BatchValidationDiagnostics(4, 4, {})])
    failed_fill = PlacementDiagnostics(2, [BatchValidationDiagnostics(4, 0, {})] * 2)
    # Rebuild 0 of the passing Run solves no relations.
    diagnostics_per_build = {
        ("passing", 1): first_build,
        ("passing", 2): second_build,
        ("failing_rollout", 0): first_build,
        ("failing_fill", 0): second_build,
    }

    def build_environment(cfg, render_mode, output_dir, rebuild_index):
        if (cfg.name, rebuild_index) == ("failing_fill", 1):
            raise PlacementPoolFillError("pool could not fill", failed_fill)
        return _environment(diagnostics_per_build.get((cfg.name, rebuild_index)))

    def roll_out(env, policy, num_steps, num_episodes):
        # A reset that refills the pool replaces its diagnostics.
        refill = PlacementDiagnostics(2, [BatchValidationDiagnostics(4, 0, {})])
        env.unwrapped.event_manager.placement_diagnostics = refill
        if env.unwrapped.episode_recorder.name == "failing_rollout":
            raise RuntimeError("rollout failed")

    monkeypatch.setattr(run_execution, "_build_environment_from_cfg", build_environment)
    monkeypatch.setattr(run_execution, "_build_policy_from_cfg", lambda cfg: _Policy())
    monkeypatch.setattr(run_execution, "wrap_env_for_video", lambda env, video_cfg, steps, episodes: env)
    monkeypatch.setattr(run_execution, "close_run_resources", lambda policy, env: None)
    monkeypatch.setattr(run_execution, "rollout_policy", roll_out)

    results = run_execution.execute_experiment(
        _experiment(
            _run(name="passing", num_rebuilds=3),
            _run(name="failing_rollout", num_rebuilds=1),
            _run(name="failing_fill", num_rebuilds=2),
        ),
        output_dir=tmp_path,
        continue_on_error=True,
    )

    assert [result.status for result in results] == [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.FAILED]
    expected_fills_per_run = {
        "passing": [first_build, second_build],
        "failing_rollout": [first_build],
        "failing_fill": [second_build, failed_fill],
    }
    for run_name, expected_fills in expected_fills_per_run.items():
        path = tmp_path / run_name / ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME
        assert AggregatedPlacementDiagnostics.load(path) == AggregatedPlacementDiagnostics(expected_fills)
