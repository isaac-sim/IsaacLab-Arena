# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation import run_execution
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_result import ARENA_SINGLE_RUN_TIMINGS_FILENAME
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult, RolloutLimitCfg, RunStatus
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.utils.timer import Timer
from isaaclab_arena.variations import variations_hydra
from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg


@dataclass
class _EnvironmentCfg(ArenaEnvironmentCfg):
    pass


@dataclass
class _PolicyCfg(PolicyCfg):
    pass


class _Policy:
    def has_length(self):
        return False


class _FinitePolicy(_Policy):
    def has_length(self):
        return True


class _EpisodeRecorder:
    def set_job_name(self, name):
        self.name = name

    def set_output_path(self, path):
        self.path = path


class _BuildTimeVariation(BuildTimeVariationBase):
    def __init__(self):
        super().__init__(cfg=VariationBaseCfg(sampler_cfg=ChoiceSamplerCfg()), name="hdr_image")

    def _realize_at_build_time(self) -> None:
        pass


def _environment():
    return SimpleNamespace(unwrapped=SimpleNamespace(episode_recorder=_EpisodeRecorder()))


@dataclass
class _RecorderManagerCfg:
    dataset_filename: str = "environment_default_filename"
    dataset_export_dir_path: str = "environment_default_export_dir"


class _ArenaEnvBuilder:
    """Stands in for the real builder so the dataset naming can be checked without a SimulationApp."""

    def __init__(self, recorders, variations=None, hydra_overrides=None):
        self.env_cfg = SimpleNamespace(recorders=recorders)
        self.made_with = None
        self.variations = variations or {}
        self.hydra_overrides = hydra_overrides

    def make_registered(self, env_cfg=None, env_kwargs=None, render_mode=None):
        # Like ArenaEnvBuilder, apply the variation overrides while building.
        if self.hydra_overrides:
            variations_hydra.apply_overrides(self.variations, self.hydra_overrides)
        self.made_with = SimpleNamespace(env_cfg=env_cfg, env_kwargs=env_kwargs, render_mode=render_mode)
        return _environment()

    def get_all_variations(self):
        return self.variations


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


_THREE_REGISTRY_NAMES = ["mustard_bottle", "sugar_box", "tomato_soup_can"]


@pytest.mark.parametrize(
    "registry_names, num_rebuilds, random_choice, rebuild_index, replay_path, names_never_spawned",
    [
        (_THREE_REGISTRY_NAMES, 2, False, 0, None, ["tomato_soup_can"]),
        (_THREE_REGISTRY_NAMES, 1, False, 0, None, ["sugar_box", "tomato_soup_can"]),
        (_THREE_REGISTRY_NAMES, 3, False, 0, None, []),
        (_THREE_REGISTRY_NAMES, 1, True, 0, None, []),
        # The first rebuild already warned.
        (_THREE_REGISTRY_NAMES, 2, False, 1, None, []),
        # A replayed Run spawns the recorded names.
        (_THREE_REGISTRY_NAMES, 2, False, 0, "recorded_variations.jsonl", []),
        # The repeated name already spawned in the first rebuild.
        (["mustard_bottle", "sugar_box", "mustard_bottle"], 2, False, 0, None, []),
        # sugar_box, listed twice after the entry of the last rebuild, appears once, and the names keep list order.
        (
            ["mustard_bottle", "mustard_bottle", "sugar_box", "tomato_soup_can", "sugar_box"],
            1,
            False,
            0,
            None,
            ["sugar_box", "tomato_soup_can"],
        ),
    ],
)
def test_build_environment_warns_when_in_order_asset_selection_skips_names(
    monkeypatch,
    tmp_path,
    capsys,
    registry_names,
    num_rebuilds,
    random_choice,
    rebuild_index,
    replay_path,
    names_never_spawned,
):
    def build_arena_builder(cfg, rebuild_index):
        # Both variations start disabled. The Run's variation overrides enable them while the environment is built.
        variations = {
            "light": [_BuildTimeVariation()],
            "pick_up_object": [AssetSelectionVariation(SimpleNamespace(name="pick_up_object"))],
        }
        hydra_overrides = variations_hydra.overrides_from_dict(cfg.variations)
        return _ArenaEnvBuilder(None, variations=variations, hydra_overrides=hydra_overrides)

    monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", build_arena_builder)
    run = _run(
        num_rebuilds=num_rebuilds,
        environment_builder=ArenaEnvBuilderCfg(recorded_variation_samples_path=replay_path),
        variations={
            "light.hdr_image.enabled": True,
            "pick_up_object.asset_selection.enabled": True,
            "pick_up_object.asset_selection.random_choice": random_choice,
            "pick_up_object.asset_selection.registry_names": registry_names,
        },
    )

    run_execution._build_environment_from_cfg(run, render_mode=None, output_dir=tmp_path, rebuild_index=rebuild_index)

    output = capsys.readouterr().out
    if names_never_spawned:
        assert f"[WARNING] Run 'test_run' has num_rebuilds: {num_rebuilds}" in output
        assert (
            f"'pick_up_object.asset_selection' spawns one of its {len(registry_names)} registry_names entries" in output
        )
        assert f"so {names_never_spawned} never spawn" in output
    else:
        # The enabled light.hdr_image, a build-time variation as well, does not warn either.
        assert "[WARNING]" not in output


def test_execute_experiment_writes_one_timings_file_per_run(monkeypatch, tmp_path):
    """Each Run's timings cover that Run alone, matching what OSMO produces per process."""

    def build_and_run(run_cfg, output_dir, video_cfg):
        with Timer(run_cfg.name):
            pass
        return ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED)

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
        return ArenaRunResult(run_name=run_cfg.name, status=RunStatus.COMPLETED)

    monkeypatch.setattr(run_execution, "build_and_run", build_and_run)

    run_execution.execute_experiment(
        _experiment(_run(name="failing"), _run(name="passing")),
        output_dir=tmp_path,
        continue_on_error=True,
    )

    assert not (tmp_path / "failing" / ARENA_SINGLE_RUN_TIMINGS_FILENAME).exists()
    assert (tmp_path / "passing" / ARENA_SINGLE_RUN_TIMINGS_FILENAME).is_file()
