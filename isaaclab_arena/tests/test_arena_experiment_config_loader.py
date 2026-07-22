# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test loading Arena Experiments at the evaluation boundary."""

from pathlib import Path

import pytest

from isaaclab_arena.evaluation import arena_experiment_config_loader, experiment_runner
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_config_loader import (
    load_arena_experiment_from_config_file,
    validate_experiment_config_path,
)
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.experiment_runner import _assert_camera_support_enabled
from isaaclab_arena.evaluation.legacy_experiment_runner import legacy_json_experiment_requires_cameras
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

GETTING_STARTED_JSON_PATH = (
    Path(TestConstants.arena_environments_dir) / "eval_jobs_configs" / "getting_started_jobs_config.json"
)
GETTING_STARTED_YAML_PATH = (
    Path(TestConstants.arena_environments_dir) / "experiment_configs" / "getting_started_experiment.yaml"
)


def _experiment_cfg(enable_cameras: bool = False) -> ArenaExperimentCfg:
    run_cfg = ArenaRunCfg(
        name="baseline",
        environment=PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=enable_cameras),
        policy=ZeroActionPolicyCfg(),
    )
    return ArenaExperimentCfg(runs={run_cfg.name: run_cfg})


def test_load_typed_yaml_experiment_applies_overrides_and_device(monkeypatch):
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_resolve_policy_cfg_type_from_name_or_class_path",
        lambda policy_name_or_class_path: {"zero_action": ZeroActionPolicyCfg}[policy_name_or_class_path],
    )

    experiment_cfg = load_arena_experiment_from_config_file(
        GETTING_STARTED_YAML_PATH,
        device="cuda:1",
        overrides=["runs.baseline.environment.light_intensity=750.0"],
    )
    runs = experiment_cfg.runs

    assert isinstance(experiment_cfg, ArenaExperimentCfg)
    assert list(runs) == [
        "baseline",
        "swap_objects",
        "change_background_hdr",
        "parallel_envs",
    ]
    assert isinstance(runs["baseline"].environment, PickAndPlaceMapleTableEnvironmentCfg)
    assert runs["baseline"].environment.light_intensity == 750.0
    assert all(run.policy == ZeroActionPolicyCfg() for run in runs.values())
    assert all(run.environment_builder.device == "cuda:1" for run in runs.values())


def test_policy_config_type_resolves_from_dotted_class_path():
    policy_cfg_type = arena_experiment_config_loader._resolve_policy_cfg_type_from_name_or_class_path(
        "isaaclab_arena.policy.zero_action_policy.ZeroActionPolicy"
    )

    assert policy_cfg_type is ZeroActionPolicyCfg


def test_load_legacy_json_experiment_returns_canonical_cfg(tmp_path, monkeypatch):
    config_path = tmp_path / "experiment.json"
    config_path.write_text('{"jobs": []}', encoding="utf-8")
    run_cfg = _experiment_cfg().runs["baseline"]
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "run_cfgs_from_legacy_eval_config",
        lambda _config, device: [run_cfg],
    )

    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:1")

    assert isinstance(experiment_cfg, ArenaExperimentCfg)
    assert experiment_cfg.runs == {"baseline": run_cfg}


def test_empty_legacy_json_experiment_is_rejected(tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text('{"jobs": []}', encoding="utf-8")

    with pytest.raises(AssertionError, match="must contain at least one Run"):
        load_arena_experiment_from_config_file(config_path, device="cuda:0")


def test_experiment_runner_loads_typed_experiment_after_simulation_starts(monkeypatch):
    runtime_is_initialized = False
    simulation_is_running = False

    class _SimulationAppContext:
        def __init__(self, _args_cli):
            pass

        def __enter__(self):
            nonlocal simulation_is_running
            simulation_is_running = True

        def __exit__(self, _exception_type, _exception, _traceback):
            nonlocal simulation_is_running
            simulation_is_running = False

    def initialize_runtime():
        nonlocal runtime_is_initialized
        assert simulation_is_running
        runtime_is_initialized = True

    def load_experiment_after_startup(*_args, **_kwargs):
        assert simulation_is_running
        assert runtime_is_initialized
        return _experiment_cfg()

    monkeypatch.setattr(experiment_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(experiment_runner, "load_arena_experiment_from_config_file", load_experiment_after_startup)
    monkeypatch.setattr(experiment_runner, "list_variations", lambda _experiment: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--list_variations",
        ],
    )

    experiment_runner.main(runtime_initializer=initialize_runtime)

    assert runtime_is_initialized


def test_experiment_runner_initializes_runtime_after_creating_exact_output(monkeypatch, tmp_path):
    simulation_is_running = False
    runtime_is_initialized = False
    exact_output = tmp_path / "exact-experiment-output"
    experiment_cfg = _experiment_cfg()

    class _SimulationAppContext:
        def __init__(self, _args_cli):
            pass

        def __enter__(self):
            nonlocal simulation_is_running
            simulation_is_running = True

        def __exit__(self, _exception_type, _exception, _traceback):
            nonlocal simulation_is_running
            simulation_is_running = False

    def initialize_runtime():
        nonlocal runtime_is_initialized
        assert simulation_is_running
        assert exact_output.is_dir()
        runtime_is_initialized = True

    def load_experiment_after_startup(*_args, **_kwargs):
        assert simulation_is_running
        assert runtime_is_initialized
        return experiment_cfg

    def execute_experiment(received_cfg, *, output_dir, **_kwargs):
        assert received_cfg is experiment_cfg
        assert output_dir == exact_output
        return []

    monkeypatch.setattr(experiment_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(experiment_runner, "load_arena_experiment_from_config_file", load_experiment_after_startup)
    monkeypatch.setattr(experiment_runner, "execute_experiment", execute_experiment)
    monkeypatch.setattr(experiment_runner, "build_runs_info_table", lambda *_args: "")
    monkeypatch.setattr(experiment_runner, "build_report", lambda output_dir: output_dir / "report.html")
    monkeypatch.setattr(
        experiment_runner,
        "MetricsLogger",
        lambda: type(
            "_MetricsLogger", (), {"append_job_metrics": lambda *_args: None, "print_metrics": lambda *_: None}
        )(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--experiment_output_directory",
            str(exact_output),
        ],
    )

    experiment_runner.main(runtime_initializer=initialize_runtime)

    assert runtime_is_initialized


def test_typed_camera_run_requires_prelaunch_camera_flag():
    experiment_cfg = _experiment_cfg(enable_cameras=True)

    with pytest.raises(AssertionError, match="Pass --enable_cameras"):
        _assert_camera_support_enabled(experiment_cfg, enable_cameras=False)

    _assert_camera_support_enabled(experiment_cfg, enable_cameras=True)


def test_legacy_json_camera_detection_is_preserved():
    legacy_experiment_config = {"jobs": [{"arena_env_args": {}}, {"arena_env_args": {"enable_cameras": True}}]}

    assert legacy_json_experiment_requires_cameras(legacy_experiment_config)


def test_experiment_runner_rejects_yaml_chunking_before_starting_simulation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--chunk_size",
            "1",
        ],
    )

    with pytest.raises(AssertionError, match="only legacy JSON Experiments"):
        experiment_runner.main()


def test_experiment_runner_rejects_runtime_initializer_with_chunking(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_JSON_PATH),
            "--chunk_size",
            "1",
        ],
    )

    with pytest.raises(
        ValueError,
        match="runtime_initializer.*chunk_size",
    ):
        experiment_runner.main(runtime_initializer=lambda: None)


def test_experiment_runner_rejects_exact_output_for_multiple_legacy_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_JSON_PATH),
            "--chunk_size",
            "1",
            "--experiment_output_directory",
            str(tmp_path / "exact-experiment-output"),
        ],
    )

    with pytest.raises(AssertionError, match="not supported when --chunk_size dispatches multiple chunks"):
        experiment_runner.main()


def test_experiment_runner_rejects_nonempty_exact_output_before_starting_simulation(monkeypatch, tmp_path):
    exact_experiment_output_directory = tmp_path / "existing-experiment-output"
    exact_experiment_output_directory.mkdir()
    (exact_experiment_output_directory / "existing-result.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--experiment_output_directory",
            str(exact_experiment_output_directory),
        ],
    )

    with pytest.raises(AssertionError, match="is not empty.*--output_base_dir"):
        experiment_runner.main()


def test_legacy_json_experiment_rejects_hydra_overrides():
    with pytest.raises(AssertionError, match="only for typed YAML"):
        load_arena_experiment_from_config_file(
            GETTING_STARTED_JSON_PATH,
            device="cuda:0",
            overrides=["runs.baseline.rollout_limit.num_steps=2"],
        )


def test_experiment_runner_rejects_native_hydra_overrides_for_legacy_json(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_JSON_PATH),
            "runs.baseline.rollout_limit.num_steps=2",
        ],
    )

    with pytest.raises(AssertionError, match="only for typed YAML"):
        experiment_runner.main()


def test_unknown_experiment_config_file_type_is_rejected(tmp_path):
    config_path = tmp_path / "experiment.toml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="must use .json, .yaml, or .yml"):
        validate_experiment_config_path(config_path)
