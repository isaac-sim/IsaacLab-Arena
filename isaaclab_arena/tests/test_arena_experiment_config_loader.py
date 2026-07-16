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

    def load_experiment_after_startup(*_args, **_kwargs):
        assert simulation_is_running
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

    experiment_runner.main()


def test_experiment_runner_prints_overrides_without_starting_simulation(monkeypatch, capsys):
    """Inspect a typed Experiment before starting simulation or evaluation."""
    experiment_cfg = _experiment_cfg()
    experiment_cfg.runs["baseline"].rollout_limit.num_episodes = 4

    def load_experiment_for_help(config_path, *, device, overrides):
        assert config_path == GETTING_STARTED_YAML_PATH
        assert device == "cuda:0"
        assert overrides == ["runs.baseline.rollout_limit.num_episodes=4"]
        return experiment_cfg

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("override help must exit before simulation and legacy loading")

    monkeypatch.setattr(experiment_runner, "load_arena_experiment_from_config_file", load_experiment_for_help)
    monkeypatch.setattr(experiment_runner, "load_legacy_json_experiment_config", fail_if_called)
    monkeypatch.setattr(experiment_runner, "SimulationAppContext", fail_if_called)
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--show-overrides",
            "runs.baseline.rollout_limit.num_episodes=4",
        ],
    )

    experiment_runner.main()

    output = capsys.readouterr().out
    assert "Available Hydra overrides for this Arena Experiment:" in output
    assert "runs.baseline.rollout_limit.num_episodes=4" in output
    assert "runs.baseline.environment.enable_cameras=false" in output
    assert "runs.baseline.name=" not in output
    assert "runs.baseline.environment_builder.device=" not in output
    assert "experiment_cfg.runs" not in output


def test_experiment_runner_override_help_requires_typed_yaml(monkeypatch):
    """Do not present legacy JSON fields as supported Hydra overrides."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_JSON_PATH),
            "--show-overrides",
        ],
    )

    with pytest.raises(AssertionError, match="requires a typed YAML Experiment"):
        experiment_runner.main()


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
