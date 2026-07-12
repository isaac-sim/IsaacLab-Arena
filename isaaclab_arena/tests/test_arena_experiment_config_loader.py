# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test loading Arena Experiments at the evaluation boundary."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation import arena_experiment_config_loader, eval_runner, legacy_eval_runner
from isaaclab_arena.evaluation.arena_experiment_config_loader import (
    load_arena_experiment_from_config_file,
    validate_experiment_config_path,
)
from isaaclab_arena.evaluation.eval_runner import _assert_camera_support_enabled, run_local_experiment
from isaaclab_arena.evaluation.eval_runner_cli import parse_eval_runner_cfg
from isaaclab_arena.evaluation.legacy_eval_runner import _run_legacy_json_chunk, legacy_json_experiment_requires_cameras
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.utils.isaaclab_utils import simulation_app
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

GETTING_STARTED_JSON_PATH = (
    Path(TestConstants.arena_environments_dir) / "eval_jobs_configs" / "getting_started_jobs_config.json"
)
GETTING_STARTED_YAML_PATH = (
    Path(TestConstants.arena_environments_dir) / "experiment_configs" / "getting_started_experiment.yaml"
)


def test_load_typed_yaml_experiment_applies_overrides_and_device(monkeypatch):
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_policy_cfg_types",
        lambda: {"zero_action": ZeroActionPolicyCfg},
    )

    experiment = load_arena_experiment_from_config_file(
        GETTING_STARTED_YAML_PATH,
        device="cuda:1",
        overrides=["runs.baseline.environment.light_intensity=750.0"],
    )

    assert [run.name for run in experiment] == [
        "baseline",
        "swap_objects",
        "change_background_hdr",
        "parallel_envs",
    ]
    assert isinstance(experiment[0].environment, PickAndPlaceMapleTableEnvironmentCfg)
    assert experiment[0].environment.light_intensity == 750.0
    assert all(run.policy == ZeroActionPolicyCfg() for run in experiment)
    assert all(run.environment_builder.device == "cuda:1" for run in experiment)


def test_eval_runner_loads_typed_experiment_after_simulation_starts(monkeypatch):
    simulation_is_running = False
    composed_experiment = [SimpleNamespace(name="baseline", environment=SimpleNamespace(enable_cameras=False))]

    class _SimulationAppContext:
        def __init__(self, _args_cli):
            pass

        def __enter__(self):
            nonlocal simulation_is_running
            simulation_is_running = True

        def __exit__(self, _exception_type, _exception, _traceback):
            nonlocal simulation_is_running
            simulation_is_running = False

    def load_experiment_after_startup(path, device, overrides):
        assert simulation_is_running
        assert path == GETTING_STARTED_YAML_PATH
        assert device == "cuda:1"
        assert overrides == ["runs.baseline.rollout_limit.num_steps=7"]
        return composed_experiment

    monkeypatch.setattr(eval_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(eval_runner, "load_arena_experiment_from_config_file", load_experiment_after_startup)
    monkeypatch.setattr(eval_runner, "list_variations", lambda _experiment: None)

    cfg = parse_eval_runner_cfg([
        "--experiment-config",
        str(GETTING_STARTED_YAML_PATH),
        "--list-variations",
        "--device",
        "cuda:1",
        "runs.baseline.rollout_limit.num_steps=7",
    ])

    assert run_local_experiment(cfg) == 0


def test_typed_camera_run_requires_prelaunch_camera_flag():
    camera_run = SimpleNamespace(
        name="camera_run",
        environment=SimpleNamespace(enable_cameras=True),
    )

    with pytest.raises(AssertionError, match="Pass --enable_cameras"):
        _assert_camera_support_enabled([camera_run], enable_cameras=False)

    _assert_camera_support_enabled([camera_run], enable_cameras=True)


def test_legacy_json_camera_detection_is_preserved():
    legacy_experiment_config = {"jobs": [{"arena_env_args": {}}, {"arena_env_args": {"enable_cameras": True}}]}

    assert legacy_json_experiment_requires_cameras(legacy_experiment_config)


def test_eval_runner_rejects_yaml_chunking_before_starting_simulation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["eval_runner.py", "--experiment-config", str(GETTING_STARTED_YAML_PATH), "--chunk-size", "1"],
    )
    with pytest.raises(AssertionError, match="only legacy JSON Experiments"):
        eval_runner.main()


def test_list_variations_preserves_precedence_over_legacy_chunking(monkeypatch):
    class _SimulationAppContext:
        def __init__(self, _launcher_args):
            pass

        def __enter__(self):
            pass

        def __exit__(self, _exception_type, _exception, _traceback):
            pass

    listed_experiment = [SimpleNamespace(name="baseline", environment=SimpleNamespace(enable_cameras=False))]
    monkeypatch.setattr(eval_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(
        eval_runner,
        "load_arena_experiment_from_config_file",
        lambda *_args, **_kwargs: listed_experiment,
    )
    monkeypatch.setattr(eval_runner, "list_variations", lambda experiment: experiment is listed_experiment)
    monkeypatch.setattr(
        eval_runner,
        "run_legacy_json_in_chunks",
        lambda *_args, **_kwargs: pytest.fail("list-variations unexpectedly dispatched chunks"),
    )

    cfg = parse_eval_runner_cfg([
        "--experiment-config",
        str(GETTING_STARTED_JSON_PATH),
        "--chunk-size",
        "1",
        "--list-variations",
    ])

    assert run_local_experiment(cfg) == 0


def test_legacy_json_experiment_rejects_hydra_overrides():
    with pytest.raises(AssertionError, match="only for typed YAML"):
        load_arena_experiment_from_config_file(
            GETTING_STARTED_JSON_PATH,
            device="cuda:0",
            overrides=["runs.baseline.rollout_limit.num_steps=2"],
        )


def test_eval_runner_rejects_native_hydra_overrides_for_legacy_json(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_runner.py",
            "--experiment-config",
            str(GETTING_STARTED_JSON_PATH),
            "runs.baseline.rollout_limit.num_steps=2",
        ],
    )
    with pytest.raises(AssertionError, match="only for typed YAML"):
        eval_runner.main()


def test_record_camera_video_preserves_automatic_app_launcher_camera_support(monkeypatch):
    class _SimulationAppContext:
        def __init__(self, launcher_args):
            assert launcher_args["enable_cameras"] is True

        def __enter__(self):
            pass

        def __exit__(self, _exception_type, _exception, _traceback):
            pass

    monkeypatch.setattr(eval_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(
        eval_runner,
        "load_arena_experiment_from_config_file",
        lambda *_args, **_kwargs: [SimpleNamespace(name="baseline", environment=SimpleNamespace())],
    )
    monkeypatch.setattr(eval_runner, "list_variations", lambda _experiment: None)

    cfg = parse_eval_runner_cfg([
        "--experiment-config",
        str(GETTING_STARTED_YAML_PATH),
        "--record-camera-video",
        "--list-variations",
    ])

    assert run_local_experiment(cfg) == 0


def test_record_viewport_video_does_not_change_existing_camera_startup_behavior(monkeypatch):
    class _SimulationAppContext:
        def __init__(self, launcher_args):
            assert launcher_args["enable_cameras"] is False

        def __enter__(self):
            pass

        def __exit__(self, _exception_type, _exception, _traceback):
            pass

    monkeypatch.setattr(eval_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(
        eval_runner,
        "load_arena_experiment_from_config_file",
        lambda *_args, **_kwargs: [SimpleNamespace(name="baseline", environment=SimpleNamespace())],
    )
    monkeypatch.setattr(eval_runner, "list_variations", lambda _experiment: None)

    cfg = parse_eval_runner_cfg([
        "--experiment-config",
        str(GETTING_STARTED_YAML_PATH),
        "--record-viewport-video",
        "--list-variations",
    ])

    assert run_local_experiment(cfg) == 0


def test_legacy_chunk_subprocess_replays_process_args_with_config_override(tmp_path, monkeypatch):
    submitted_command = None
    chunk_path = None

    def _run(command, check):
        nonlocal chunk_path, submitted_command
        submitted_command = command
        chunk_path = Path(command[-1])
        assert check is False
        assert chunk_path.is_file()
        return SimpleNamespace(returncode=0)

    parent_path = tmp_path / "parent.json"
    parent_args = [
        "--experiment_config",
        str(parent_path),
        "--chunk_size",
        "1",
        "--serve_evaluation_report",
    ]
    monkeypatch.setattr(legacy_eval_runner.subprocess, "run", _run)
    monkeypatch.setattr(legacy_eval_runner.sys, "argv", ["parent-entry-point", *parent_args])

    assert _run_legacy_json_chunk("chunk 1/1", [{"policy_type": "zero_action"}]) == 0
    assert submitted_command is not None
    assert submitted_command[1].endswith("isaaclab_arena/evaluation/eval_runner.py")
    assert "parent-entry-point" not in submitted_command
    assert "--serve_evaluation_report" not in submitted_command
    assert str(parent_path) in submitted_command
    assert chunk_path is not None
    assert submitted_command[-2:] == ["--experiment-config", str(chunk_path)]
    assert parse_eval_runner_cfg(submitted_command[2:]).experiment_config == chunk_path
    assert not chunk_path.exists()


def test_app_launcher_resolved_device_propagates_through_typed_args(monkeypatch):
    launcher_args = {"device": "cuda:0", "device_explicit": False, "enable_cameras": True}

    class _AppLauncher:
        def __init__(self, resolved_args):
            resolved_args.pop("device_explicit")
            resolved_args["device"] = "cpu"
            self.app = SimpleNamespace()

    monkeypatch.setattr(simulation_app, "AppLauncher", _AppLauncher)

    simulation_app.get_app_launcher(launcher_args)

    assert launcher_args == {"device": "cpu", "enable_cameras": True}


def test_unknown_experiment_config_file_type_is_rejected(tmp_path):
    config_path = tmp_path / "experiment.toml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="must use .json, .yaml, or .yml"):
        validate_experiment_config_path(config_path)
