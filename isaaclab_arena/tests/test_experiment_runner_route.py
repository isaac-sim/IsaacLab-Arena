# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify routing between typed local evaluation, OSMO, and the legacy CLI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation import experiment_runner
from isaaclab_arena.evaluation.experiment_runner_cfg import ExperimentRunnerCfg
from isaaclab_arena.evaluation.experiment_runner_route import parse_experiment_runner_route, uses_experiment_runner_cfg


def test_parse_experiment_runner_route_leaves_backend_arguments():
    route, backend_args = parse_experiment_runner_route([
        "--config",
        "evaluation.yaml",
        "--local",
        "--viz",
        "none",
        "runs.demo.rollout_limit.num_steps=4",
    ])

    assert route.config_path == "evaluation.yaml"
    assert route.local
    assert route.osmo_config_path is None
    assert backend_args == ["--viz", "none", "runs.demo.rollout_limit.num_steps=4"]
    assert uses_experiment_runner_cfg(["--config=evaluation.yaml"])


def test_parse_experiment_runner_route_rejects_osmo_config_for_local_execution():
    with pytest.raises(SystemExit):
        parse_experiment_runner_route([
            "--config",
            "evaluation.yaml",
            "--local",
            "--osmo-config",
            "osmo.yaml",
        ])


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--help"],
        ["--config", "evaluation.yaml", "--help"],
        ["--local", "--help"],
        ["--osmo-config", "osmo.yaml", "--help"],
    ],
)
def test_typed_experiment_runner_help_exposes_primary_interface(cli_args, capsys, monkeypatch):
    monkeypatch.setattr(
        experiment_runner,
        "_load_experiment_runner_cfg",
        lambda _path: pytest.fail("help must not load a configuration"),
    )

    assert experiment_runner.main(cli_args) == 0

    help_text = capsys.readouterr().out
    assert "--config PATH" in help_text
    assert "--local" in help_text
    assert "--osmo-config PATH" in help_text
    assert "isaaclab_arena_environments/evaluation_configs/openpi.yaml" in help_text
    assert "osmo/config/arena_experiment_workflow.yaml" in help_text
    assert "deprecated legacy interface" in help_text


def test_main_without_config_preserves_legacy_entry_path(monkeypatch):
    received_args = None

    def run_legacy(cli_args):
        nonlocal received_args
        received_args = cli_args
        return 17

    monkeypatch.setattr(experiment_runner, "_run_legacy_experiment_runner", run_legacy)
    monkeypatch.setattr(
        experiment_runner,
        "_run_experiment_runner_cfg_route",
        lambda _args: pytest.fail("typed route must not receive a legacy invocation"),
    )

    assert experiment_runner.main(["--experiment_config", "legacy.json"]) == 17
    assert received_args == ["--experiment_config", "legacy.json"]


def test_typed_local_route_combines_yaml_and_trailing_overrides(monkeypatch):
    cfg = ExperimentRunnerCfg(
        experiment_config="experiment.yaml",
        experiment_overrides=["runs.demo.rollout_limit.num_steps=2"],
    )
    app_launcher_args = SimpleNamespace(device="cuda:0", enable_cameras=False)
    received = None

    monkeypatch.setattr(experiment_runner, "_load_experiment_runner_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        experiment_runner,
        "_parse_local_app_launcher_args",
        lambda backend_args: (
            app_launcher_args,
            ["runs.demo.rollout_limit.num_steps=4"],
        ),
    )

    def run_locally(effective_cfg, received_app_launcher_args):
        nonlocal received
        received = (effective_cfg, received_app_launcher_args)
        return 0

    monkeypatch.setattr(experiment_runner, "_run_experiment_runner_cfg_locally", run_locally)
    monkeypatch.setattr(
        "osmo.experiment_dispatcher.submit_experiment_to_osmo",
        lambda *_args: pytest.fail("local invocation must not submit an OSMO workflow"),
    )

    result = experiment_runner.main([
        "--config",
        "evaluation.yaml",
        "--local",
        "--device",
        "cuda:0",
        "runs.demo.rollout_limit.num_steps=4",
    ])

    assert result == 0
    assert received is not None
    effective_cfg, received_app_launcher_args = received
    assert effective_cfg.experiment_overrides == [
        "runs.demo.rollout_limit.num_steps=2",
        "runs.demo.rollout_limit.num_steps=4",
    ]
    assert received_app_launcher_args is app_launcher_args


def test_typed_remote_route_submits_without_entering_local_runtime(monkeypatch):
    cfg = ExperimentRunnerCfg(experiment_config="experiment.yaml")
    received = None

    monkeypatch.setattr(experiment_runner, "_load_experiment_runner_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        experiment_runner,
        "_parse_local_app_launcher_args",
        lambda _args: pytest.fail("remote submission must not parse AppLauncher arguments"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "_run_experiment_runner_cfg_locally",
        lambda *_args: pytest.fail("remote submission must not start local evaluation"),
    )

    def submit(effective_cfg, osmo_config_path):
        nonlocal received
        received = (effective_cfg, osmo_config_path)
        return 23

    monkeypatch.setattr("osmo.experiment_dispatcher.submit_experiment_to_osmo", submit)

    result = experiment_runner.main([
        "--config",
        "evaluation.yaml",
        "--osmo-config",
        "osmo.yaml",
        "runs.remote.rollout_limit.num_episodes=3",
    ])

    assert result == 23
    assert received is not None
    effective_cfg, osmo_config_path = received
    assert effective_cfg.experiment_overrides == ["runs.remote.rollout_limit.num_episodes=3"]
    assert osmo_config_path == "osmo.yaml"


def test_local_typed_execution_applies_experiment_runner_settings(monkeypatch):
    cfg = ExperimentRunnerCfg(
        experiment_config="experiment.yaml",
        experiment_overrides=["runs.demo.rollout_limit.num_steps=4"],
        output_base_dir="/tmp/evaluation",
        record_viewport_video=True,
        record_camera_video=True,
        continue_on_error=True,
        serve_evaluation_report=True,
        evaluation_report_port=9000,
    )
    app_launcher_args = SimpleNamespace(device="cuda:1", enable_cameras=False, list_variations=False)
    load_call = None
    execution_call = None

    class SimulationContext:
        def __init__(self, received_args):
            assert received_args is app_launcher_args

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def load_experiment(path, *, device, overrides):
        nonlocal load_call
        load_call = (path, device, overrides)
        return []

    def execute(experiment, **kwargs):
        nonlocal execution_call
        execution_call = (experiment, kwargs)

    monkeypatch.setattr(experiment_runner, "_get_simulation_app_context_type", lambda: SimulationContext)
    monkeypatch.setattr(experiment_runner, "_get_experiment_loader", lambda: load_experiment)
    monkeypatch.setattr(experiment_runner, "_execute_experiment_and_report", execute)

    assert experiment_runner._run_experiment_runner_cfg_locally(cfg, app_launcher_args) == 0
    assert app_launcher_args.enable_cameras
    assert load_call == (
        "experiment.yaml",
        "cuda:1",
        ["runs.demo.rollout_limit.num_steps=4"],
    )
    assert execution_call == (
        [],
        {
            "output_base_dir": "/tmp/evaluation",
            "record_viewport_video": True,
            "record_camera_video": True,
            "continue_on_error": True,
            "serve_evaluation_report": True,
            "evaluation_report_port": 9000,
        },
    )
