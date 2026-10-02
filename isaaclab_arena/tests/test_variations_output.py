# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep discovery artifacts parseable despite simulator console output."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.variations.catalogue_output import emit_variations_catalogue


@pytest.mark.parametrize("runner_name", ("policy_runner", "experiment_runner"))
@pytest.mark.parametrize("output_format", ("text", "json"))
def test_runner_writes_catalogue_separately_from_simulation_logs(
    runner_name, output_format, monkeypatch, tmp_path, capsys
):
    from isaaclab_arena.evaluation import experiment_runner, policy_runner, run_execution

    runner = {"policy_runner": policy_runner, "experiment_runner": experiment_runner}[runner_name]
    simulation_is_running = False

    class SimulationContext:
        def __init__(self, _args):
            pass

        def __enter__(self):
            nonlocal simulation_is_running
            simulation_is_running = True
            print("Simulation startup log")

        def __exit__(self, *_args):
            nonlocal simulation_is_running
            simulation_is_running = False
            print("Simulation shutdown log")

    catalogue = {
        "schema_version": 1,
        "variations": [{"enable_path": "part.mass.enabled", "supported": False, "restriction_reason": "fixture"}],
    }

    def build(*_args, **_kwargs):
        assert simulation_is_running
        print("Asset construction log")
        return SimpleNamespace(
            get_variations_catalogue_as_dict=lambda: catalogue,
            get_variations_catalogue_as_string=lambda: "Variation text\n",
        )

    monkeypatch.setattr(runner, "SimulationAppContext", SimulationContext)
    output_path = tmp_path / "new" / "catalogue.txt"
    argv = [
        runner_name,
        "--list_variations",
        "--variations_format",
        output_format,
        "--variations_output",
        str(output_path),
    ]
    if runner_name == "policy_runner":
        monkeypatch.setattr(policy_runner, "get_isaaclab_arena_environments_cli_parser", lambda parser: parser)
        monkeypatch.setattr(policy_runner, "get_arena_builder_from_cli", build)
        expected_json = catalogue
        expected_text = "Variation text\n\n"
    else:
        experiment_path = tmp_path / "experiment.yaml"
        experiment_path.write_text("{}", encoding="utf-8")
        argv.extend(["--experiment_config", str(experiment_path)])
        run = SimpleNamespace(name="baseline", environment=SimpleNamespace(enable_cameras=False))
        monkeypatch.setattr(experiment_runner, "load_legacy_json_experiment_config", lambda *_args: None)
        monkeypatch.setattr(experiment_runner, "_experiment_requires_cameras", lambda *_args: False)
        monkeypatch.setattr(
            experiment_runner,
            "load_arena_experiment_from_config_file",
            lambda *_args, **_kwargs: SimpleNamespace(runs={"baseline": run}),
        )
        monkeypatch.setattr(run_execution, "build_arena_builder_from_run_cfg", build)
        expected_json = {"schema_version": 1, "runs": {"baseline": catalogue}}
        expected_text = "=== Variations for run 'baseline' ===\nVariation text\n\n"
    monkeypatch.setattr("sys.argv", argv)
    runner.main()

    artifact = output_path.read_text(encoding="utf-8")
    if output_format == "json":
        assert json.loads(artifact) == expected_json
    else:
        assert artifact == expected_text
    console = capsys.readouterr().out
    assert artifact in console
    for message in ("Simulation startup log", "Asset construction log", "Simulation shutdown log"):
        assert message in console
        assert message not in artifact


@pytest.mark.parametrize("runner_name", ("policy_runner", "experiment_runner"))
def test_output_without_discovery_is_rejected_before_simulation(runner_name, tmp_path):
    # Invalid argparse paths must stay in a child process, even if the suite has started Kit.
    script = f"""
from isaaclab_arena.evaluation import {runner_name} as runner
def unexpected_simulation(*args, **kwargs):
    raise RuntimeError("Simulation must not start")
runner.SimulationAppContext = unexpected_simulation
runner.main()
"""
    output_path = tmp_path / "catalogue.json"
    result = subprocess.run(
        [TestConstants.python_path, "-c", script, "--variations_output", str(output_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The Isaac Sim shell wrapper normalizes the parser's exit status to 1.
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--variations_output requires --list_variations" in result.stderr
    assert "Simulation must not start" not in result.stderr
    assert not output_path.exists()


def test_output_replaces_existing_file_and_preserves_stdout_default(tmp_path, capsys):
    output_path = tmp_path / "catalogue.json"
    output_path.write_text("obsolete content", encoding="utf-8")
    emit_variations_catalogue({"schema_version": 1, "variations": []}, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"schema_version": 1, "variations": []}
    capsys.readouterr()
    emit_variations_catalogue("Existing text output\n")
    assert capsys.readouterr().out == "Existing text output\n\n"
