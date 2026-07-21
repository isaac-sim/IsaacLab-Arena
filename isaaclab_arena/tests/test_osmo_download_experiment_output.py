# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify downloading complete Arena Experiment outputs from OSMO object storage."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from osmo.scripts.download_experiment_output import download_experiment_output, main
from osmo.workflows.workflow_constants import DATASETS_SWIFT_URL


def test_downloads_experiment_output_to_default_directory(monkeypatch):
    captured_download = None

    def capture_download(workflow_id, output_directory):
        nonlocal captured_download
        captured_download = (workflow_id, output_directory)
        return 0

    monkeypatch.setattr("osmo.scripts.download_experiment_output.download_experiment_output", capture_download)

    return_code = main(["arena-experiment-123"])

    assert return_code == 0
    assert captured_download == ("arena-experiment-123", Path("/eval/arena-experiment-123"))


def test_help_states_default_output_directory(capsys):
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])

    assert help_exit.value.code == 0
    assert "/eval/<workflow-id>" in capsys.readouterr().out


def test_downloads_to_explicit_base_directory_without_shell_splitting(monkeypatch, tmp_path):
    output_base_directory = tmp_path / "experiment output"
    expected_output_directory = output_base_directory / "arena-experiment-123"
    captured_command = None

    def capture_download(command):
        nonlocal captured_command
        captured_command = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", capture_download)

    return_code = main([
        "arena-experiment-123",
        "--output-base-directory",
        str(output_base_directory),
    ])

    assert return_code == 0
    assert captured_command == [
        "osmo",
        "data",
        "download",
        f"{DATASETS_SWIFT_URL}/arena-experiment-123/",
        str(expected_output_directory),
    ]
    assert expected_output_directory.is_dir()


@pytest.mark.parametrize("workflow_id", ["", ".", "..", "workflow/name", "workflow name"])
def test_rejects_invalid_workflow_id_at_both_entry_points(monkeypatch, tmp_path, workflow_id):
    def fail_if_called(command):
        pytest.fail(f"Unexpected download command: {command}")

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", fail_if_called)

    with pytest.raises(SystemExit) as invalid_argument_exit:
        main([workflow_id])
    assert invalid_argument_exit.value.code == 2

    with pytest.raises(AssertionError, match="Invalid OSMO workflow ID"):
        download_experiment_output(workflow_id, tmp_path / "output")


def test_rejects_nonempty_output_directory_before_download(monkeypatch, tmp_path):
    output_directory = tmp_path / "existing-output"
    output_directory.mkdir()
    (output_directory / "stale-results.jsonl").write_text("stale", encoding="utf-8")

    def fail_if_called(command):
        pytest.fail(f"Unexpected download command: {command}")

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", fail_if_called)

    with pytest.raises(AssertionError, match="Experiment output directory must be empty"):
        download_experiment_output("arena-experiment-123", output_directory)


def test_propagates_osmo_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "osmo.scripts.download_experiment_output.subprocess.run",
        lambda command: SimpleNamespace(returncode=23),
    )

    return_code = download_experiment_output("arena-experiment-123", tmp_path / "output")

    assert return_code == 23
