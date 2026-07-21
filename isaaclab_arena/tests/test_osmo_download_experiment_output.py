# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify downloading complete Arena Experiment outputs from OSMO object storage."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from osmo.scripts.download_experiment_output import download_experiment_output, main
from osmo.workflows.workflow_constants import DATASETS_SWIFT_URL


def test_downloads_experiment_output_to_default_directory(monkeypatch):
    captured_download = None

    def capture_download(workflow_id, output_directory, remote_base_uri):
        nonlocal captured_download
        captured_download = (workflow_id, output_directory, remote_base_uri)
        return 0

    monkeypatch.setattr("osmo.scripts.download_experiment_output.download_experiment_output", capture_download)

    return_code = main(["arena-experiment-123"])

    assert return_code == 0
    assert captured_download == (
        "arena-experiment-123",
        Path("/eval/arena-experiment-123"),
        DATASETS_SWIFT_URL,
    )


def test_cli_help_states_default_output_directory():
    result = subprocess.run(
        [sys.executable, "-m", "osmo.scripts.download_experiment_output", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0
    assert "/eval/<workflow-id>" in result.stdout


def test_downloads_from_explicit_remote_and_output_bases_without_shell_splitting(monkeypatch, tmp_path):
    output_base_directory = tmp_path / "experiment output"
    expected_output_directory = output_base_directory / "arena-experiment-123"
    remote_base_uri = "s3://my-bucket/experiment-outputs/"
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
        "--remote-base-uri",
        remote_base_uri,
    ])

    assert return_code == 0
    assert captured_command == [
        "osmo",
        "data",
        "download",
        "s3://my-bucket/experiment-outputs/arena-experiment-123",
        str(expected_output_directory),
    ]
    assert expected_output_directory.is_dir()


@pytest.mark.parametrize("workflow_id", ["", ".", "..", "workflow/name", "workflow name"])
def test_cli_rejects_invalid_workflow_id(workflow_id):
    result = subprocess.run(
        [sys.executable, "-m", "osmo.scripts.download_experiment_output", workflow_id],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 2
    assert "invalid OSMO workflow ID" in result.stderr


@pytest.mark.parametrize("workflow_id", ["", ".", "..", "workflow/name", "workflow name"])
def test_download_rejects_invalid_workflow_id(monkeypatch, tmp_path, workflow_id):
    def fail_if_called(command):
        pytest.fail(f"Unexpected download command: {command}")

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", fail_if_called)

    with pytest.raises(AssertionError, match="Invalid OSMO workflow ID"):
        download_experiment_output(workflow_id, tmp_path / "output", DATASETS_SWIFT_URL)


def test_rejects_nonempty_output_directory_before_download(monkeypatch, tmp_path):
    output_directory = tmp_path / "existing-output"
    output_directory.mkdir()
    (output_directory / "stale-results.jsonl").write_text("stale", encoding="utf-8")

    def fail_if_called(command):
        pytest.fail(f"Unexpected download command: {command}")

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", fail_if_called)

    with pytest.raises(AssertionError, match="Experiment output directory must be empty"):
        download_experiment_output("arena-experiment-123", output_directory, DATASETS_SWIFT_URL)


def test_propagates_osmo_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "osmo.scripts.download_experiment_output.subprocess.run",
        lambda command: SimpleNamespace(returncode=23),
    )

    return_code = download_experiment_output("arena-experiment-123", tmp_path / "output", DATASETS_SWIFT_URL)

    assert return_code == 23
