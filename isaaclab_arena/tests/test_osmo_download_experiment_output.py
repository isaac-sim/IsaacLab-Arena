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

from osmo.scripts.download_experiment_output import _parse_listed_output_paths, download_experiment_output, main
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
    remote_uri = "s3://my-bucket/experiment-outputs/arena-experiment-123"
    list_outputs = iter([
        "experiment-outputs/arena-experiment-123/run one/episode results.jsonl\n\nTotal 1 objects found",
        "experiment-outputs/arena-experiment-123/index.html\n\nTotal 1 objects found",
        "experiment-outputs/arena-experiment-123/index.html\n\nTotal 1 objects found",
    ])
    captured_commands = []

    def capture_command(command, **kwargs):
        captured_commands.append(command)
        if command[2] == "list":
            assert kwargs == {"capture_output": True, "text": True}
            listed_objects, reported_total = next(list_outputs).rsplit("\n\n", maxsplit=1)
            Path(command[5]).write_text(listed_objects, encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=reported_total, stderr="")
        assert kwargs == {}
        if command[3] == remote_uri:
            (expected_output_directory / "index.html").write_text("report", encoding="utf-8")
        else:
            local_directory = Path(command[4])
            (local_directory / "episode results.jsonl").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", capture_command)

    return_code = main([
        "arena-experiment-123",
        "--output-base-directory",
        str(output_base_directory),
        "--remote-base-uri",
        remote_base_uri,
    ])

    assert return_code == 0
    assert len(captured_commands) == 5
    for list_command in captured_commands[:3]:
        assert list_command[:5] == ["osmo", "data", "list", "--recursive", remote_uri]
        assert Path(list_command[5]).name.startswith("objects-")
    assert captured_commands[3:] == [
        ["osmo", "data", "download", remote_uri, str(expected_output_directory)],
        [
            "osmo",
            "data",
            "download",
            f"{remote_uri}/run one/episode results.jsonl",
            str(expected_output_directory / "run one"),
        ],
    ]
    assert sorted(
        path.relative_to(expected_output_directory).as_posix() for path in expected_output_directory.rglob("*")
    ) == [
        "index.html",
        "run one",
        "run one/episode results.jsonl",
    ]


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


def test_propagates_osmo_list_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "osmo.scripts.download_experiment_output.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=23, stdout="", stderr="listing failed\n"),
    )

    return_code = download_experiment_output("arena-experiment-123", tmp_path / "output", DATASETS_SWIFT_URL)

    assert return_code == 23


def test_propagates_osmo_download_failure(monkeypatch, tmp_path):
    def fail_bulk_download(command, **kwargs):
        if command[2] == "list":
            Path(command[5]).write_text("workflows/arena-experiment-123/index.html\n", encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout="Total 1 object found\n",
                stderr="",
            )
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", fail_bulk_download)

    return_code = download_experiment_output("arena-experiment-123", tmp_path / "output", DATASETS_SWIFT_URL)

    assert return_code == 23


def test_retries_and_rejects_listing_without_report(monkeypatch, tmp_path, capsys):
    captured_commands = []

    def list_without_report(command, **kwargs):
        captured_commands.append(command)
        Path(command[5]).write_text("workflows/arena-experiment-123/run/episode.jsonl\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout="Total 1 object found\n",
            stderr="",
        )

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", list_without_report)

    return_code = download_experiment_output("arena-experiment-123", tmp_path / "output", DATASETS_SWIFT_URL)

    assert return_code == 1
    assert len(captured_commands) == 3
    assert "index.html is missing" in capsys.readouterr().err


def test_rejects_successful_exact_download_that_does_not_create_file(monkeypatch, tmp_path, capsys):
    output_directory = tmp_path / "output"

    def omit_listed_object(command, **kwargs):
        if command[2] == "list":
            Path(command[5]).write_text(
                "\n".join([
                    "workflows/arena-experiment-123/index.html",
                    "workflows/arena-experiment-123/run/episode.jsonl",
                ]),
                encoding="utf-8",
            )
            return SimpleNamespace(
                returncode=0,
                stdout="Total 2 objects found\n",
                stderr="",
            )
        if command[3].endswith("arena-experiment-123"):
            (output_directory / "index.html").write_text("report", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", omit_listed_object)

    return_code = download_experiment_output("arena-experiment-123", output_directory, DATASETS_SWIFT_URL)

    assert return_code == 1
    assert "reported success without downloading 'run/episode.jsonl'" in capsys.readouterr().err


def test_propagates_exact_object_download_failure(monkeypatch, tmp_path):
    output_directory = tmp_path / "output"

    def fail_exact_download(command, **kwargs):
        if command[2] == "list":
            Path(command[5]).write_text(
                "\n".join([
                    "workflows/arena-experiment-123/index.html",
                    "workflows/arena-experiment-123/run/episode.jsonl",
                ]),
                encoding="utf-8",
            )
            return SimpleNamespace(
                returncode=0,
                stdout="Total 2 objects found\n",
                stderr="",
            )
        if command[3].endswith("arena-experiment-123"):
            (output_directory / "index.html").write_text("report", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=24)

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", fail_exact_download)

    return_code = download_experiment_output("arena-experiment-123", output_directory, DATASETS_SWIFT_URL)

    assert return_code == 24


def test_rejects_unexpected_downloaded_file(monkeypatch, tmp_path, capsys):
    output_directory = tmp_path / "output"

    def download_with_unexpected_file(command, **kwargs):
        if command[2] == "list":
            Path(command[5]).write_text("workflows/arena-experiment-123/index.html\n", encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout="Total 1 object found\n",
                stderr="",
            )
        (output_directory / "index.html").write_text("report", encoding="utf-8")
        (output_directory / "unexpected.jsonl").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osmo.scripts.download_experiment_output.subprocess.run", download_with_unexpected_file)

    return_code = download_experiment_output("arena-experiment-123", output_directory, DATASETS_SWIFT_URL)

    assert return_code == 1
    assert "contains unexpected files: unexpected.jsonl" in capsys.readouterr().err


@pytest.mark.parametrize(
    "listed_path",
    [
        "/workflows/arena-experiment-123/index.html",
        "workflows/arena-experiment-123/../index.html",
        "workflows/arena-experiment-123//index.html",
        "workflows/./arena-experiment-123/index.html",
    ],
)
def test_rejects_unsafe_listed_object_path(listed_path):
    list_output = f"{listed_path}\n\nTotal 1 object found\n"

    with pytest.raises(ValueError, match="Unsafe remote object path"):
        _parse_listed_output_paths(
            list_output,
            "swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/arena-experiment-123",
        )


def test_rejects_list_output_whose_total_does_not_match_object_keys():
    list_output = "workflows/arena-experiment-123/index.html\n\nTotal 2 objects found\n"

    with pytest.raises(ValueError, match="listed 1 object keys but reported 2"):
        _parse_listed_output_paths(
            list_output,
            "swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/arena-experiment-123",
        )
