# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest

from isaaclab_arena.evaluation import legacy_experiment_runner


def test_chunk_subprocess_uses_shared_experiment_output_directory(monkeypatch, tmp_path):
    shared_experiment_output_directory = tmp_path / "shared-experiment-output"
    legacy_job_configs = [{"name": "first"}]
    captured_child_command: list[str] | None = None
    captured_chunk_config: dict | None = None

    def capture_child_process(child_command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        nonlocal captured_child_command, captured_chunk_config
        captured_child_command = child_command
        chunk_config_path = Path(child_command[-3])
        captured_chunk_config = json.loads(chunk_config_path.read_text(encoding="utf-8"))
        assert not check
        return subprocess.CompletedProcess(child_command, returncode=0)

    monkeypatch.setattr(
        legacy_experiment_runner.sys,
        "argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            "complete-experiment.json",
            "--chunk_size",
            "1",
            "--serve_evaluation_report",
        ],
    )
    monkeypatch.setattr(legacy_experiment_runner.subprocess, "run", capture_child_process)

    return_code = legacy_experiment_runner._run_legacy_json_chunk(
        "chunk 1/1: Runs 0..0",
        legacy_job_configs,
        shared_experiment_output_directory,
    )

    assert return_code == 0
    assert captured_chunk_config == {"jobs": legacy_job_configs}
    assert captured_child_command is not None
    assert "--serve_evaluation_report" not in captured_child_command
    assert captured_child_command[-4] == "--experiment_config"
    assert captured_child_command[-2:] == [
        "--experiment_output_directory",
        str(shared_experiment_output_directory),
    ]
    assert not Path(captured_child_command[-3]).exists()


def test_all_chunks_receive_same_experiment_output_directory(monkeypatch, tmp_path):
    shared_experiment_output_directory = tmp_path / "shared-experiment-output"
    legacy_experiment_config = {"jobs": [{"name": f"run-{run_index}"} for run_index in range(5)]}
    received_chunks: list[tuple[str, list[dict], Path]] = []

    def capture_chunk(
        chunk_label: str,
        legacy_job_configs: list[dict],
        experiment_output_directory: Path,
    ) -> int:
        received_chunks.append((chunk_label, legacy_job_configs, experiment_output_directory))
        return 0

    monkeypatch.setattr(legacy_experiment_runner, "_run_legacy_json_chunk", capture_chunk)

    legacy_experiment_runner.run_legacy_json_in_chunks(
        legacy_experiment_config,
        chunk_size=2,
        experiment_output_directory=shared_experiment_output_directory,
        serve_evaluation_report=False,
    )

    assert [len(legacy_job_configs) for _, legacy_job_configs, _ in received_chunks] == [2, 2, 1]
    assert [output_directory for _, _, output_directory in received_chunks] == [
        shared_experiment_output_directory,
        shared_experiment_output_directory,
        shared_experiment_output_directory,
    ]


def test_chunking_rejects_duplicate_run_names_before_starting_children(monkeypatch, tmp_path):
    def fail_if_chunk_starts(*_args, **_kwargs) -> int:
        pytest.fail("A child process must not start when Run names are not unique")

    monkeypatch.setattr(legacy_experiment_runner, "_run_legacy_json_chunk", fail_if_chunk_starts)

    with pytest.raises(AssertionError, match="run names must be unique"):
        legacy_experiment_runner.run_legacy_json_in_chunks(
            {"jobs": [{"name": "duplicate"}, {"name": "duplicate"}]},
            chunk_size=1,
            experiment_output_directory=tmp_path / "shared-experiment-output",
            serve_evaluation_report=False,
        )
