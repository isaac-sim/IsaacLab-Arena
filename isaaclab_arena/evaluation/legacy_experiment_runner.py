# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy JSON preflight and subprocess dispatch for the Experiment Runner."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_CHUNK_PARENT_EXPERIMENT_OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE = (
    "ISAACLAB_ARENA_CHUNK_PARENT_EXPERIMENT_OUTPUT_DIRECTORY"
)


def get_inherited_chunk_experiment_output_directory() -> Path | None:
    """Return the output directory selected by a chunk parent, when one exists.

    Returns:
        The inherited directory in a chunk child, or None in the top-level process.
    """
    inherited_experiment_output_directory = os.environ.get(
        _CHUNK_PARENT_EXPERIMENT_OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE
    )
    if inherited_experiment_output_directory is None:
        return None
    return Path(inherited_experiment_output_directory)


def load_legacy_json_experiment_config(
    experiment_config_path: Path,
    experiment_overrides: list[str],
) -> dict | None:
    """Load legacy JSON data needed before startup, or return None for YAML.

    Args:
        experiment_config_path: Validated path to an Experiment configuration.
        experiment_overrides: Hydra overrides supplied for a typed YAML Experiment.

    Returns:
        The legacy JSON mapping, or None when the Experiment uses YAML.
    """
    if experiment_config_path.suffix.lower() != ".json":
        return None

    assert not experiment_overrides, "Experiment overrides are supported only for typed YAML Experiments"
    with experiment_config_path.open(encoding="utf-8") as experiment_config_file:
        return json.load(experiment_config_file)


def legacy_json_experiment_requires_cameras(experiment_config: dict) -> bool:
    """Return whether a legacy JSON Experiment requires camera support.

    Args:
        experiment_config: Legacy Experiment mapping containing entries below jobs.

    Returns:
        Whether any legacy entry enables environment cameras.
    """
    return any(
        job_config.get("arena_env_args", {}).get("enable_cameras", False) for job_config in experiment_config["jobs"]
    )


def _run_legacy_json_chunk(
    chunk_label: str,
    legacy_job_configs: list[dict],
    experiment_output_directory: Path,
) -> int:
    """Run legacy JSON entries in a fresh subprocess using the shared Experiment directory.

    Args:
        chunk_label: Human-readable position of this chunk in the Experiment.
        legacy_job_configs: Legacy Run configurations assigned to this child process.
        experiment_output_directory: Timestamped Experiment directory shared by every chunk.

    Returns:
        The child process return code.
    """
    print(f"[experiment_runner] {chunk_label}", flush=True)
    # Serialize this chunk's jobs to a temp config the child loads via --experiment_config.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temporary_chunk_config_file:
        json.dump({"jobs": legacy_job_configs}, temporary_chunk_config_file)
        chunk_config_path = Path(temporary_chunk_config_file.name)
    # Re-run this invocation in the child, with --experiment_config appended so it wins over
    # the master config (argparse keeps the last value).
    # Strip --serve_evaluation_report: a child that served its report would block on
    # serve_until_ctrl_c forever.
    forwarded_arguments = [argument for argument in sys.argv if argument != "--serve_evaluation_report"]
    experiment_config_override = ["--experiment_config", str(chunk_config_path)]
    child_command = [
        sys.executable,
        *forwarded_arguments,
        *experiment_config_override,
    ]
    child_environment = os.environ.copy()
    child_environment[_CHUNK_PARENT_EXPERIMENT_OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE] = str(experiment_output_directory)
    try:
        completed_child_process = subprocess.run(child_command, check=False, env=child_environment)
    finally:
        # Remove the temp chunk config now that the child has loaded it.
        chunk_config_path.unlink(missing_ok=True)
    return completed_child_process.returncode


def run_legacy_json_in_chunks(
    legacy_experiment_config: dict,
    *,
    chunk_size: int,
    experiment_output_directory: Path,
    serve_evaluation_report: bool,
) -> None:
    """Run every chunk sequentially in fresh processes sharing one Experiment directory.

    Args:
        legacy_experiment_config: Complete legacy Experiment mapping containing all Runs.
        chunk_size: Maximum number of Runs assigned to one child process.
        experiment_output_directory: Timestamped Experiment directory shared by every chunk.
        serve_evaluation_report: Whether report serving was requested by the parent.
    """
    # TODO(cvolk): Aggregate per-chunk metrics into one centralized view. Each chunk
    # subprocess currently prints its own MetricsLogger summary and nothing is merged
    # or persisted (save_metrics_to_file() is unused). Have each chunk write metrics
    # JSON to a temp file, then merge and print or save them here.
    legacy_job_configs = legacy_experiment_config["jobs"]
    assert chunk_size > 0, f"--chunk_size must be positive, got {chunk_size}"
    legacy_run_names = [legacy_job_config["name"] for legacy_job_config in legacy_job_configs]
    assert len(legacy_run_names) == len(set(legacy_run_names)), "run names must be unique"
    number_of_chunks = math.ceil(len(legacy_job_configs) / chunk_size)
    print(
        f"[experiment_runner] {len(legacy_job_configs)} Runs → {number_of_chunks} chunks of <= {chunk_size}",
        flush=True,
    )

    if serve_evaluation_report:
        print("--serve_evaluation_report is ignored with --chunk_size.", flush=True)

    for chunk_index in range(number_of_chunks):
        first_run_index = chunk_index * chunk_size
        end_run_index_exclusive = min(first_run_index + chunk_size, len(legacy_job_configs))
        chunk_label = (
            f"chunk {chunk_index + 1}/{number_of_chunks}: Runs {first_run_index}..{end_run_index_exclusive - 1}"
        )
        child_return_code = _run_legacy_json_chunk(
            chunk_label,
            legacy_job_configs[first_run_index:end_run_index_exclusive],
            experiment_output_directory,
        )
        if child_return_code != 0:
            print(
                f"[experiment_runner] chunk {chunk_index + 1} failed (exit {child_return_code}).",
                flush=True,
            )
            sys.exit(child_return_code)
