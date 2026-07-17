# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy JSON preflight and subprocess dispatch for the Experiment Runner."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


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


def _run_legacy_json_chunk(chunk_label: str, legacy_job_configs: list[dict]) -> int:
    """Run legacy JSON entries in a fresh experiment_runner subprocess."""
    print(f"[experiment_runner] {chunk_label}", flush=True)
    # Serialize this chunk's jobs to a temp config the child loads via --experiment_config.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"jobs": legacy_job_configs}, tmp)
        chunk_path = Path(tmp.name)
    # Re-run this invocation in the child, with --experiment_config appended so it wins over
    # the master config (argparse keeps the last value).
    # Strip --serve_evaluation_report: a child that served its report would block on
    # serve_until_ctrl_c forever.
    forwarded_args = [arg for arg in sys.argv if arg != "--serve_evaluation_report"]
    config_override = ["--experiment_config", str(chunk_path)]
    child_cmd = [sys.executable, *forwarded_args, *config_override]
    try:
        result = subprocess.run(child_cmd, check=False)
    finally:
        # Remove the temp chunk config now that the child has loaded it.
        chunk_path.unlink(missing_ok=True)
    return result.returncode


def run_legacy_json_in_chunks(args_cli: argparse.Namespace, legacy_experiment_config: dict) -> None:
    """Run each chunk of a legacy JSON Experiment in a fresh subprocess."""
    # TODO(cvolk): Aggregate per-chunk metrics into one centralized view. Each chunk
    # subprocess currently prints its own MetricsLogger summary and nothing is merged
    # or persisted (save_metrics_to_file() is unused). Have each chunk write metrics
    # JSON to a temp file, then merge and print or save them here.
    jobs = legacy_experiment_config["jobs"]
    chunk_size = args_cli.chunk_size
    assert chunk_size > 0, f"--chunk_size must be positive, got {chunk_size}"
    n_chunks = math.ceil(len(jobs) / chunk_size)
    print(f"[experiment_runner] {len(jobs)} Runs → {n_chunks} chunks of <= {chunk_size}", flush=True)

    if args_cli.serve_evaluation_report:
        print("--serve_evaluation_report is ignored with --chunk_size.", flush=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(jobs))
        chunk_label = f"chunk {chunk_idx + 1}/{n_chunks}: Runs {start}..{end - 1}"
        returncode = _run_legacy_json_chunk(chunk_label, jobs[start:end])
        if returncode != 0:
            print(f"[experiment_runner] chunk {chunk_idx} failed (exit {returncode}).", flush=True)
            sys.exit(returncode)
