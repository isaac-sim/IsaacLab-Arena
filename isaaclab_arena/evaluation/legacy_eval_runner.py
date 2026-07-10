# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy JSON preflight and subprocess dispatch for the evaluation runner."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab_arena.evaluation.eval_runner_cli import EvalRunnerCfg


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


def _without_experiment_config(arguments: list[str]) -> list[str]:
    """Remove every supported Experiment-path option and its value."""
    option_names = {
        "--experiment-config",
        "--experiment_config",
        "--eval-jobs-config",
        "--eval_jobs_config",
    }
    filtered: list[str] = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument in option_names:
            skip_next = True
            continue
        if any(argument.startswith(f"{name}=") for name in option_names):
            continue
        filtered.append(argument)
    return filtered


def _run_legacy_json_chunk(cfg: EvalRunnerCfg, chunk_label: str, legacy_job_configs: list[dict]) -> int:
    """Run legacy JSON entries in a fresh eval_runner subprocess."""
    print(f"[eval_runner] {chunk_label}", flush=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"jobs": legacy_job_configs}, tmp)
        chunk_path = Path(tmp.name)

    forwarded_args = _without_experiment_config(cfg.invocation_args)
    forwarded_args = [
        arg for arg in forwarded_args if arg not in {"--serve-evaluation-report", "--serve_evaluation_report"}
    ]
    eval_runner_path = Path(__file__).with_name("eval_runner.py")
    child_cmd = [
        sys.executable,
        str(eval_runner_path),
        *forwarded_args,
        "--experiment-config",
        str(chunk_path),
    ]
    try:
        result = subprocess.run(child_cmd, check=False)
    finally:
        chunk_path.unlink(missing_ok=True)
    return result.returncode


def run_legacy_json_in_chunks(cfg: EvalRunnerCfg, legacy_experiment_config: dict) -> None:
    """Run each chunk of a legacy JSON Experiment in a fresh subprocess."""
    # TODO(cvolk): Aggregate per-chunk metrics into one centralized view. Each chunk
    # subprocess currently prints its own MetricsLogger summary and nothing is merged
    # or persisted (save_metrics_to_file() is unused). Have each chunk write metrics
    # JSON to a temp file, then merge and print or save them here.
    jobs = legacy_experiment_config["jobs"]
    assert cfg.chunk_size is not None and cfg.chunk_size > 0, f"--chunk-size must be positive, got {cfg.chunk_size}"
    n_chunks = math.ceil(len(jobs) / cfg.chunk_size)
    print(f"[eval_runner] {len(jobs)} Runs → {n_chunks} chunks of <= {cfg.chunk_size}", flush=True)

    if cfg.serve_evaluation_report:
        print("--serve-evaluation-report is ignored with --chunk-size.", flush=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * cfg.chunk_size
        end = min(start + cfg.chunk_size, len(jobs))
        chunk_label = f"chunk {chunk_idx + 1}/{n_chunks}: Runs {start}..{end - 1}"
        returncode = _run_legacy_json_chunk(cfg, chunk_label, jobs[start:end])
        if returncode != 0:
            raise SystemExit(returncode)
