# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VLM usage KPI exporter in ``isaaclab_arena_benchmarking/benchmarks/vlm_usage``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME,
)
from isaaclab_arena_benchmarking.benchmarks.vlm_usage.vlm_usage_exporter import VLMUsageKpiExporter

START_TIMESTAMP_NS = 1792799000000000000
"""Before every Run end timestamp, in Unix epoch nanoseconds."""

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}

EXPERIMENT_NAME = "vlm_chunk_policy_benchmark_experiment"


def _usage(num_requests: int, prompt_tokens: int, completion_tokens: int, **overrides) -> dict:
    usage = {
        "endpoint": "internal",
        "model": "test-model",
        "num_requests": num_requests,
        "num_responses_without_usage": 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        # Every request takes one second unless overridden.
        "request_latency_total_ms": 1000.0 * num_requests,
        "request_latency_max_ms": 1000.0,
    }
    usage.update(overrides)
    return usage


def _write_experiment(output_dir: Path, runs: dict[str, dict]) -> None:
    """Write a result file and each Run's inference usage.

    Each entry of ``runs`` sets ``num_episodes_per_rebuild`` and may set ``usage_per_rebuild``, the inference
    usage written for each rebuild.
    """
    result_runs = {}
    for run_name, run in runs.items():
        rebuilds = []
        for rebuild_index, num_episodes in enumerate(run["num_episodes_per_rebuild"]):
            episodes = [{"env_id": 0, "episode_in_env": episode, "success": True} for episode in range(num_episodes)]
            rebuilds.append({"index": rebuild_index, "episodes": episodes})
        if "usage_per_rebuild" in run:
            usage_path = output_dir / run_name / ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            usage_path.write_text(json.dumps(run["usage_per_rebuild"]), encoding="utf-8")
        result_runs[run_name] = {
            "environment": {"name": f"{run_name}_env", "definition": "unused"},
            "policy_variant": "vlm_chunk",
            "status": "completed",
            "start_timestamp_ns": START_TIMESTAMP_NS,
            "end_timestamp_ns": END_TIMESTAMP_NS,
            "rebuilds": rebuilds,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(
        json.dumps({"git_stamp": GIT_STAMP, "experiment_name": EXPERIMENT_NAME, "runs": result_runs}), encoding="utf-8"
    )


def test_vlm_usage_sums_runs_that_used_a_vlm(tmp_path: Path):
    _write_experiment(
        tmp_path,
        {
            "vlm_a": {
                "num_episodes_per_rebuild": [2, 1],
                "usage_per_rebuild": [
                    _usage(10, 4000, 300),
                    _usage(5, 2000, 150, request_latency_total_ms=5500.0, request_latency_max_ms=1500.0),
                ],
            },
            "vlm_b": {"num_episodes_per_rebuild": [1], "usage_per_rebuild": [_usage(5, 2000, 150)]},
            # A Run whose policy wrote no inference usage adds neither requests nor episodes.
            "zero_action": {"num_episodes_per_rebuild": [6]},
        },
    )

    records_path = VLMUsageKpiExporter().export_to_json(tmp_path)

    assert records_path == tmp_path / "kpis_vlm_usage.json"
    assert json.loads(records_path.read_text(encoding="utf-8")) == [{
        "type": "vlm_usage",
        "timestamp_ns": END_TIMESTAMP_NS,
        **GIT_STAMP,
        "benchmark_id": EXPERIMENT_NAME,
        "endpoint": "internal",
        "model": "test-model",
        "num_requests": 20,
        "num_episodes": 4,
        "requests_per_episode": 5.0,
        "request_latency_mean_ms": 1025.0,
        "request_latency_max_ms": 1500.0,
        "prompt_tokens": 8000,
        "completion_tokens": 600,
        "tokens_per_episode": 2150.0,
    }]


def test_vlm_usage_leaves_out_tokens_when_a_response_reported_none(tmp_path: Path):
    _write_experiment(
        tmp_path,
        {
            "vlm_a": {
                "num_episodes_per_rebuild": [2],
                "usage_per_rebuild": [_usage(4, 1000, 100, num_responses_without_usage=1)],
            }
        },
    )

    (record,) = json.loads(VLMUsageKpiExporter().export_to_json(tmp_path).read_text(encoding="utf-8"))

    assert record["num_requests"] == 4
    assert record["requests_per_episode"] == 2.0
    assert "prompt_tokens" not in record
    assert "completion_tokens" not in record
    assert "tokens_per_episode" not in record


def test_vlm_usage_writes_no_file_without_vlm_runs(tmp_path: Path):
    _write_experiment(tmp_path, {"zero_action": {"num_episodes_per_rebuild": [3]}})

    assert VLMUsageKpiExporter().export_to_json(tmp_path) is None
    assert not (tmp_path / "kpis_vlm_usage.json").exists()


def test_vlm_usage_rejects_runs_with_different_models(tmp_path: Path):
    _write_experiment(
        tmp_path,
        {
            "vlm_a": {"num_episodes_per_rebuild": [1], "usage_per_rebuild": [_usage(1, 10, 1)]},
            "vlm_b": {"num_episodes_per_rebuild": [1], "usage_per_rebuild": [_usage(1, 10, 1, model="other")]},
        },
    )

    with pytest.raises(AssertionError, match="one endpoint and model"):
        VLMUsageKpiExporter().export_to_json(tmp_path)
