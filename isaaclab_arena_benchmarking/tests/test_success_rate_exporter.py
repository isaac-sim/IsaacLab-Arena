# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the success-rate KPI exporter in ``isaaclab_arena_benchmarking/benchmarks/success_rate``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_RESULT_FILENAME  # noqa: E402
from isaaclab_arena.evaluation.report_data import build_experiment_summary  # noqa: E402
from isaaclab_arena_benchmarking.benchmarks.success_rate.success_rate_exporter import (  # noqa: E402
    PerExperimentRecord,
    SuccessRateKpiExporter,
)

START_TIMESTAMP_NS = 1792799000000000000
"""Before every Run end timestamp, in Unix epoch nanoseconds."""

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}

EXPERIMENT_NAME = "pnp_benchmark"

SHARED_VALUES = {"timestamp_ns": END_TIMESTAMP_NS, **GIT_STAMP, "benchmark_id": EXPERIMENT_NAME}
"""Values every record built from a default ``_write_experiment`` output carries."""


def _episode_records(num_episodes: int, num_successes: int, *, episode_length: int = 10) -> list[dict]:
    return [
        {
            "env_id": 0,
            "episode_in_env": episode_index,
            "success": episode_index < num_successes,
            "episode_length": episode_length,
        }
        for episode_index in range(num_episodes)
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def _write_experiment(experiment_output_directory: Path, runs: dict[str, dict]) -> Path:
    """Write a result file and the per-Run episode JSONL.

    Each entry of ``runs`` may set ``episodes`` (records written to ``<run>/episode_results.jsonl``) and
    ``end_timestamp_ns``.
    """
    result_runs = {}
    for run_name, run in runs.items():
        episodes = run.get("episodes", [])
        if episodes:
            _write_jsonl(experiment_output_directory / run_name / "episode_results.jsonl", episodes)
        result_runs[run_name] = {
            "environment": {"name": f"{run_name}_env", "definition": "unused"},
            "policy_variant": "pi05",
            "status": "completed",
            "start_timestamp_ns": START_TIMESTAMP_NS,
            "end_timestamp_ns": run.get("end_timestamp_ns", END_TIMESTAMP_NS),
            "rebuilds": [{"index": 0, "episodes": episodes}],
        }
    experiment_output_directory.mkdir(parents=True, exist_ok=True)
    (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(
        json.dumps({"git_stamp": GIT_STAMP, "experiment_name": EXPERIMENT_NAME, "runs": result_runs}), encoding="utf-8"
    )
    return experiment_output_directory


def _build_record(experiment_output_directory: Path) -> dict:
    (record,) = SuccessRateKpiExporter().build_records_from_output_dir(experiment_output_directory)
    assert isinstance(record, PerExperimentRecord)
    return record.to_flat_dict()


def test_success_rate_is_episode_weighted_over_runs(tmp_path: Path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {"pnp_pi05": {"episodes": _episode_records(50, 41)}, "pnp_pi0": {"episodes": _episode_records(10, 2)}},
    )

    assert _build_record(experiment_output_directory) == {
        "type": "per_experiment",
        **SHARED_VALUES,
        "success_rate": 43 / 60,
        "num_episodes": 60,
        "num_runs": 2,
    }


def test_zero_scored_episodes_omits_success_rate(tmp_path: Path):
    unscored_episodes = [
        {"env_id": 0, "episode_in_env": episode_index, "success": None, "episode_length": 5}
        for episode_index in range(3)
    ]
    experiment_output_directory = _write_experiment(tmp_path / "experiment", {"run_a": {"episodes": unscored_episodes}})

    record = _build_record(experiment_output_directory)

    assert "success_rate" not in record
    assert record["num_episodes"] == 3
    assert record["num_runs"] == 1


def test_record_has_no_run_counts_of_completed_or_failed(tmp_path: Path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment", {"run_a": {"episodes": _episode_records(2, 1)}}
    )

    record = _build_record(experiment_output_directory)

    assert set(record) == {
        "type",
        "timestamp_ns",
        "git_sha",
        "git_branch",
        "benchmark_id",
        "success_rate",
        "num_episodes",
        "num_runs",
    }


def test_success_rate_exporter_writes_kpis_success_rates_file(tmp_path: Path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment", {"run_a": {"episodes": _episode_records(4, 3)}}
    )

    exporter = SuccessRateKpiExporter()
    assert exporter.get_records_filename() == "kpis_success_rates.json"
    records_path = exporter.export_to_json(experiment_output_directory)

    assert records_path == experiment_output_directory / "kpis_success_rates.json"
    assert json.loads(records_path.read_text(encoding="utf-8")) == [{
        "type": "per_experiment",
        **SHARED_VALUES,
        "success_rate": 0.75,
        "num_episodes": 4,
        "num_runs": 1,
    }]


def test_report_job_names_match_run_names(tmp_path: Path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {"run_a": {"episodes": _episode_records(2, 1)}, "run_b": {"episodes": _episode_records(3, 3)}},
    )

    summary = build_experiment_summary(experiment_output_directory, experiment_output_directory.name)

    assert sorted(job.name for job in summary.jobs) == ["run_a", "run_b"]


def test_episodes_in_nested_run_directories_are_pooled_into_their_run(tmp_path: Path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment", {"run_a": {"episodes": _episode_records(2, 1)}}
    )
    _write_jsonl(
        experiment_output_directory / "run_a" / "nested" / "episode_results_rebuild1.jsonl",
        [{"env_id": 0, "episode_in_env": 0, "success": True, "episode_length": 4}],
    )

    record = _build_record(experiment_output_directory)

    assert record["num_episodes"] == 3
    assert record["success_rate"] == 2 / 3


def test_timestamp_is_latest_run_end(tmp_path: Path):
    latest_end_timestamp_ns = END_TIMESTAMP_NS + 5_000_000_000
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {
            "early": {"episodes": _episode_records(1, 0), "end_timestamp_ns": END_TIMESTAMP_NS},
            "late": {"episodes": _episode_records(1, 1), "end_timestamp_ns": latest_end_timestamp_ns},
        },
    )

    assert _build_record(experiment_output_directory)["timestamp_ns"] == latest_end_timestamp_ns
