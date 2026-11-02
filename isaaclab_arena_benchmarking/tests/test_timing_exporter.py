# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timing KPI exporter in ``isaaclab_arena_benchmarking/benchmarks/timing``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_arena.evaluation.arena_experiment_result import (  # noqa: E402
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
)
from isaaclab_arena_benchmarking.benchmarks.timing.timing_exporter import TimingKpiExporter  # noqa: E402

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}

BENCHMARK_ID = "pnp_benchmark"


def _timing(name: str, count: int, total_ms: float, run_name: str | None = None) -> dict:
    timing = {"name": name, "count": count, "total_ms": total_ms, "min_ms": 5.0, "max_ms": 15.0}
    if run_name is not None:
        timing["run_name"] = run_name
    return timing


def _write_result(output_dir: Path) -> None:
    result = {
        "git_stamp": GIT_STAMP,
        "benchmark_id": BENCHMARK_ID,
        "runs": {
            "run_a": {"status": "completed", "end_timestamp_ns": END_TIMESTAMP_NS - 1},
            "run_b": {"status": "completed", "end_timestamp_ns": END_TIMESTAMP_NS},
        },
    }
    (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result), encoding="utf-8")


def test_timing_exporter_writes_totals_only(tmp_path: Path):
    _write_result(tmp_path)
    timings = {
        "totals": [_timing("rollout/step", 4, 40.0), _timing("never_ran", 0, 0.0)],
        "runs": [_timing("rollout/step", 4, 40.0, run_name="run_a")],
    }
    (tmp_path / ARENA_AGGREGATED_TIMINGS_FILENAME).write_text(json.dumps(timings), encoding="utf-8")

    exporter = TimingKpiExporter()
    assert exporter.get_records_filename() == "kpis_timing.json"
    records_path = exporter.export_to_json(tmp_path)

    assert records_path == tmp_path / "kpis_timing.json"
    assert json.loads(records_path.read_text(encoding="utf-8")) == [{
        "type": "timing",
        "timestamp_ns": END_TIMESTAMP_NS,
        "git_sha": "1a2b3c4",
        "git_branch": "main",
        "benchmark_id": BENCHMARK_ID,
        "name": "rollout/step",
        "mean_ms": 10.0,
        "total_ms": 40.0,
        "min_ms": 5.0,
        "max_ms": 15.0,
        "count": 4,
    }]


def test_timing_exporter_writes_nothing_without_timings_file(tmp_path: Path):
    _write_result(tmp_path)
    assert TimingKpiExporter().build_records_from_output_dir(tmp_path) == []
