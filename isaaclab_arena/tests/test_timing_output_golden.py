# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Golden test pinning the exact text of the timing files and the timing KPI file."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
)
from isaaclab_arena.evaluation.experiment_timings import aggregate_experiment_timings, write_run_timings
from isaaclab_arena.utils.timer import _update_stats, reset_timer_stats, write_timer_stats_json
from isaaclab_arena_benchmarking.benchmarks.timing.timing_exporter import TimingKpiExporter

END_TIMESTAMP_NS = 1792800000000000000
GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}
EXPERIMENT_NAME = "pnp_benchmark"

RUN_NAMES = ["run_a", "run_b"]

SAMPLES_BY_RUN = {
    "run_a": {"step": [10.0, 20.0, 30.0], "step/env_step": [1.5, 2.5]},
    "run_b": {"step": [12.0, 18.0], "step/env_step": [2.0, 3.0, 4.0]},
}
"""Fixed measurements per Run, fewer than the reservoir size so no random sampling happens."""

GOLDEN_TIMER_STATS_JSON = (
    "[\n"
    "  {\n"
    '    "type": "timing",\n'
    '    "name": "step",\n'
    '    "app_name": "experiment_runner",\n'
    '    "count": 3,\n'
    '    "mean_ms": 20.0,\n'
    '    "total_ms": 60.0,\n'
    '    "min_ms": 10.0,\n'
    '    "max_ms": 30.0,\n'
    '    "p10_ms": 10.0,\n'
    '    "p50_ms": 20.0,\n'
    '    "p90_ms": 30.0,\n'
    '    "samples_ms": [10.0, 20.0, 30.0]\n'
    "  },\n"
    "  {\n"
    '    "type": "timing",\n'
    '    "name": "step/env_step",\n'
    '    "app_name": "experiment_runner",\n'
    '    "count": 2,\n'
    '    "mean_ms": 2.0,\n'
    '    "total_ms": 4.0,\n'
    '    "min_ms": 1.5,\n'
    '    "max_ms": 2.5,\n'
    '    "p10_ms": 1.5,\n'
    '    "p50_ms": 1.5,\n'
    '    "p90_ms": 2.5,\n'
    '    "samples_ms": [1.5, 2.5]\n'
    "  }\n"
    "]\n"
)

GOLDEN_AGGREGATED_TIMINGS_JSON = (
    "{\n"
    '  "totals": [\n'
    "    {\n"
    '      "name": "step",\n'
    '      "count": 5,\n'
    '      "total_ms": 90.0,\n'
    '      "min_ms": 10.0,\n'
    '      "max_ms": 30.0,\n'
    '      "mean_ms": 18.0,\n'
    '      "p10_ms": 10.0,\n'
    '      "p50_ms": 18.0,\n'
    '      "p90_ms": 30.0\n'
    "    },\n"
    "    {\n"
    '      "name": "step/env_step",\n'
    '      "count": 5,\n'
    '      "total_ms": 13.0,\n'
    '      "min_ms": 1.5,\n'
    '      "max_ms": 4.0,\n'
    '      "mean_ms": 2.6,\n'
    '      "p10_ms": 1.5,\n'
    '      "p50_ms": 2.5,\n'
    '      "p90_ms": 4.0\n'
    "    }\n"
    "  ],\n"
    '  "runs": [\n'
    "    {\n"
    '      "run_name": "run_a",\n'
    '      "type": "timing",\n'
    '      "name": "step",\n'
    '      "app_name": "experiment_runner",\n'
    '      "count": 3,\n'
    '      "mean_ms": 20.0,\n'
    '      "total_ms": 60.0,\n'
    '      "min_ms": 10.0,\n'
    '      "max_ms": 30.0,\n'
    '      "p10_ms": 10.0,\n'
    '      "p50_ms": 20.0,\n'
    '      "p90_ms": 30.0\n'
    "    },\n"
    "    {\n"
    '      "run_name": "run_a",\n'
    '      "type": "timing",\n'
    '      "name": "step/env_step",\n'
    '      "app_name": "experiment_runner",\n'
    '      "count": 2,\n'
    '      "mean_ms": 2.0,\n'
    '      "total_ms": 4.0,\n'
    '      "min_ms": 1.5,\n'
    '      "max_ms": 2.5,\n'
    '      "p10_ms": 1.5,\n'
    '      "p50_ms": 1.5,\n'
    '      "p90_ms": 2.5\n'
    "    },\n"
    "    {\n"
    '      "run_name": "run_b",\n'
    '      "type": "timing",\n'
    '      "name": "step",\n'
    '      "app_name": "experiment_runner",\n'
    '      "count": 2,\n'
    '      "mean_ms": 15.0,\n'
    '      "total_ms": 30.0,\n'
    '      "min_ms": 12.0,\n'
    '      "max_ms": 18.0,\n'
    '      "p10_ms": 12.0,\n'
    '      "p50_ms": 12.0,\n'
    '      "p90_ms": 18.0\n'
    "    },\n"
    "    {\n"
    '      "run_name": "run_b",\n'
    '      "type": "timing",\n'
    '      "name": "step/env_step",\n'
    '      "app_name": "experiment_runner",\n'
    '      "count": 3,\n'
    '      "mean_ms": 3.0,\n'
    '      "total_ms": 9.0,\n'
    '      "min_ms": 2.0,\n'
    '      "max_ms": 4.0,\n'
    '      "p10_ms": 2.0,\n'
    '      "p50_ms": 3.0,\n'
    '      "p90_ms": 4.0\n'
    "    }\n"
    "  ]\n"
    "}\n"
)

GOLDEN_KPIS_TIMING_JSON = (
    "[\n"
    "  {\n"
    '    "type": "timing",\n'
    '    "timestamp_ns": 1792800000000000000,\n'
    '    "git_sha": "1a2b3c4",\n'
    '    "git_branch": "main",\n'
    '    "benchmark_id": "pnp_benchmark",\n'
    '    "name": "step",\n'
    '    "mean_ms": 18.0,\n'
    '    "total_ms": 90.0,\n'
    '    "min_ms": 10.0,\n'
    '    "max_ms": 30.0,\n'
    '    "p10_ms": 10.0,\n'
    '    "p50_ms": 18.0,\n'
    '    "p90_ms": 30.0,\n'
    '    "count": 5\n'
    "  },\n"
    "  {\n"
    '    "type": "timing",\n'
    '    "timestamp_ns": 1792800000000000000,\n'
    '    "git_sha": "1a2b3c4",\n'
    '    "git_branch": "main",\n'
    '    "benchmark_id": "pnp_benchmark",\n'
    '    "name": "step/env_step",\n'
    '    "mean_ms": 2.6,\n'
    '    "total_ms": 13.0,\n'
    '    "min_ms": 1.5,\n'
    '    "max_ms": 4.0,\n'
    '    "p10_ms": 1.5,\n'
    '    "p50_ms": 2.5,\n'
    '    "p90_ms": 4.0,\n'
    '    "count": 5\n'
    "  }\n"
    "]\n"
)


def _record_run_samples(run_name: str) -> None:
    reset_timer_stats()
    for name, samples in SAMPLES_BY_RUN[run_name].items():
        for sample in samples:
            _update_stats(name, sample)


def _write_result(output_dir: Path) -> None:
    result = {
        "git_stamp": GIT_STAMP,
        "experiment_name": EXPERIMENT_NAME,
        "runs": {
            "run_a": {"status": "completed", "end_timestamp_ns": END_TIMESTAMP_NS - 1},
            "run_b": {"status": "completed", "end_timestamp_ns": END_TIMESTAMP_NS},
        },
    }
    (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result), encoding="utf-8")


def test_timer_stats_json_text_is_unchanged(tmp_path: Path):
    _record_run_samples("run_a")
    path = write_timer_stats_json(tmp_path / "t.json", app_name="experiment_runner")
    reset_timer_stats()
    assert path.read_text(encoding="utf-8") == GOLDEN_TIMER_STATS_JSON


def test_aggregated_timings_and_kpis_text_are_unchanged(tmp_path: Path):
    for run_name in RUN_NAMES:
        _record_run_samples(run_name)
        write_run_timings(tmp_path / run_name)
    reset_timer_stats()
    aggregated_path = aggregate_experiment_timings(tmp_path, RUN_NAMES)
    assert aggregated_path == tmp_path / ARENA_AGGREGATED_TIMINGS_FILENAME
    assert aggregated_path.read_text(encoding="utf-8") == GOLDEN_AGGREGATED_TIMINGS_JSON

    _write_result(tmp_path)
    kpis_path = TimingKpiExporter().export_to_json(tmp_path)
    assert kpis_path.read_text(encoding="utf-8") == GOLDEN_KPIS_TIMING_JSON
