# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the rollout timing plot. No Isaac Sim or GPU required."""

import json
from pathlib import Path

from isaaclab_arena.visualization.plot_timings import read_run_timer_records


def _timer_record(name: str, count: int, total_ms: float, with_percentiles: bool = True) -> dict:
    """Build one record in the shape get_timer_stats_json writes."""
    mean_ms = total_ms / count
    record = {
        "type": "timing",
        "name": name,
        "app_name": "experiment_runner",
        "count": count,
        "mean_ms": mean_ms,
        "total_ms": total_ms,
        "min_ms": mean_ms,
        "max_ms": mean_ms,
    }
    if with_percentiles:
        record.update({"p10_ms": mean_ms, "p50_ms": mean_ms, "p90_ms": mean_ms * 2})
    return record


def _nested_records(steps: int = 10, with_percentiles: bool = True) -> list[dict]:
    """Records for a step that splits into policy inference and an env step with two children."""
    return [
        _timer_record("step", steps, 100.0 * steps, with_percentiles),
        _timer_record("step/policy_inference", steps, 20.0 * steps, with_percentiles),
        _timer_record("step/env_step", steps, 70.0 * steps, with_percentiles),
        _timer_record("step/env_step/sim_step", steps, 40.0 * steps, with_percentiles),
        _timer_record("step/env_step/record_camera_frames", steps, 25.0 * steps, with_percentiles),
    ]


def _write_timings(path: Path, records: list[dict], run_names: list[str] | None = None) -> Path:
    """Write either Experiment Runner shape (a bare list) or collected shape (runs keyed by name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_names is None:
        document: object = records
    else:
        document = {
            "totals": [],
            "runs": [{"run_name": run_name, **record} for run_name in run_names for record in records],
        }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_reads_the_experiment_runner_shape_and_names_the_run_after_its_directory(tmp_path):
    timings_path = _write_timings(tmp_path / "banana_in_bowl_pi0/timings.json", _nested_records())

    records_by_run_name = read_run_timer_records(timings_path)

    assert list(records_by_run_name) == ["banana_in_bowl_pi0"]
    assert len(records_by_run_name["banana_in_bowl_pi0"]) == 5


def test_reads_the_collected_shape_and_groups_by_run_name(tmp_path):
    timings_path = _write_timings(tmp_path / "timings.json", _nested_records(), run_names=["first", "second"])

    records_by_run_name = read_run_timer_records(timings_path)

    assert sorted(records_by_run_name) == ["first", "second"]
    assert len(records_by_run_name["first"]) == 5
