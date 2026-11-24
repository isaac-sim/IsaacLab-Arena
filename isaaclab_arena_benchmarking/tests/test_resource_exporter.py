# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resource KPI exporter in ``isaaclab_arena_benchmarking/benchmarks/resource``."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_RESOURCES_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
)
from isaaclab_arena_benchmarking.benchmarks.registry import export_kpis
from isaaclab_arena_benchmarking.benchmarks.resource.resource_exporter import ResourceKpiExporter
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import assert_unique_tag_sets

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}

EXPERIMENT_NAME = "pnp_benchmark"


def _total(name: str, unit: str, count: int = 4, run_name: str | None = None, **extra: float) -> dict:
    total = {
        "name": name,
        "count": count,
        f"total_{unit}": 400.0,
        f"min_{unit}": 50.0,
        f"max_{unit}": 150.0,
        f"mean_{unit}": 100.0,
        f"samples_{unit}": [50.0, 150.0],
        **extra,
    }
    if run_name is not None:
        total["run_name"] = run_name
    return total


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


def _write_resources(output_dir: Path, totals: list[dict]) -> None:
    resources = {"totals": totals, "runs": [_total("vram", "mb", run_name="run_a")]}
    (output_dir / ARENA_AGGREGATED_RESOURCES_FILENAME).write_text(json.dumps(resources), encoding="utf-8")


def test_resource_exporter_writes_megabyte_total_without_total_or_samples(tmp_path: Path):
    _write_result(tmp_path)
    _write_resources(
        tmp_path,
        [_total("vram", "mb", p10_mb=60.0, p50_mb=100.0, p90_mb=140.0), _total("ram", "mb", count=0)],
    )

    exporter = ResourceKpiExporter()
    assert exporter.get_records_filename() == "kpis_resource.json"
    records_path = exporter.export_to_json(tmp_path)

    assert records_path == tmp_path / "kpis_resource.json"
    (record,) = json.loads(records_path.read_text(encoding="utf-8"))
    assert record == {
        "type": "resource",
        "timestamp_ns": END_TIMESTAMP_NS,
        "git_sha": "1a2b3c4",
        "git_branch": "main",
        "benchmark_id": EXPERIMENT_NAME,
        "name": "vram",
        "count": 4,
        "min_mb": 50.0,
        "max_mb": 150.0,
        "mean_mb": 100.0,
        "p10_mb": 60.0,
        "p50_mb": 100.0,
        "p90_mb": 140.0,
    }
    assert list(record)[5:] == ["name", "count", "min_mb", "max_mb", "mean_mb", "p10_mb", "p50_mb", "p90_mb"]


def test_resource_exporter_writes_percent_total(tmp_path: Path):
    _write_result(tmp_path)
    _write_resources(tmp_path, [_total("gpu_util", "percent")])

    records_path = ResourceKpiExporter().export_to_json(tmp_path)

    (record,) = json.loads(records_path.read_text(encoding="utf-8"))
    assert record["name"] == "gpu_util"
    assert (record["min_percent"], record["max_percent"], record["mean_percent"]) == (50.0, 150.0, 100.0)
    assert not [key for key in record if key.endswith("_mb") or key.startswith(("total_", "samples_", "p"))]


def test_resource_records_of_different_names_have_unique_tag_sets(tmp_path: Path):
    _write_result(tmp_path)
    _write_resources(tmp_path, [_total("vram", "mb"), _total("ram", "mb"), _total("cpu_util", "percent")])

    records = ResourceKpiExporter().build_records_from_output_dir(tmp_path)

    assert len(records) == 3
    assert_unique_tag_sets(records)


def test_resource_exporter_writes_nothing_without_resources_file(tmp_path: Path):
    _write_result(tmp_path)
    assert ResourceKpiExporter().build_records_from_output_dir(tmp_path) == []
    assert ResourceKpiExporter().export_to_json(tmp_path) is None


def test_resource_exporter_is_selectable_through_export_kpis(tmp_path: Path):
    _write_result(tmp_path)
    _write_resources(tmp_path, [_total("vram", "mb")])

    assert export_kpis(tmp_path, ["resource"]) == [tmp_path / "kpis_resource.json"]
