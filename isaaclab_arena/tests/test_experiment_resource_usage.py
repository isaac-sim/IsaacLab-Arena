# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-Run and Experiment-level resource files. No Isaac Sim or GPU required."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME,
    ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME,
)
from isaaclab_arena.evaluation.experiment_resource_usage import (
    aggregate_experiment_resource_usage,
    write_run_resource_usage,
)
from isaaclab_arena.utils.compute_resource.compute_resource_usage import ResourceUsage


def _usage(vram_mb: list[float]) -> ResourceUsage:
    usage = ResourceUsage()
    for value in vram_mb:
        usage.vram.update(value)
    return usage


def _write_run(run_output_directory: Path, vram_mb: list[float]) -> Path:
    return write_run_resource_usage(run_output_directory, _usage(vram_mb))


def test_constants():
    assert ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME == "arena_run_resource_usage.json"
    assert ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME == "arena_experiment_resource_usage.json"


def test_writes_a_runs_resources_into_its_output_directory(tmp_path):
    written_path = _write_run(tmp_path / "first", [100.0, 300.0])

    assert written_path == tmp_path / "first" / ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME
    records = json.loads(written_path.read_text(encoding="utf-8"))
    assert [record["name"] for record in records] == ["vram"]
    assert records[0]["app_name"] == "experiment_runner"
    assert records[0]["samples_mb"] == [100.0, 300.0]
    assert '"samples_mb": [100.0, 300.0]' in written_path.read_text(encoding="utf-8")


def test_creates_the_run_output_directory_if_it_is_missing(tmp_path):
    assert _write_run(tmp_path / "not_yet_there", [1.0]).is_file()


def test_combines_runs_into_totals_and_per_run_records(tmp_path):
    _write_run(tmp_path / "first", [100.0, 300.0])
    _write_run(tmp_path / "second", [200.0, 500.0])

    path = aggregate_experiment_resource_usage(tmp_path, ["first", "second"])

    assert path == tmp_path / ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME
    experiment_resource_usage = json.loads(path.read_text(encoding="utf-8"))
    (total,) = experiment_resource_usage["totals"]
    assert total["name"] == "vram"
    assert total["count"] == 4
    assert total["max_mb"] == 500.0
    assert [(record["run_name"], record["name"]) for record in experiment_resource_usage["runs"]] == [
        ("first", "vram"),
        ("second", "vram"),
    ]
    assert "samples_mb" not in total
    assert all("samples_mb" not in record for record in experiment_resource_usage["runs"])
    assert experiment_resource_usage["runs"][0]["max_mb"] == 300.0


def test_orders_runs_by_name_regardless_of_the_order_given(tmp_path):
    _write_run(tmp_path / "alpha", [1.0])
    _write_run(tmp_path / "zulu", [2.0])

    path = aggregate_experiment_resource_usage(tmp_path, ["zulu", "alpha"])

    experiment_resource_usage = json.loads(path.read_text(encoding="utf-8"))
    assert [record["run_name"] for record in experiment_resource_usage["runs"]] == ["alpha", "zulu"]


def test_rejects_a_named_run_without_a_resources_file(tmp_path):
    _write_run(tmp_path / "first", [1.0])

    with pytest.raises(AssertionError, match="missing its resources file") as error:
        aggregate_experiment_resource_usage(tmp_path, ["first", "never_ran"])
    assert "never_ran" in str(error.value)
    assert str(tmp_path / "never_ran" / ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME) in str(error.value)


def test_writes_an_empty_experiment_file_when_no_runs_completed(tmp_path):
    path = aggregate_experiment_resource_usage(tmp_path, [])

    assert json.loads(path.read_text(encoding="utf-8")) == {"totals": [], "runs": []}
