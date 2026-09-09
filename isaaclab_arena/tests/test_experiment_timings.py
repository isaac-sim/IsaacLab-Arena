# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-Run and Experiment-level timings files. No Isaac Sim or GPU required."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_TIMINGS_FILENAME
from isaaclab_arena.evaluation.experiment_timings import aggregate_experiment_timings, write_run_timings
from isaaclab_arena.utils.timer import Timer, reset_timer_stats


def _write_run_timings_file(run_output_directory: Path, timer_names: list[str]) -> Path:
    """Record one measurement per named timer and write them as that Run's timings."""
    reset_timer_stats()
    for timer_name in timer_names:
        with Timer(timer_name):
            pass
    return write_run_timings(run_output_directory)


def test_writes_a_runs_timings_into_its_output_directory(tmp_path):
    written_path = _write_run_timings_file(tmp_path / "first", ["step"])

    assert written_path == tmp_path / "first" / ARENA_EXPERIMENT_TIMINGS_FILENAME
    assert written_path.is_file()
    records = json.loads(written_path.read_text(encoding="utf-8"))
    assert [record["name"] for record in records] == ["step"]
    assert records[0]["app_name"] == "experiment_runner"


def test_creates_the_run_output_directory_if_it_is_missing(tmp_path):
    reset_timer_stats()
    with Timer("step"):
        pass

    written_path = write_run_timings(tmp_path / "not_yet_there")

    assert written_path.is_file()


def test_combines_runs_into_totals_and_per_run_records(tmp_path):
    _write_run_timings_file(tmp_path / "first", ["step"])
    _write_run_timings_file(tmp_path / "second", ["step"])

    experiment_timings_path = aggregate_experiment_timings(tmp_path, ["first", "second"])

    experiment_timings = json.loads(experiment_timings_path.read_text(encoding="utf-8"))
    assert experiment_timings_path == tmp_path / ARENA_EXPERIMENT_TIMINGS_FILENAME
    assert [total["name"] for total in experiment_timings["totals"]] == ["step"]
    # One measurement per Run, combined into the Experiment total.
    assert experiment_timings["totals"][0]["count"] == 2
    assert [(record["run_name"], record["name"]) for record in experiment_timings["runs"]] == [
        ("first", "step"),
        ("second", "step"),
    ]


def test_orders_runs_by_name_regardless_of_the_order_given(tmp_path):
    """Sorting here is what keeps the local and OSMO files identical for the same Experiment."""
    _write_run_timings_file(tmp_path / "alpha", ["step"])
    _write_run_timings_file(tmp_path / "zulu", ["step"])

    experiment_timings_path = aggregate_experiment_timings(tmp_path, ["zulu", "alpha"])

    experiment_timings = json.loads(experiment_timings_path.read_text(encoding="utf-8"))
    assert [record["run_name"] for record in experiment_timings["runs"]] == ["alpha", "zulu"]


def test_excludes_a_run_that_was_not_named(tmp_path):
    """Failed Runs are left out by the caller, and their directory is never read."""
    _write_run_timings_file(tmp_path / "passing", ["step"])
    (tmp_path / "failing").mkdir()

    experiment_timings_path = aggregate_experiment_timings(tmp_path, ["passing"])

    experiment_timings = json.loads(experiment_timings_path.read_text(encoding="utf-8"))
    assert [record["run_name"] for record in experiment_timings["runs"]] == ["passing"]


def test_rejects_a_named_run_without_a_timings_file(tmp_path):
    _write_run_timings_file(tmp_path / "first", ["step"])

    with pytest.raises(AssertionError, match="missing its timings file"):
        aggregate_experiment_timings(tmp_path, ["first", "never_ran"])


def test_writes_an_empty_experiment_file_when_no_runs_completed(tmp_path):
    experiment_timings_path = aggregate_experiment_timings(tmp_path, [])

    experiment_timings = json.loads(experiment_timings_path.read_text(encoding="utf-8"))
    assert experiment_timings == {"totals": [], "runs": []}
