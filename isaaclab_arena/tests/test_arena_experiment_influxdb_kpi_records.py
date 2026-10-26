# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for building and writing the Arena Experiment InfluxDB KPI records file."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_influxdb_kpi_records import (
    ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME,
    PerExperimentRecord,
    PerRunRecord,
    TimingRecord,
    assert_unique_tag_sets,
    build_experiment_influxdb_kpi_records,
    write_experiment_influxdb_kpi_records,
)
from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
)
from isaaclab_arena.tests.utils.influxdb_kpi_records_validation import assert_influxdb_kpi_records_are_valid
from isaaclab_arena.visualization.report_data import build_experiment_summary

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

PROVENANCE = {"git_sha": "1a2b3c4", "git_branch": "main"}

SHARED_VALUES = {"timestamp_ns": END_TIMESTAMP_NS, **PROVENANCE}
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


def _write_experiment(
    experiment_output_directory: Path,
    runs: dict[str, dict],
    timings: dict | None = None,
) -> Path:
    """Write a result file, per-Run episode JSONL and optionally a timings file.

    Each entry of ``runs`` may set ``status``, ``end_timestamp_ns``, ``environment``, ``policy_variant`` and
    ``episodes`` (records written to ``<run>/episode_results.jsonl``).
    """
    result_runs = {}
    for run_name, run in runs.items():
        episodes = run.get("episodes", [])
        if episodes:
            _write_jsonl(experiment_output_directory / run_name / "episode_results.jsonl", episodes)
        result_run = {
            "environment": {"name": run.get("environment", f"{run_name}_env"), "definition": "unused"},
            "policy_variant": run.get("policy_variant", "pi05"),
            "status": run.get("status", "completed"),
            "rebuilds": [{"index": 0, "episodes": episodes}],
        }
        if run.get("end_timestamp_ns", END_TIMESTAMP_NS) is not None:
            result_run["end_timestamp_ns"] = run.get("end_timestamp_ns", END_TIMESTAMP_NS)
        result_runs[run_name] = result_run
    experiment_output_directory.mkdir(parents=True, exist_ok=True)
    (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(
        json.dumps({"provenance": PROVENANCE, "runs": result_runs}), encoding="utf-8"
    )
    if timings is not None:
        (experiment_output_directory / ARENA_AGGREGATED_TIMINGS_FILENAME).write_text(
            json.dumps(timings), encoding="utf-8"
        )
    return experiment_output_directory


def _build_kpi_records_list(experiment_output_directory: Path) -> list[dict]:
    return [record.to_flat_dict() for record in build_experiment_influxdb_kpi_records(experiment_output_directory)]


def _records_of_type(kpis: list[dict], record_type: str) -> list[dict]:
    return [record for record in kpis if record["type"] == record_type]


def _per_run(kpis: list[dict], run_name: str) -> dict:
    (record,) = (
        record for record in _records_of_type(kpis, PerRunRecord.record_type) if record["run_name"] == run_name
    )
    return record


def _per_experiment(kpis: list[dict]) -> dict:
    (record,) = _records_of_type(kpis, PerExperimentRecord.record_type)
    return record


def test_per_run_and_per_experiment_success_rate(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {
            "pnp_pi05": {"episodes": _episode_records(50, 41), "environment": "droid_pnp"},
            "pnp_pi0": {"episodes": _episode_records(50, 40, episode_length=20), "policy_variant": "pi0"},
        },
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert _per_run(kpis, "pnp_pi05") == {
        "type": "per_run",
        **SHARED_VALUES,
        "run_name": "pnp_pi05",
        "environment": "droid_pnp",
        "policy_variant": "pi05",
        "success_rate": 41 / 50,
        "num_episodes": 50,
        "num_successes": 41,
        "mean_episode_length": 10.0,
        "completed": 1,
    }
    assert _per_run(kpis, "pnp_pi0")["num_successes"] == 40
    assert _per_run(kpis, "pnp_pi0")["policy_variant"] == "pi0"
    assert _per_run(kpis, "pnp_pi0")["mean_episode_length"] == 20.0
    assert _per_experiment(kpis) == {
        "type": "per_experiment",
        **SHARED_VALUES,
        "success_rate": 81 / 100,
        "num_episodes": 100,
        "num_runs": 2,
        "num_runs_completed": 2,
        "num_runs_failed": 0,
    }
    assert_influxdb_kpi_records_are_valid(kpis)


def test_report_job_names_match_run_names(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {"run_a": {"episodes": _episode_records(2, 1)}, "run_b": {"episodes": _episode_records(3, 3)}},
    )

    summary = build_experiment_summary(experiment_output_directory, experiment_output_directory.name)

    assert sorted(job.name for job in summary.jobs) == ["run_a", "run_b"]


def test_episodes_in_nested_run_directories_are_pooled_into_their_run(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {"run_a": {"episodes": _episode_records(2, 1)}},
    )
    _write_jsonl(
        experiment_output_directory / "run_a" / "nested" / "episode_results_rebuild1.jsonl",
        [{"env_id": 0, "episode_in_env": 0, "success": True, "episode_length": 4}],
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert _per_run(kpis, "run_a")["num_episodes"] == 3
    assert _per_run(kpis, "run_a")["num_successes"] == 2


def test_timestamp_is_latest_run_end(tmp_path):
    t1 = END_TIMESTAMP_NS
    t2 = END_TIMESTAMP_NS + 5_000_000_000
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {
            "late": {"episodes": _episode_records(1, 1), "end_timestamp_ns": t2},
            "early": {"episodes": _episode_records(1, 0), "end_timestamp_ns": t1},
        },
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert {record["timestamp_ns"] for record in kpis} == {t2}


def test_failed_run_has_no_success_rate(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {
            "completed_run": {"episodes": _episode_records(10, 4)},
            # Partial episodes of a failed Run do not count.
            "failed_run": {"episodes": _episode_records(10, 10), "status": "failed"},
        },
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    failed_record = _per_run(kpis, "failed_run")
    assert failed_record["completed"] == 0
    assert "success_rate" not in failed_record
    per_experiment = _per_experiment(kpis)
    assert per_experiment["num_runs"] == 2
    assert per_experiment["num_runs_completed"] == 1
    assert per_experiment["num_runs_failed"] == 1
    assert per_experiment["num_episodes"] == 10
    assert per_experiment["success_rate"] == 4 / 10
    assert_influxdb_kpi_records_are_valid(kpis)


def test_all_failed_experiment_still_has_timestamp(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {
            "a": {"status": "failed", "end_timestamp_ns": END_TIMESTAMP_NS + 1},
            "b": {"status": "failed"},
        },
        timings={"totals": [], "runs": []},
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert_influxdb_kpi_records_are_valid(kpis)
    assert {record["timestamp_ns"] for record in kpis} == {END_TIMESTAMP_NS + 1}
    per_experiment = _per_experiment(kpis)
    assert "success_rate" not in per_experiment
    assert per_experiment["num_runs_failed"] == 2
    assert per_experiment["num_runs_completed"] == 0
    assert per_experiment["num_episodes"] == 0


def test_run_with_only_unscored_episodes_has_no_success_rate(tmp_path):
    unscored_episodes = [
        {"env_id": 0, "episode_in_env": episode_index, "success": None, "episode_length": 5}
        for episode_index in range(3)
    ]
    experiment_output_directory = _write_experiment(tmp_path / "experiment", {"run": {"episodes": unscored_episodes}})

    kpis = _build_kpi_records_list(experiment_output_directory)

    per_run = _per_run(kpis, "run")
    assert "success_rate" not in per_run
    assert per_run["num_episodes"] == 3
    assert per_run["num_successes"] == 0
    assert per_run["completed"] == 1
    assert "success_rate" not in _per_experiment(kpis)
    assert_influxdb_kpi_records_are_valid(kpis)


def test_mean_progress_only_when_episodes_carry_progress(tmp_path):
    progress_episodes = [
        {"env_id": 0, "episode_in_env": 0, "success": True, "episode_length": 5, "progress": {"overall_score": 1.0}},
        {"env_id": 0, "episode_in_env": 1, "success": False, "episode_length": 5, "progress": {"overall_score": 0.5}},
    ]
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {"with_progress": {"episodes": progress_episodes}, "without_progress": {"episodes": _episode_records(2, 1)}},
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert _per_run(kpis, "with_progress")["mean_progress"] == 0.75
    assert "mean_progress" not in _per_run(kpis, "without_progress")


def test_timing_records(tmp_path):
    timings = {
        "totals": [
            {"name": "rollout/step", "count": 4, "total_ms": 40.0, "min_ms": 5.0, "max_ms": 15.0, "mean_ms": 10.0},
            {"name": "never_ran", "count": 0, "total_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0},
        ],
        "runs": [
            {
                "run_name": "a",
                "type": "timing",
                "app_name": "experiment_runner",
                "name": "rollout/step",
                "count": 4,
                "total_ms": 40.0,
                "min_ms": 5.0,
                "max_ms": 15.0,
                "mean_ms": 10.0,
                "p50_ms": 9.0,
            },
        ],
    }
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment", {"a": {"episodes": _episode_records(1, 1)}}, timings=timings
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert _records_of_type(kpis, TimingRecord.record_type) == [
        {
            "type": "timing",
            **SHARED_VALUES,
            "name": "rollout/step",
            "run_name": "all",
            "mean_ms": 10.0,
            "total_ms": 40.0,
            "min_ms": 5.0,
            "max_ms": 15.0,
            "count": 4,
        },
        {
            "type": "timing",
            **SHARED_VALUES,
            "name": "rollout/step",
            "run_name": "a",
            "mean_ms": 10.0,
            "total_ms": 40.0,
            "min_ms": 5.0,
            "max_ms": 15.0,
            "count": 4,
        },
    ]
    assert_influxdb_kpi_records_are_valid(kpis)


def test_missing_timings_file_omits_timing_records(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment", {"a": {"episodes": _episode_records(1, 1)}}
    )

    kpis = _build_kpi_records_list(experiment_output_directory)

    assert _records_of_type(kpis, TimingRecord.record_type) == []


def test_missing_end_timestamp_names_the_run(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {"has_end": {"episodes": _episode_records(1, 1)}, "old_run": {"end_timestamp_ns": None}},
    )

    with pytest.raises(AssertionError, match="old_run"):
        _build_kpi_records_list(experiment_output_directory)


def _per_run_record(**overrides) -> PerRunRecord:
    """Return a valid completed PerRunRecord, updated with ``overrides``."""
    return PerRunRecord(
        **{**SHARED_VALUES, "run_name": "a", "environment": "e", "policy_variant": "p", "completed": 1, **overrides}
    )


def _valid_records() -> list[dict]:
    """Return the flat dicts of a valid per-Experiment and per-Run record."""
    return [
        PerExperimentRecord(
            **SHARED_VALUES, num_episodes=1, num_runs=1, num_runs_completed=1, num_runs_failed=0
        ).to_flat_dict(),
        _per_run_record().to_flat_dict(),
    ]


def test_records_are_valid_influxdb_points():
    assert_influxdb_kpi_records_are_valid(_valid_records())


def test_unique_tag_sets_reject_duplicate_records():
    assert_unique_tag_sets([_per_run_record(), _per_run_record(run_name="b")])
    with pytest.raises(AssertionError, match="record 1 has the same tags as record 0"):
        assert_unique_tag_sets([_per_run_record(), _per_run_record(completed=0)])


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"timestamp_ns": 1.5e18}, id="float_timestamp_ns"),
        pytest.param({"timestamp_ns": True}, id="bool_timestamp_ns"),
        pytest.param({"timestamp_ns": 1_000}, id="tiny_timestamp_ns"),
        pytest.param({"timestamp_ns": 1767225599999999999}, id="timestamp_ns_before_2026"),
        pytest.param({"git_sha": ""}, id="empty_git_sha"),
        pytest.param({"git_branch": "a\nb"}, id="line_break_git_branch"),
        pytest.param({"run_name": ""}, id="empty_tag"),
        pytest.param({"run_name": "a\nb"}, id="line_break_tag"),
        pytest.param({"completed": True}, id="bool_field"),
        pytest.param({"success_rate": math.nan}, id="nan_field"),
        pytest.param({"success_rate": math.inf}, id="inf_field"),
        pytest.param({"success_rate": "1"}, id="string_field"),
    ],
)
def test_record_rejects_invalid_values(overrides):
    with pytest.raises(AssertionError):
        _per_run_record(**overrides)


def test_timestamp_ns_does_not_count_as_a_field():
    with pytest.raises(AssertionError, match="at least one numeric field"):
        _per_run_record(completed=None)


def test_failed_run_record_omits_missing_fields():
    record = _per_run_record(completed=0)

    assert list(record.to_flat_dict().items()) == [
        ("type", "per_run"),
        ("timestamp_ns", END_TIMESTAMP_NS),
        ("git_sha", "1a2b3c4"),
        ("git_branch", "main"),
        ("run_name", "a"),
        ("environment", "e"),
        ("policy_variant", "p"),
        ("completed", 0),
    ]


def _set_record_value(kpis: list[dict], key: str, value: object) -> None:
    kpis[1][key] = value


def _duplicate_record(kpis: list[dict]) -> None:
    kpis.append({**kpis[1], "completed": 0})


def _keep_only_timestamp_ns(kpis: list[dict]) -> None:
    kpis[1] = {key: value for key, value in kpis[1].items() if not isinstance(value, (int, float))}
    kpis[1]["timestamp_ns"] = END_TIMESTAMP_NS


@pytest.mark.parametrize(
    "make_invalid",
    [
        pytest.param(lambda kpis: kpis.clear(), id="empty_list"),
        pytest.param(lambda kpis: kpis[1].pop("timestamp_ns"), id="missing_timestamp_ns"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", 1.8e18), id="float_timestamp_ns"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", True), id="bool_timestamp_ns"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", 1767225599999999999), id="before_2026"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", 2**62), id="future_timestamp_ns"),
        pytest.param(_keep_only_timestamp_ns, id="timestamp_ns_is_only_number"),
        pytest.param(lambda kpis: _set_record_value(kpis, "git_sha", ""), id="empty_git_sha"),
        pytest.param(lambda kpis: _set_record_value(kpis, "nested", {"a": 1}), id="nested_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "missing", None), id="null_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "flag", True), id="bool_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "rate", math.inf), id="infinite_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "env", "a\nb"), id="line_break_value"),
        pytest.param(_duplicate_record, id="duplicate_tag_set"),
        pytest.param(lambda kpis: _set_record_value(kpis, "_x", 1), id="underscore_prefixed_key"),
        pytest.param(lambda kpis: _set_record_value(kpis, "time", 1), id="time_key"),
    ],
)
def test_validation_helper_rejects_invalid_records(make_invalid):
    """The validation helper must catch invalid records, or tests that rely on it would pass vacuously."""
    kpis = _valid_records()
    make_invalid(kpis)

    with pytest.raises(AssertionError):
        assert_influxdb_kpi_records_are_valid(kpis)


def test_validation_helper_rejects_non_list_contents():
    with pytest.raises(AssertionError):
        assert_influxdb_kpi_records_are_valid({"records": _valid_records()})


def test_written_file_is_valid_influxdb_points(tmp_path):
    experiment_output_directory = _write_experiment(
        tmp_path / "experiment",
        {
            "a": {"episodes": _episode_records(4, 3)},
            "b": {"status": "failed"},
            "c": {"episodes": [{"env_id": 0, "episode_in_env": 0, "success": None, "episode_length": 5}]},
        },
        timings={
            "totals": [{"name": "step", "count": 2, "total_ms": 3.0, "min_ms": 1.0, "max_ms": 2.0, "mean_ms": 1.5}],
            "runs": [
                {"run_name": "a", "name": "step", "count": 2, "total_ms": 3.0, "min_ms": 1.0, "max_ms": 2.0},
            ],
        },
    )

    kpi_records_path = write_experiment_influxdb_kpi_records(experiment_output_directory)

    assert kpi_records_path == experiment_output_directory / ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME
    text = kpi_records_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    written_kpis = json.loads(text)
    assert_influxdb_kpi_records_are_valid(written_kpis)
    assert written_kpis == _build_kpi_records_list(experiment_output_directory)
