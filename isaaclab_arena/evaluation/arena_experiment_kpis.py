# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build, validate and write the KPI file of a finished Arena Experiment.

The KPI file holds flat records for dashboards: string values are tags and numeric values are fields.
Success counts come from the same report data model as the HTML report, so both always agree.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
)
from isaaclab_arena.visualization.report_data import EpisodeSummary, JobSummary, build_experiment_summary

ARENA_EXPERIMENT_KPIS_FILENAME = "arena_experiment_kpis.json"
KPI_SCHEMA_VERSION = 1

MIN_TIMESTAMP_NS = 1767225600 * 10**9
"""2026-01-01T00:00:00Z in Unix epoch nanoseconds; the KPI collector rejects earlier timestamps."""

ALL_RUNS_TIMING_RUN_NAME = "all"
"""``run_name`` tag of timing records summed over every Run."""

RESERVED_RECORD_KEYS = frozenset({"benchmark", "launch_time", "git_sha", "branch"})
"""Keys the KPI collector adds to every record, so records must not use them."""

PROVENANCE_KEYS = ("git_sha", "git_branch")


def _is_number(value: object) -> bool:
    """Return whether a value is an int or float; bool is not a number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _line_protocol_text_problem(text: str) -> str:
    """Return why a string cannot be a line-protocol key or tag value (empty or with a line break), or ''."""
    if not text:
        return "must not be empty"
    if "\n" in text or "\r" in text:
        return "must not contain a line break"
    return ""


def _key_problem(key: str) -> str:
    """Return why a record key cannot be an InfluxDB tag or field key, or ''."""
    problem = _line_protocol_text_problem(key)
    if problem:
        return problem
    if key.startswith("_"):
        return "must not start with '_' (reserved by InfluxDB)"
    if key == "time":
        return "must not be 'time' (reserved by InfluxDB)"
    return ""


def _is_finite_number(value: int | float) -> bool:
    """Return whether a number converts to a finite float."""
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _run_name_for_job(job_name: str, run_names: set[str]) -> str | None:
    """Return the Run whose output directory holds a report job's results, or None for files outside every Run."""
    job_path_parts = PurePosixPath(job_name).parts
    if job_path_parts and job_path_parts[0] in run_names:
        return job_path_parts[0]
    return None


def _episodes_by_run_name(experiment_output_directory: Path, run_names: set[str]) -> dict[str, list[EpisodeSummary]]:
    """Return the report's episodes of each Run, pooled over every results file under the Run's directory."""
    summary = build_experiment_summary(experiment_output_directory, experiment_output_directory.name)
    episodes_by_run_name: dict[str, list[EpisodeSummary]] = {run_name: [] for run_name in run_names}
    for job in summary.jobs:
        # Report job names are results-file directories relative to the Experiment output directory.
        # Run outputs normally sit directly in <experiment>/<run_name>; nested directories belong to that Run.
        run_name = _run_name_for_job(job.name, run_names)
        if run_name is not None:
            episodes_by_run_name[run_name].extend(job.episodes)
    return episodes_by_run_name


def _mean_episode_length(episodes: list[EpisodeSummary]) -> float | None:
    """Return the mean recorded ``episode_length``, or None when no episode records one."""
    episode_lengths = [
        episode.record["episode_length"] for episode in episodes if _is_number(episode.record.get("episode_length"))
    ]
    return None if not episode_lengths else sum(episode_lengths) / len(episode_lengths)


def _build_per_run_record(run_name: str, run_result: dict[str, Any], pooled_job: JobSummary | None) -> dict:
    """Return the ``per_run`` record of one Run; a failed Run (no pooled job) gets ``completed=0`` and no counts."""
    record: dict[str, Any] = {
        "type": "per_run",
        "run_name": run_name,
        "environment": run_result["environment"]["name"],
        "policy_variant": run_result["policy_variant"],
    }
    if pooled_job is None:
        record["completed"] = 0
        return record

    if pooled_job.success_rate is not None:
        record["success_rate"] = pooled_job.success_rate
    record["num_episodes"] = pooled_job.num_episodes
    record["num_successes"] = pooled_job.num_successes
    mean_episode_length = _mean_episode_length(pooled_job.episodes)
    if mean_episode_length is not None:
        record["mean_episode_length"] = mean_episode_length
    if pooled_job.mean_progress is not None:
        record["mean_progress"] = pooled_job.mean_progress
    record["completed"] = 1
    return record


def _build_per_experiment_record(num_runs: int, completed_jobs: list[JobSummary]) -> dict:
    """Return the ``per_experiment`` record, with the success rate episode-weighted over completed Runs."""
    record: dict[str, Any] = {"type": "per_experiment"}
    num_scored_episodes = sum(job.num_scored_episodes for job in completed_jobs)
    if num_scored_episodes > 0:
        record["success_rate"] = sum(job.num_successes for job in completed_jobs) / num_scored_episodes
    record["num_episodes"] = sum(job.num_episodes for job in completed_jobs)
    record["num_runs"] = num_runs
    record["num_runs_completed"] = len(completed_jobs)
    record["num_runs_failed"] = num_runs - len(completed_jobs)
    return record


def _build_timing_record(timing: dict[str, Any], run_name: str) -> dict:
    """Return one ``timing`` record from an Experiment timings entry with a nonzero count."""
    count = int(timing["count"])
    total_ms = float(timing["total_ms"])
    return {
        "type": "timing",
        "name": timing["name"],
        "run_name": run_name,
        "mean_ms": total_ms / count,
        "total_ms": total_ms,
        "min_ms": float(timing["min_ms"]),
        "max_ms": float(timing["max_ms"]),
        "count": count,
    }


def _build_timing_records(experiment_output_directory: Path, run_names: set[str]) -> list[dict]:
    """Return timing records from the Experiment timings file, or none when the file is absent."""
    timings_path = experiment_output_directory / ARENA_AGGREGATED_TIMINGS_FILENAME
    if not timings_path.is_file():
        return []
    assert (
        ALL_RUNS_TIMING_RUN_NAME not in run_names
    ), f"Run name {ALL_RUNS_TIMING_RUN_NAME!r} is reserved for timing totals in {ARENA_EXPERIMENT_KPIS_FILENAME}"
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    records = [
        _build_timing_record(timing, ALL_RUNS_TIMING_RUN_NAME) for timing in timings["totals"] if timing["count"] > 0
    ]
    records.extend(
        _build_timing_record(timing, timing["run_name"]) for timing in timings["runs"] if timing["count"] > 0
    )
    return records


def build_experiment_kpis(experiment_output_directory: Path) -> dict:
    """Build the KPI file contents of a finished Experiment output directory.

    Reads ``arena_experiment_result.json``, the Runs' episode results and, when present,
    ``arena_experiment_timings.json``. The timestamp is the earliest Run start.

    Args:
        experiment_output_directory: Experiment output root holding the result file and one directory per Run.

    Returns:
        JSON-compatible KPI file contents with ``per_experiment``, ``per_run`` and ``timing`` records.
    """
    experiment_output_directory = Path(experiment_output_directory)
    result = json.loads((experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8"))
    runs: dict[str, dict[str, Any]] = result["runs"]
    assert runs, f"Experiment result in '{experiment_output_directory}' has no Runs"
    assert "provenance" in result, f"Experiment result in '{experiment_output_directory}' has no provenance"
    missing_provenance_keys = [key for key in PROVENANCE_KEYS if key not in result["provenance"]]
    assert (
        not missing_provenance_keys
    ), f"Experiment result in '{experiment_output_directory}' has no provenance {missing_provenance_keys}"

    for run_name, run_result in runs.items():
        start_timestamp_ns = run_result.get("start_timestamp_ns")
        assert isinstance(start_timestamp_ns, int) and not isinstance(start_timestamp_ns, bool), (
            f"Run {run_name!r} in '{experiment_output_directory}' has no integer start_timestamp_ns; results written"
            " before Run start times were recorded cannot produce a KPI file"
        )

    run_names = set(runs)
    episodes_by_run_name = _episodes_by_run_name(experiment_output_directory, run_names)
    # Pool each completed Run's episodes into one report job, so success counts use the report's definitions.
    pooled_jobs_by_run_name = {
        run_name: JobSummary(
            name=run_name,
            task=run_name,
            policy=run_result["policy_variant"],
            cameras=[],
            episodes=episodes_by_run_name[run_name],
        )
        for run_name, run_result in runs.items()
        if run_result["status"] == "completed"
    }
    per_run_records = [
        _build_per_run_record(run_name, runs[run_name], pooled_jobs_by_run_name.get(run_name))
        for run_name in sorted(runs)
    ]
    per_experiment_record = _build_per_experiment_record(len(runs), list(pooled_jobs_by_run_name.values()))

    return {
        "schema_version": KPI_SCHEMA_VERSION,
        "timestamp_ns": min(run_result["start_timestamp_ns"] for run_result in runs.values()),
        "provenance": {key: result["provenance"][key] for key in PROVENANCE_KEYS},
        "records": [
            per_experiment_record,
            *per_run_records,
            *_build_timing_records(experiment_output_directory, run_names),
        ],
    }


def _validate_record(index: int, record: object) -> frozenset:
    """Check one KPI record and return its tag set: the record's string-valued items."""
    assert isinstance(record, dict), f"record {index} must be a JSON object"
    assert isinstance(record.get("type"), str), f"record {index} must have a string type"
    has_number = False
    for key, value in record.items():
        assert key not in RESERVED_RECORD_KEYS, f"record {index}: key {key!r} is reserved"
        assert not _key_problem(key), f"record {index}: key {key!r} {_key_problem(key)}"
        if _is_number(value):
            assert _is_finite_number(value), f"record {index}: value for {key!r} must be finite"
            has_number = True
        else:
            assert isinstance(value, str), f"record {index}: value for {key!r} must be a string or a number"
            assert not _line_protocol_text_problem(
                value
            ), f"record {index}: value for {key!r} {_line_protocol_text_problem(value)}"
    assert has_number, f"record {index} must have at least one numeric value"
    return frozenset((key, value) for key, value in record.items() if isinstance(value, str))


def validate_experiment_kpis(kpis: dict) -> None:
    """Assert that KPI file contents follow the KPI file contract.

    Args:
        kpis: Parsed or built KPI file contents.
    """
    assert isinstance(kpis, dict), "KPI file contents must be a JSON object"
    schema_version = kpis.get("schema_version")
    assert (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == KPI_SCHEMA_VERSION
    ), f"schema_version must be {KPI_SCHEMA_VERSION}, got {schema_version!r}"

    assert "timestamp_ns" in kpis, "timestamp_ns is required"
    timestamp_ns = kpis["timestamp_ns"]
    assert (
        isinstance(timestamp_ns, int) and not isinstance(timestamp_ns, bool) and timestamp_ns >= MIN_TIMESTAMP_NS
    ), f"timestamp_ns must be an integer no earlier than 2026-01-01T00:00:00Z, got {timestamp_ns!r}"

    provenance = kpis.get("provenance")
    assert isinstance(provenance, dict), "provenance must be a JSON object"
    for key in PROVENANCE_KEYS:
        assert key in provenance, f"provenance.{key} is required"
        value = provenance[key]
        assert isinstance(value, str), f"provenance.{key} must be a string, got {value!r}"
        assert not _line_protocol_text_problem(value), f"provenance.{key} {_line_protocol_text_problem(value)}"

    records = kpis.get("records")
    assert isinstance(records, list) and records, "records must be a non-empty list"
    index_by_tag_set: dict[frozenset, int] = {}
    for index, record in enumerate(records):
        tag_set = _validate_record(index, record)
        assert (
            tag_set not in index_by_tag_set
        ), f"record {index} has the same tag set as record {index_by_tag_set.get(tag_set)}"
        index_by_tag_set[tag_set] = index


def write_experiment_kpis(experiment_output_directory: Path) -> Path:
    """Build, validate and write ``arena_experiment_kpis.json`` into an Experiment output directory.

    Args:
        experiment_output_directory: Experiment output root holding ``arena_experiment_result.json``.

    Returns:
        Path to the written KPI file.
    """
    experiment_output_directory = Path(experiment_output_directory)
    kpis = build_experiment_kpis(experiment_output_directory)
    validate_experiment_kpis(kpis)
    kpis_path = experiment_output_directory / ARENA_EXPERIMENT_KPIS_FILENAME
    kpis_path.write_text(json.dumps(kpis, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Arena Experiment KPIs to: {kpis_path}")
    return kpis_path
