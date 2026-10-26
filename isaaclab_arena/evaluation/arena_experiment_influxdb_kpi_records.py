# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and write the InfluxDB KPI records file of a finished Arena Experiment.

This module allow us to track Arena's success rates and timings over time in a time-series database
such as InfluxDB. Note that module ONLY formats data for InfluxDB. Ingesting the file into a
database is not part of this repository.

InfluxDB tags and fields
------------------------
InfluxDB stores time series of records. A record has tags, which are indexed strings that identify the series it
belongs to and that queries filter and group by, and fields, which are the measured values (here always numbers).
A record is identified by its measurement, tag set and time, so a second record with the same identity overwrites
the first. A record is always flat and can only store scalars, i.e. no nested lists or dicts.

File format
-----------
The generated ``arena_experiment_influxdb_kpi_records.json`` holds a JSON list of flat records:

    [
      {"type": "per_experiment", "timestamp_ns": 1792800000000000000, "git_sha": "1a2b3c4", "git_branch": "main",
       "success_rate": 0.81, "num_episodes": 100, "num_runs": 2, "num_runs_completed": 2, "num_runs_failed": 0},
      {"type": "per_run", "timestamp_ns": 1792800000000000000, "git_sha": "1a2b3c4", "git_branch": "main",
       "run_name": "pnp_pi05", "environment": "droid_pnp", "policy_variant": "pi05",
       "success_rate": 0.82, "num_episodes": 50, "num_successes": 41, "mean_episode_length": 10.0, "completed": 1},
      {"type": "timing", "timestamp_ns": 1792800000000000000, "git_sha": "1a2b3c4", "git_branch": "main",
       "name": "rollout/step", "run_name": "all",
       "mean_ms": 10.0, "total_ms": 40.0, "min_ms": 5.0, "max_ms": 15.0, "count": 4}
    ]

Fields appearing in every record:

- ``timestamp_ns``: Record time in Unix epoch nanoseconds (UTC) of the latest run end.
- ``type``: Record type, see below.
- ``git_sha``: Arena git commit SHA the Experiment ran with.
- ``git_branch``: Arena git branch the Experiment ran with.


Record types, told apart by the ``type`` tag:

- ``per_experiment``: Totals over the Experiment. ``success_rate`` is episode-weighted over completed Runs
  and left out when no episode was scored.
- ``per_run``: One record per Run, tagged with ``run_name``, ``environment`` and ``policy_variant``.
  ``completed`` is 1 for a completed Run and 0 for a failed one; a failed Run has no other fields.
- ``timing``: Timer statistics in milliseconds, tagged with the timer ``name`` and the ``run_name``
  (``"all"`` for totals over every Run). Present only when the Experiment timings file exists.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ArenaExperimentResultData,
    ArenaExperimentRunData,
)
from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.evaluation.provenance import Provenance
from isaaclab_arena.visualization.report_data import EpisodeSummary, JobSummary, build_experiment_summary

ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME = "arena_experiment_influxdb_kpi_records.json"
"""Name of the KPI records file in an Experiment output directory."""

MIN_TIMESTAMP_NS = 1767225600 * 10**9
"""2026-01-01T00:00:00Z in Unix epoch nanoseconds.

Arena wrote no KPI records before this date, so an earlier timestamp is malformed, for example one in seconds
instead of nanoseconds.
"""

RECORD_TYPE_KEY = "type"
"""Key of the tag that holds a record's type."""

TIMESTAMP_NS_KEY = "timestamp_ns"
"""Key of a record's point time, the only numeric value that is not a field."""

ALL_RUNS_TIMING_RUN_NAME = "all"
"""``run_name`` tag of timing records summed over every Run."""


def _is_number(value: object) -> bool:
    """Return whether a value is an int or float; bool is not a number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: int | float) -> bool:
    """Return whether a number converts to a finite float."""
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_single_line_text(value: object) -> bool:
    """Return whether a value is a non-empty string without line breaks."""
    return isinstance(value, str) and bool(value) and "\n" not in value and "\r" not in value


@dataclass(kw_only=True)
class InfluxdbKpiRecord(ABC):
    """Abstract base class for KPI records.

    Each record is written as a flat JSON object for one InfluxDB point. Subclasses set ``record_type`` and add their own tags and fields. Attributes annotated as ``str`` are tags;
    the others, except ``timestamp_ns``, are numeric fields, left out when set to None.
    """

    timestamp_ns: int
    """Point time in Unix epoch nanoseconds (UTC); not a field."""

    git_sha: str
    """Arena git commit SHA the Experiment ran with."""

    git_branch: str
    """Arena git branch the Experiment ran with."""

    @property
    @abstractmethod
    def record_type(self) -> str:
        """Value of the record's ``type`` tag."""

    def __post_init__(self) -> None:
        # Check that the record can be written as one valid InfluxDB point: a plausible nanosecond timestamp,
        # non-empty single-line tags, finite numeric fields and at least one field.
        assert (
            isinstance(self.timestamp_ns, int)
            and not isinstance(self.timestamp_ns, bool)
            and self.timestamp_ns >= MIN_TIMESTAMP_NS
        ), (
            f"{self.record_type} record: timestamp_ns must be an integer no earlier than 2026-01-01T00:00:00Z,"
            f" got {self.timestamp_ns!r}"
        )
        has_field = False
        for record_field in fields(self):
            if record_field.name == TIMESTAMP_NS_KEY:
                continue
            value = getattr(self, record_field.name)
            # Annotations are strings because of ``from __future__ import annotations``.
            if record_field.type == "str":
                assert _is_single_line_text(value), (
                    f"{self.record_type} record: tag {record_field.name!r} must be a non-empty single-line string,"
                    f" got {value!r}"
                )
            elif value is not None:
                assert _is_number(value) and _is_finite_number(
                    value
                ), f"{self.record_type} record: field {record_field.name!r} must be a finite number, got {value!r}"
                has_field = True
        assert has_field, f"{self.record_type} record must have at least one numeric field"

    def to_flat_dict(self) -> dict[str, str | int | float]:
        """Return the record as a flat JSON object.

        The keys are ``type``, ``timestamp_ns``, ``git_sha`` and ``git_branch``, then every other attribute that is
        not None.
        """
        flat_dict: dict[str, str | int | float] = {RECORD_TYPE_KEY: self.record_type}
        for record_field in fields(self):
            value = getattr(self, record_field.name)
            if value is not None:
                flat_dict[record_field.name] = value
        return flat_dict

    def tag_set(self) -> frozenset[tuple[str, str]]:
        """Return the record's tags as ``(key, value)`` pairs, which identify its InfluxDB series."""
        return frozenset((key, value) for key, value in self.to_flat_dict().items() if isinstance(value, str))


def assert_unique_tag_sets(records: list[InfluxdbKpiRecord]) -> None:
    """Assert that no two records have the same tag set, so each record is a distinct InfluxDB point.

    Args:
        records: KPI records that share one timestamp.
    """
    record_index_by_tag_set: dict[frozenset[tuple[str, str]], int] = {}
    for record_index, record in enumerate(records):
        tag_set = record.tag_set()
        assert (
            tag_set not in record_index_by_tag_set
        ), f"record {record_index} has the same tags as record {record_index_by_tag_set.get(tag_set)}"
        record_index_by_tag_set[tag_set] = record_index


@dataclass(kw_only=True)
class PerExperimentRecord(InfluxdbKpiRecord):
    """KPIs totalled over every Run of the Experiment."""

    record_type: ClassVar[str] = "per_experiment"

    success_rate: float | None = None
    """Successes over scored episodes of completed Runs, or None when no episode was scored."""

    num_episodes: int
    """Number of episodes of completed Runs."""

    num_runs: int
    """Number of Runs in the Experiment."""

    num_runs_completed: int
    """Number of Runs that completed."""

    num_runs_failed: int
    """Number of Runs that failed."""


@dataclass(kw_only=True)
class PerRunRecord(InfluxdbKpiRecord):
    """KPIs of one Run; a failed Run has only ``completed=0``."""

    record_type: ClassVar[str] = "per_run"

    run_name: str
    """Name of the Run."""

    environment: str
    """Name of the Run's environment."""

    policy_variant: str
    """Policy variant evaluated by the Run."""

    success_rate: float | None = None
    """Successes over scored episodes, or None for a failed Run or when no episode was scored."""

    num_episodes: int | None = None
    """Number of episodes, or None for a failed Run."""

    num_successes: int | None = None
    """Number of successful episodes, or None for a failed Run."""

    mean_episode_length: float | None = None
    """Mean episode length in steps, or None when no episode records one."""

    mean_progress: float | None = None
    """Mean task progress fraction, or None when no episode records progress."""

    completed: int
    """1 when the Run completed, 0 when it failed."""


@dataclass(kw_only=True)
class TimingRecord(InfluxdbKpiRecord):
    """Statistics of one timer, for one Run or summed over every Run."""

    record_type: ClassVar[str] = "timing"

    name: str
    """Timer name."""

    run_name: str
    """Name of the Run, or ``"all"`` for totals over every Run."""

    mean_ms: float
    """Mean duration in milliseconds."""

    total_ms: float
    """Total duration in milliseconds."""

    min_ms: float
    """Shortest duration in milliseconds."""

    max_ms: float
    """Longest duration in milliseconds."""

    count: int
    """Number of timed intervals."""


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


def _build_per_run_record(
    run_name: str, run_result: ArenaExperimentRunData, pooled_job: JobSummary | None, shared_values: dict[str, Any]
) -> PerRunRecord:
    """Return the record of one Run; a failed Run (no pooled job) gets ``completed=0`` and no counts."""
    tags = {
        **shared_values,
        "run_name": run_name,
        "environment": run_result["environment"]["name"],
        "policy_variant": run_result["policy_variant"],
    }
    if pooled_job is None:
        return PerRunRecord(**tags, completed=0)
    return PerRunRecord(
        **tags,
        success_rate=pooled_job.success_rate,
        num_episodes=pooled_job.num_episodes,
        num_successes=pooled_job.num_successes,
        mean_episode_length=_mean_episode_length(pooled_job.episodes),
        mean_progress=pooled_job.mean_progress,
        completed=1,
    )


def _build_per_experiment_record(
    num_runs: int, completed_jobs: list[JobSummary], shared_values: dict[str, Any]
) -> PerExperimentRecord:
    """Return the per-Experiment record, with the success rate episode-weighted over completed Runs."""
    num_scored_episodes = sum(job.num_scored_episodes for job in completed_jobs)
    success_rate = (
        sum(job.num_successes for job in completed_jobs) / num_scored_episodes if num_scored_episodes > 0 else None
    )
    return PerExperimentRecord(
        **shared_values,
        success_rate=success_rate,
        num_episodes=sum(job.num_episodes for job in completed_jobs),
        num_runs=num_runs,
        num_runs_completed=len(completed_jobs),
        num_runs_failed=num_runs - len(completed_jobs),
    )


def _build_timing_record(timing: dict[str, Any], run_name: str, shared_values: dict[str, Any]) -> TimingRecord:
    """Return one timing record from an Experiment timings entry with a nonzero count."""
    count = int(timing["count"])
    total_ms = float(timing["total_ms"])
    return TimingRecord(
        **shared_values,
        name=timing["name"],
        run_name=run_name,
        mean_ms=total_ms / count,
        total_ms=total_ms,
        min_ms=float(timing["min_ms"]),
        max_ms=float(timing["max_ms"]),
        count=count,
    )


def _build_timing_records(
    experiment_output_directory: Path, run_names: set[str], shared_values: dict[str, Any]
) -> list[TimingRecord]:
    """Return timing records from the Experiment timings file, or none when the file is absent."""
    timings_path = experiment_output_directory / ARENA_AGGREGATED_TIMINGS_FILENAME
    if not timings_path.is_file():
        return []
    assert ALL_RUNS_TIMING_RUN_NAME not in run_names, (
        f"Run name {ALL_RUNS_TIMING_RUN_NAME!r} is reserved for timing totals in"
        f" {ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME}"
    )
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    # Totals over every Run first, then each Run's own timers; timers that never ran have no statistics.
    records = [
        _build_timing_record(timing, ALL_RUNS_TIMING_RUN_NAME, shared_values)
        for timing in timings["totals"]
        if timing["count"] > 0
    ]
    records.extend(
        _build_timing_record(timing, timing["run_name"], shared_values)
        for timing in timings["runs"]
        if timing["count"] > 0
    )
    return records


def build_experiment_influxdb_kpi_records(experiment_output_directory: Path) -> list[InfluxdbKpiRecord]:
    """Build the KPI records of a finished Experiment output directory.

    Reads ``arena_experiment_result.json``, the Runs' episode results and, when present,
    ``arena_experiment_timings.json``. The module docstring describes the resulting records.

    Args:
        experiment_output_directory: Experiment output root holding the result file and one directory per Run.

    Returns:
        The per-Experiment record, one per-Run record per Run sorted by Run name, then the timing records, all
        timestamped with the latest Run end and tagged with the Experiment's provenance.
    """
    experiment_output_directory = Path(experiment_output_directory)

    # Read the Experiment result and check that it holds what the records need.
    result: ArenaExperimentResultData = json.loads(
        (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    runs = result["runs"]
    assert runs, f"Experiment result in '{experiment_output_directory}' has no Runs"
    assert "provenance" in result, f"Experiment result in '{experiment_output_directory}' has no provenance"
    provenance = Provenance.from_dict(result["provenance"])
    for run_name, run_result in runs.items():
        end_timestamp_ns = run_result.get("end_timestamp_ns")
        assert isinstance(end_timestamp_ns, int) and not isinstance(end_timestamp_ns, bool), (
            f"Run {run_name!r} in '{experiment_output_directory}' has no integer end_timestamp_ns; results written"
            " before Run end times were recorded cannot produce KPI records"
        )

    # Pool each completed Run's episodes into one report job, so success counts use the report's definitions.
    run_names = set(runs)
    episodes_by_run_name = _episodes_by_run_name(experiment_output_directory, run_names)
    pooled_jobs_by_run_name = {
        run_name: JobSummary(
            name=run_name,
            task=run_name,
            policy=run_result["policy_variant"],
            cameras=[],
            episodes=episodes_by_run_name[run_name],
        )
        for run_name, run_result in runs.items()
        if run_result["status"] == RunStatus.COMPLETED.value
    }

    # Every record is a point at the latest Run end, tagged with the Experiment's provenance.
    shared_values = {
        TIMESTAMP_NS_KEY: max(run_result["end_timestamp_ns"] for run_result in runs.values()),
        **provenance.to_dict(),
    }

    records: list[InfluxdbKpiRecord] = [
        _build_per_experiment_record(len(runs), list(pooled_jobs_by_run_name.values()), shared_values)
    ]
    # Failed Runs have no pooled job, so their records carry no episode counts.
    records.extend(
        _build_per_run_record(run_name, runs[run_name], pooled_jobs_by_run_name.get(run_name), shared_values)
        for run_name in sorted(runs)
    )
    records.extend(_build_timing_records(experiment_output_directory, run_names, shared_values))
    assert_unique_tag_sets(records)
    return records


def write_experiment_influxdb_kpi_records(experiment_output_directory: Path) -> Path:
    """Build and write ``arena_experiment_influxdb_kpi_records.json`` into an Experiment output directory.

    Args:
        experiment_output_directory: Experiment output root holding ``arena_experiment_result.json``.

    Returns:
        Path to the written KPI records file.
    """
    experiment_output_directory = Path(experiment_output_directory)
    kpi_records = build_experiment_influxdb_kpi_records(experiment_output_directory)
    kpi_records_path = experiment_output_directory / ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME
    kpi_records_json = json.dumps([record.to_flat_dict() for record in kpi_records], allow_nan=False, indent=2)
    kpi_records_path.write_text(kpi_records_json + "\n", encoding="utf-8")
    print(f"Wrote Arena Experiment InfluxDB KPI records to: {kpi_records_path}")
    return kpi_records_path
