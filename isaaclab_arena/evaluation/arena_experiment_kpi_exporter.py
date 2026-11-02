# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and write the InfluxDB KPI records file of a finished Arena Experiment.

This module allow us to track Arena's success rates and timings over time in a time-series database
such as InfluxDB. Note that module ONLY formats data for InfluxDB. Ingesting the file into a
database is not part of this repository. The generic record and exporter classes live in
``isaaclab_arena_benchmarking.utils.influxdb_kpi_records``.

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

Every record is timestamped with the latest Run end and tagged with the Arena git commit SHA and branch the
Experiment ran with.

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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import (
    ALL_RUNS_TIMING_RUN_NAME,
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ArenaExperimentResultData,
    ArenaExperimentRunData,
)
from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena.visualization.report_data import EpisodeSummary, JobSummary, build_experiment_summary
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import (
    InfluxdbKpiExporter,
    InfluxdbKpiRecord,
    TimingRecord,
    _is_number,
    assert_unique_tag_sets,
)

ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME = "arena_experiment_influxdb_kpi_records.json"
"""Name of the KPI records file in an Experiment output directory."""


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


def build_per_experiment_record(
    num_runs: int, completed_jobs: list[JobSummary], *, timestamp_ns: int, git_stamp: GitStamp
) -> PerExperimentRecord:
    """Return the per-Experiment record, with the success rate episode-weighted over completed Runs.

    Args:
        num_runs: Number of Runs in the Experiment, completed or failed.
        completed_jobs: One report job per completed Run, pooling that Run's episodes.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.

    Returns:
        The per-Experiment record.
    """
    num_scored_episodes = sum(job.num_scored_episodes for job in completed_jobs)
    success_rate = (
        sum(job.num_successes for job in completed_jobs) / num_scored_episodes if num_scored_episodes > 0 else None
    )
    return PerExperimentRecord(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        success_rate=success_rate,
        num_episodes=sum(job.num_episodes for job in completed_jobs),
        num_runs=num_runs,
        num_runs_completed=len(completed_jobs),
        num_runs_failed=num_runs - len(completed_jobs),
    )


def build_per_run_record(
    run_name: str,
    run_result: ArenaExperimentRunData,
    pooled_job: JobSummary | None,
    *,
    timestamp_ns: int,
    git_stamp: GitStamp,
) -> PerRunRecord:
    """Return the record of one Run; a failed Run (no pooled job) gets ``completed=0`` and no counts.

    Args:
        run_name: Name of the Run.
        run_result: The Run's entry in the Experiment result.
        pooled_job: Report job pooling the Run's episodes, or None for a failed Run.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.

    Returns:
        The per-Run record.
    """
    shared_values = {
        "timestamp_ns": timestamp_ns,
        "git_sha": git_stamp.git_sha,
        "git_branch": git_stamp.git_branch,
        "run_name": run_name,
        "environment": run_result["environment"]["name"],
        "policy_variant": run_result["policy_variant"],
    }
    if pooled_job is None:
        return PerRunRecord(**shared_values, completed=0)
    return PerRunRecord(
        **shared_values,
        success_rate=pooled_job.success_rate,
        num_episodes=pooled_job.num_episodes,
        num_successes=pooled_job.num_successes,
        mean_episode_length=_mean_episode_length(pooled_job.episodes),
        mean_progress=pooled_job.mean_progress,
        completed=1,
    )


def build_timing_record(
    timing: dict[str, Any], run_name: str, *, timestamp_ns: int, git_stamp: GitStamp
) -> TimingRecord:
    """Return the record of one Experiment timings entry; the entry must have a nonzero count.

    Args:
        timing: Timer statistics with ``name``, ``count``, ``total_ms``, ``min_ms`` and ``max_ms``.
        run_name: Name of the Run the statistics belong to, or ``"all"`` for totals over every Run.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.

    Returns:
        The timing record, with the mean computed from the total and count.
    """
    count = int(timing["count"])
    total_ms = float(timing["total_ms"])
    return TimingRecord(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        name=timing["name"],
        run_name=run_name,
        mean_ms=total_ms / count,
        total_ms=total_ms,
        min_ms=float(timing["min_ms"]),
        max_ms=float(timing["max_ms"]),
        count=count,
    )


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


def _build_timing_records(
    experiment_output_directory: Path, *, timestamp_ns: int, git_stamp: GitStamp
) -> list[TimingRecord]:
    """Return timing records from the Experiment timings file, or none when the file is absent.

    Args:
        experiment_output_directory: Experiment output root that may hold ``arena_experiment_timings.json``.
        timestamp_ns: Point time of every record in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.

    Returns:
        Timing records summed over every Run first, then each Run's own, skipping timers that never ran.
    """
    timings_path = experiment_output_directory / ARENA_AGGREGATED_TIMINGS_FILENAME
    if not timings_path.is_file():
        return []
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    # Totals over every Run first, then each Run's own timers; timers that never ran have no statistics.
    records = [
        build_timing_record(timing, ALL_RUNS_TIMING_RUN_NAME, timestamp_ns=timestamp_ns, git_stamp=git_stamp)
        for timing in timings["totals"]
        if timing["count"] > 0
    ]
    records.extend(
        build_timing_record(timing, timing["run_name"], timestamp_ns=timestamp_ns, git_stamp=git_stamp)
        for timing in timings["runs"]
        if timing["count"] > 0
    )
    return records


class ArenaExperimentKpiExporter(InfluxdbKpiExporter):
    """Exports the KPI records of a finished Arena Experiment output directory."""

    def get_records_filename(self) -> str:
        return ARENA_EXPERIMENT_INFLUXDB_KPI_RECORDS_FILENAME

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the KPI records of a finished Experiment output directory.

        Reads ``arena_experiment_result.json``, the Runs' episode results and, when present,
        ``arena_experiment_timings.json``. The module docstring describes the resulting records.

        Args:
            output_dir: Experiment output root holding the result file and one directory per Run.

        Returns:
            The per-Experiment record, one per-Run record per Run sorted by Run name, then the timing records, all
            timestamped with the latest Run end and tagged with the Experiment's git stamp.
        """
        output_dir = Path(output_dir)

        # Read the Experiment result and check that it holds what the records need.
        result: ArenaExperimentResultData = json.loads(
            (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8")
        )
        runs = result["runs"]
        assert runs, f"Experiment result in '{output_dir}' has no Runs"
        assert "git_stamp" in result, f"Experiment result in '{output_dir}' has no git stamp"
        git_stamp = GitStamp.from_dict(result["git_stamp"])
        for run_name, run_result in runs.items():
            end_timestamp_ns = run_result.get("end_timestamp_ns")
            assert isinstance(end_timestamp_ns, int) and not isinstance(end_timestamp_ns, bool), (
                f"Run {run_name!r} in '{output_dir}' has no integer end_timestamp_ns; results written"
                " before Run end times were recorded cannot produce KPI records"
            )

        # Pool each completed Run's episodes into one report job, so success counts use the report's definitions.
        run_names = set(runs)
        episodes_by_run_name = _episodes_by_run_name(output_dir, run_names)
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

        # Every record is a point at the latest Run end, tagged with the Experiment's git stamp.
        timestamp_ns = max(run_result["end_timestamp_ns"] for run_result in runs.values())

        records: list[InfluxdbKpiRecord] = [
            build_per_experiment_record(
                len(runs), list(pooled_jobs_by_run_name.values()), timestamp_ns=timestamp_ns, git_stamp=git_stamp
            )
        ]
        # Failed Runs have no pooled job, so their records carry no episode counts.
        records.extend(
            build_per_run_record(
                run_name,
                runs[run_name],
                pooled_jobs_by_run_name.get(run_name),
                timestamp_ns=timestamp_ns,
                git_stamp=git_stamp,
            )
            for run_name in sorted(runs)
        )
        records.extend(_build_timing_records(output_dir, timestamp_ns=timestamp_ns, git_stamp=git_stamp))
        assert_unique_tag_sets(records)
        return records
