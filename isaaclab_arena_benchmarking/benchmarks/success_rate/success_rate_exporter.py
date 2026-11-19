# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the success rate of a finished Arena Experiment as an InfluxDB KPI record.

The records file is ``kpis_success_rates.json``. It holds one ``per_experiment`` record with the episode-weighted
success rate over every Run, timestamped with the latest Run end and tagged with the Arena git commit SHA, branch
and benchmark id (the Experiment name).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena.evaluation.report_data import EpisodeSummary, JobSummary, build_experiment_summary
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena_benchmarking.benchmarks.registry import register_kpi_exporter
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter, InfluxdbKpiRecord

SUCCESS_RATE_RECORDS_FILENAME = "kpis_success_rates.json"
"""Name of the success-rate KPI records file in an Experiment output directory."""


@dataclass(kw_only=True)
class PerExperimentRecord(InfluxdbKpiRecord):
    """Success rate and episode count totalled over every Run of the Experiment."""

    record_type: ClassVar[str] = "per_experiment"

    success_rate: float | None = None
    """Successes over scored episodes, or None when no episode was scored."""

    num_episodes: int
    """Number of episodes."""

    num_runs: int
    """Number of Runs in the Experiment."""


def build_per_experiment_record(
    num_runs: int, jobs: list[JobSummary], *, timestamp_ns: int, git_stamp: GitStamp, benchmark_id: str
) -> PerExperimentRecord:
    """Return the per-Experiment record, with the success rate episode-weighted over the given jobs.

    Args:
        num_runs: Number of Runs in the Experiment.
        jobs: One report job per Run, pooling that Run's episodes.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.
        benchmark_id: Benchmark the Experiment belongs to.

    Returns:
        The per-Experiment record.
    """
    num_scored_episodes = sum(job.num_scored_episodes for job in jobs)
    success_rate = sum(job.num_successes for job in jobs) / num_scored_episodes if num_scored_episodes > 0 else None
    return PerExperimentRecord(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        benchmark_id=benchmark_id,
        success_rate=success_rate,
        num_episodes=sum(job.num_episodes for job in jobs),
        num_runs=num_runs,
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


@register_kpi_exporter("success_rate")
class SuccessRateKpiExporter(InfluxdbKpiExporter):
    """Exports the success rate of a finished Arena Experiment output directory.

    The exporter pools every Run in the result and does not check Run statuses.
    """

    record_types: ClassVar[tuple[str, ...]] = (PerExperimentRecord.record_type,)

    def get_records_filename(self) -> str:
        return SUCCESS_RATE_RECORDS_FILENAME

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the success-rate record of a finished Experiment output directory.

        Reads ``arena_experiment_result.json`` and the Runs' episode results.

        Args:
            output_dir: Experiment output root holding the result file and one directory per Run.

        Returns:
            One per-Experiment record over every Run, timestamped with the latest Run end and tagged with the
            Experiment's git stamp and name, the latter as the benchmark id.
        """
        output_dir = Path(output_dir)
        result = ArenaExperimentResult.from_experiment_output(output_dir)
        runs = result.runs

        # Pool each Run's episodes into one report job, so success counts use the report's definitions.
        episodes_by_run_name = _episodes_by_run_name(output_dir, set(runs))
        jobs = [
            JobSummary(
                name=run_name,
                task=run_name,
                policy=run_result["policy_variant"],
                cameras=[],
                episodes=episodes_by_run_name[run_name],
            )
            for run_name, run_result in runs.items()
        ]
        records: list[InfluxdbKpiRecord] = [
            build_per_experiment_record(
                len(runs),
                jobs,
                timestamp_ns=result.latest_run_end_timestamp_ns,
                git_stamp=result.git_stamp,
                benchmark_id=result.experiment_name,
            )
        ]
        return records
