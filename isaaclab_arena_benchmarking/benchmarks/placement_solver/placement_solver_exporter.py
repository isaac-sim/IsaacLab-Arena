# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the placement solver KPIs of a finished Arena Experiment as an InfluxDB KPI record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME,
    ArenaExperimentResult,
)
from isaaclab_arena.relations.placement_diagnostics import AggregatedPlacementDiagnostics
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena_benchmarking.benchmarks.registry import register_kpi_exporter
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter, InfluxdbKpiRecord

PLACEMENT_SOLVER_RECORDS_FILENAME = "kpis_placement_solver.json"
"""Name of the placement solver KPI records file in an Experiment output directory."""


@dataclass
class RunPlacementCounts:
    """Placement counts of one Run, from the placement diagnostics that it wrote."""

    run_name: str
    """Name of the Run in its Experiment."""

    placement_diagnostics: AggregatedPlacementDiagnostics | None = None
    """Placement diagnostics of the Run's builds, or None for a Run that wrote none, because it solved no relations
    or replayed recorded layouts."""

    @property
    def valid_candidate_rate_first_batch(self) -> float:
        """Share of first-batch candidates that are valid.

        The first batch has the same size in every build with the same num_envs. A later batch runs only while some
        environment still has empty slots, and its size depends on how many are missing, so counting it would give
        scenes that need more batches a larger sample.
        """
        first_batch = self.placement_diagnostics.batches[0]
        return first_batch.num_valid / first_batch.num_candidates

    @property
    def valid_env_rate_after_retries(self) -> float:
        """Share of environments whose pool holds only valid layouts, without a best-loss fallback."""
        return 1 - self.placement_diagnostics.num_fallback_envs / self.placement_diagnostics.num_envs

    @property
    def batches_per_build(self) -> float:
        """Mean number of pool batches a build solved."""
        return self.placement_diagnostics.num_batches / self.placement_diagnostics.num_builds


@dataclass(kw_only=True)
class PlacementSolverRecord(InfluxdbKpiRecord):
    """Placement solver KPIs of the Experiment, the mean over its scored Runs."""

    record_type: ClassVar[str] = "placement_solver"

    valid_candidate_rate_first_batch: float
    """Mean over the scored Runs of the share of first-batch candidates that pass every required check."""

    valid_env_rate_after_retries: float
    """Mean over the scored Runs of the share of environments filled with valid layouts, without best-loss fallback."""

    batches_per_build: float
    """Mean over the scored Runs of the number of pool batches a build solved."""

    num_scored_runs: int
    """Number of Runs that wrote placement diagnostics."""

    num_runs: int
    """Number of Runs in the Experiment."""


def read_run_placement_counts(output_dir: Path) -> list[RunPlacementCounts]:
    """Return the placement counts of each Run of an Experiment, in Experiment order.

    Args:
        output_dir: Experiment output root holding the result file and one directory per Run.
    """
    result = ArenaExperimentResult.from_experiment_output(output_dir)
    counts_per_run = []
    for run_name in result.runs:
        diagnostics_path = Path(output_dir) / run_name / ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME
        placement_diagnostics = (
            AggregatedPlacementDiagnostics.load(diagnostics_path) if diagnostics_path.is_file() else None
        )
        counts_per_run.append(RunPlacementCounts(run_name, placement_diagnostics))
    return counts_per_run


def build_placement_solver_record(
    counts_per_run: list[RunPlacementCounts], *, timestamp_ns: int, git_stamp: GitStamp, benchmark_id: str
) -> PlacementSolverRecord | None:
    """Return the record of an Experiment, or None when no Run wrote placement diagnostics.

    Runs without diagnostics are left out of the means. Every scored Run has the same weight.

    Args:
        counts_per_run: Placement counts of the Experiment's Runs.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.
        benchmark_id: Benchmark the Experiment belongs to.
    """
    scored = [run_counts for run_counts in counts_per_run if run_counts.placement_diagnostics is not None]
    if not scored:
        return None
    return PlacementSolverRecord(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        benchmark_id=benchmark_id,
        valid_candidate_rate_first_batch=_mean([run_counts.valid_candidate_rate_first_batch for run_counts in scored]),
        valid_env_rate_after_retries=_mean([run_counts.valid_env_rate_after_retries for run_counts in scored]),
        batches_per_build=_mean([run_counts.batches_per_build for run_counts in scored]),
        num_scored_runs=len(scored),
        num_runs=len(counts_per_run),
    )


@register_kpi_exporter("placement_solver")
class PlacementSolverKpiExporter(InfluxdbKpiExporter):
    """Exports the placement solver KPIs of a finished Arena Experiment output directory."""

    record_types: ClassVar[tuple[str, ...]] = (PlacementSolverRecord.record_type,)

    def get_records_filename(self) -> str:
        return PLACEMENT_SOLVER_RECORDS_FILENAME

    def export_to_json(self, output_dir: Path) -> Path | None:
        """Write the placement solver record and print the placement rates of each Run.

        Args:
            output_dir: Experiment output root holding the result file and one directory per Run.

        Returns:
            Path of the records file, or None when no Run wrote placement diagnostics.
        """
        written_path = super().export_to_json(output_dir)
        self._print_summary(output_dir, failed_run_names=[])
        return written_path

    def on_experiment_incomplete(self, output_dir: Path, failed_run_names: list[str]) -> None:
        """Print the placement rates of the Runs that completed; the failed Runs are named, not scored.

        Args:
            output_dir: Experiment output root holding the result file and one directory per Run.
            failed_run_names: Names of the Runs that did not complete.
        """
        self._print_summary(output_dir, failed_run_names)

    def _print_summary(self, output_dir: Path, failed_run_names: list[str]) -> None:
        print("Placement solver KPIs per Run:")
        print(_format_run_placement_counts(read_run_placement_counts(Path(output_dir)), failed_run_names))

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the placement solver record of a finished Experiment output directory.

        Reads ``arena_experiment_result.json`` for the Runs, timestamp, git stamp and Experiment name, and each
        Run's ``arena_run_placement_diagnostics.json``.

        Args:
            output_dir: Experiment output root holding the result file and one directory per Run.

        Returns:
            One record over the Experiment, or none when no Run wrote placement diagnostics.
        """
        output_dir = Path(output_dir)
        result = ArenaExperimentResult.from_experiment_output(output_dir)
        record = build_placement_solver_record(
            read_run_placement_counts(output_dir),
            timestamp_ns=result.latest_run_end_timestamp_ns,
            git_stamp=result.git_stamp,
            benchmark_id=result.experiment_name,
        )
        return [] if record is None else [record]


def _format_run_placement_counts(counts_per_run: list[RunPlacementCounts], failed_run_names: list[str]) -> str:
    """Return one line per Run with its placement rates, the failed Runs and the Runs without diagnostics.

    Args:
        counts_per_run: Placement counts of the Experiment's Runs.
        failed_run_names: Names of the Runs that did not complete; their diagnostics are not used.
    """
    lines = []
    for run_counts in counts_per_run:
        if run_counts.run_name in failed_run_names:
            lines.append(f"  {run_counts.run_name}: failed")
        elif run_counts.placement_diagnostics is None:
            lines.append(f"  {run_counts.run_name}: no placement diagnostics")
        else:
            lines.append(
                f"  {run_counts.run_name}: valid candidates {100 * run_counts.valid_candidate_rate_first_batch:.1f}%,"
                f" valid environments {100 * run_counts.valid_env_rate_after_retries:.1f}%,"
                f" {run_counts.batches_per_build:.2f} batches per build"
            )
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
