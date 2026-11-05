# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the placement success KPIs of an Arena Experiment from the placement diagnostics that its Runs wrote.

Print the KPIs of an Experiment output directory, without the simulator:

    python -m isaaclab_arena_benchmarking.object_placement.compute_placement_kpis_from_experiment_output \\
        <experiment_output_directory>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME,
)
from isaaclab_arena.relations.placement_diagnostics import AggregatedPlacementDiagnostics


@dataclass
class RunPlacementCounts:
    """Placement counts of one Run, from the placement diagnostics that it wrote."""

    run_name: str
    """Name of the Run in its Experiment."""

    failed: bool = False
    """Whether the Run failed, for example because its placement pool could not fill every environment."""

    placement_diagnostics: AggregatedPlacementDiagnostics | None = None
    """Placement diagnostics of the Run's builds, or None for a Run that wrote none."""

    @property
    def is_scored(self) -> bool:
        """Whether the Run completed and wrote placement diagnostics. A failed Run's diagnostics are not scored."""
        return not self.failed and self.placement_diagnostics is not None

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


@dataclass
class PlacementSuccessSummary:
    """Placement success of an Experiment, with every scored Run weighted equally."""

    num_scored: int
    """Runs that completed and wrote placement diagnostics."""

    num_failed: int
    """Runs that failed."""

    valid_candidate_rate_first_batch: float | None
    """Mean valid-candidate rate over scored Runs, or None when none were scored or any failed."""

    valid_env_rate_after_retries: float | None
    """Mean valid-environment rate over scored Runs, or None when none were scored or any failed."""

    batches_per_build: float | None
    """Mean pool batches per build over scored Runs, or None when none were scored or any failed."""


def read_run_placement_counts(experiment_output_directory: str | Path) -> list[RunPlacementCounts]:
    """Return the placement counts of each Run of an Experiment, in Experiment order.

    Args:
        experiment_output_directory: Directory that experiment_runner.py wrote the Experiment result to.
    """
    experiment_output_directory = Path(experiment_output_directory)
    result_path = experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME
    experiment_result = json.loads(result_path.read_text(encoding="utf-8"))
    counts_per_run = []
    for run_name, run in experiment_result["runs"].items():
        diagnostics_path = experiment_output_directory / run_name / ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME
        placement_diagnostics = (
            AggregatedPlacementDiagnostics.load(diagnostics_path) if diagnostics_path.is_file() else None
        )
        counts_per_run.append(RunPlacementCounts(run_name, run["status"] == "failed", placement_diagnostics))
    return counts_per_run


def summarize_placement_success(counts_per_run: list[RunPlacementCounts]) -> PlacementSuccessSummary:
    """Return the mean placement success over the scored Runs.

    Any failed Run makes every rate None, since the failed Run could have lowered them. Runs that completed without
    placement diagnostics solved no relations or replayed recorded layouts, and are left out.

    Args:
        counts_per_run: Placement counts of the Experiment's Runs.
    """
    scored = [run_counts for run_counts in counts_per_run if run_counts.is_scored]
    num_failed = sum(run_counts.failed for run_counts in counts_per_run)
    # The mean over no Runs is None.
    rated = [] if num_failed else scored
    return PlacementSuccessSummary(
        num_scored=len(scored),
        num_failed=num_failed,
        valid_candidate_rate_first_batch=_mean([run_counts.valid_candidate_rate_first_batch for run_counts in rated]),
        valid_env_rate_after_retries=_mean([run_counts.valid_env_rate_after_retries for run_counts in rated]),
        batches_per_build=_mean([run_counts.batches_per_build for run_counts in rated]),
    )


def format_placement_success(counts_per_run: list[RunPlacementCounts]) -> str:
    """Return a Markdown table with one row per Run and a last row with the Experiment's means.

    Args:
        counts_per_run: Placement counts of the Experiment's Runs.
    """
    lines = ["| run | valid candidates | valid environments | batches per build |", "|---|---|---|---|"]
    for run_counts in counts_per_run:
        if run_counts.is_scored:
            lines.append(
                f"| {run_counts.run_name} | {_format_percent(run_counts.valid_candidate_rate_first_batch)} |"
                f" {_format_percent(run_counts.valid_env_rate_after_retries)} |"
                f" {_format_number(run_counts.batches_per_build)} |"
            )
        else:
            outcome = "failed" if run_counts.failed else "no diagnostics"
            lines.append(f"| {run_counts.run_name} | {outcome} | {outcome} | {outcome} |")
    summary = summarize_placement_success(counts_per_run)
    lines.append(
        f"| mean of {summary.num_scored} scored, {summary.num_failed} failed |"
        f" {_format_percent(summary.valid_candidate_rate_first_batch)} |"
        f" {_format_percent(summary.valid_env_rate_after_retries)} | {_format_number(summary.batches_per_build)} |"
    )
    return "\n".join(lines)


def main() -> None:
    """Print the placement success KPIs of the Experiment output directory given on the command line."""
    parser = argparse.ArgumentParser(description="Print the placement success KPIs of an Arena Experiment.")
    parser.add_argument("experiment_output_directory", type=Path, help="Directory that experiment_runner.py wrote.")
    args_cli = parser.parse_args()
    counts_per_run = read_run_placement_counts(args_cli.experiment_output_directory)
    print(format_placement_success(counts_per_run))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


if __name__ == "__main__":
    main()
