# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement-success benchmark results: results files, suite summaries and run comparisons.

A results file holds the start time, the commit and branch, the settings, a flat list of metrics and one record per
environment, so that two runs can be compared.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

_NORMAL_QUANTILE_95 = 1.959963984540054
"""Two-sided 95% quantile of the standard normal distribution."""

_FAMILY_ALPHA = 0.05
"""Family-wise error rate of the Holm adjustment over every environment and rate."""

_RESULTS_FILE_KEYS = {"start_time", "commit", "branch", "dirty", "settings", "metrics", "environments"}


class EnvironmentStatus(StrEnum):
    """Outcome of benchmarking one environment spec."""

    SCORED = "scored"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class EnvironmentRecord:
    """Placement outcome for one environment spec, accumulated over all seeds."""

    suite: str
    """Suite the spec belongs to, such as robolab or kitchen_bench."""

    environment: str
    """Spec file stem."""

    status: EnvironmentStatus
    """Whether the environment was scored, skipped or failed to run."""

    reason: str = ""
    """Why the environment was skipped or failed; empty when it was scored."""

    num_layouts: int = 0
    """Layouts solved; each layout keeps the best of its candidates."""

    layout_successes: int = 0
    """Layouts whose best candidate passed every required check."""

    num_candidates: int = 0
    """Candidates solved."""

    candidate_successes: int = 0
    """Candidates that passed every required check."""

    failed_checks: dict[str, int] = field(default_factory=dict)
    """Failed check names, including optional checks, counted over the best candidates of failed layouts."""

    checks: dict[str, bool] = field(default_factory=dict)
    """Checks that ran for this environment, mapped to whether they are required for success."""

    candidate_failures_per_check: dict[str, int] = field(default_factory=dict)
    """Candidates rejected by each check, including optional checks."""

    wall_time_s: float = 0.0
    """Time to prepare the placement inputs and place every seed."""

    @property
    def layout_success_rate(self) -> float:
        """Share of layouts whose best candidate is valid."""
        return self.layout_successes / self.num_layouts

    @property
    def valid_candidate_rate(self) -> float:
        """Share of candidates that are valid."""
        return self.candidate_successes / self.num_candidates


@dataclass
class PlacementBenchmarkRun:
    """One benchmark invocation: its provenance, settings and one record per environment."""

    start_time: str
    """UTC start time in ISO 8601 format."""

    commit: str | None
    """Git commit of the measured code, or None when unknown."""

    branch: str | None
    """Git branch of the measured code, or None when unknown."""

    dirty: bool | None
    """Whether the checkout had uncommitted changes, or None when unknown."""

    settings: dict[str, object]
    """Benchmark arguments and solver device, shared by every environment."""

    environments: list[EnvironmentRecord]
    """Records in discovery order."""

    def get_metric_records(self) -> list[dict[str, object]]:
        """Return a flat list of metrics for each suite and each scored environment.

        A suite has no rate metrics when none of its environments were scored or any of them failed.
        """
        metrics = []
        for summary in summarize_suites(self):
            tags = {"suite": summary.suite}
            metrics.append(_metric("environments_scored", summary.scored, tags))
            metrics.append(_metric("environments_skipped", summary.skipped, tags))
            metrics.append(_metric("environments_failed", summary.failed, tags))
            metrics.append(_metric("wall_time_s", summary.wall_time_s, tags))
            if summary.layout_success_rate is not None:
                metrics.append(_metric("layout_success_rate", summary.layout_success_rate, tags))
                metrics.append(_metric("valid_candidate_rate", summary.valid_candidate_rate, tags))
        for record in self.environments:
            if record.status != EnvironmentStatus.SCORED:
                continue
            tags = {"suite": record.suite, "environment": record.environment}
            metrics.append(_metric("layout_success_rate", record.layout_success_rate, tags))
            metrics.append(_metric("valid_candidate_rate", record.valid_candidate_rate, tags))
        return metrics

    def save(self, path: str | Path) -> None:
        """Write the run, its metrics and its environment records as JSON.

        Args:
            path: File to write, replacing any existing file.
        """
        content = {
            "start_time": self.start_time,
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "settings": self.settings,
            "metrics": self.get_metric_records(),
            "environments": [asdict(record) for record in self.environments],
        }
        # Write a sibling file and rename it, so a run stopped mid-write keeps the previous results.
        temporary_path = Path(path).with_name(Path(path).name + ".partial")
        temporary_path.write_text(json.dumps(content, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> PlacementBenchmarkRun:
        """Read a run written by save().

        Args:
            path: Results file to read.
        """
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        assert (
            set(content) == _RESULTS_FILE_KEYS
        ), f"{path} is not a results file written by this version of the benchmark; rerun the benchmark to update it"
        environments = []
        for record in content["environments"]:
            record["status"] = EnvironmentStatus(record["status"])
            environments.append(EnvironmentRecord(**record))
        return cls(
            start_time=content["start_time"],
            commit=content["commit"],
            branch=content["branch"],
            dirty=content["dirty"],
            settings=content["settings"],
            environments=environments,
        )


@dataclass
class SuiteSummary:
    """Suite-level counts and rates, with every scored environment weighted equally."""

    suite: str
    """Suite name."""

    scored: int
    """Environments scored."""

    skipped: int
    """Environments skipped because they place only anchors and the robot."""

    failed: int
    """Environments that failed to build or place."""

    layout_success_rate: float | None
    """Mean layout success rate over scored environments, or None when none were scored or any failed."""

    valid_candidate_rate: float | None
    """Mean valid-candidate rate over scored environments, or None when none were scored or any failed."""

    wall_time_s: float
    """Total time spent on the suite."""


@dataclass
class RateChange:
    """A rate in two runs and the 95% interval for the difference, candidate minus baseline."""

    baseline: float
    """Rate in the baseline run."""

    candidate: float
    """Rate in the candidate run."""

    low: float
    """Lower end of the 95% interval for the difference."""

    high: float
    """Upper end of the 95% interval for the difference."""

    p_value: float
    """Two-sided p-value for a zero difference."""


@dataclass
class SuiteComparison:
    """Rate changes over the environments a suite scored in both runs with the same checks."""

    suite: str
    """Suite name."""

    environments: int
    """Environments compared."""

    layout_success: RateChange
    """Change in mean layout success rate."""

    valid_candidates: RateChange
    """Change in mean valid-candidate rate."""


@dataclass
class EnvironmentComparison:
    """Rate changes for one environment scored in both runs."""

    suite: str
    """Suite name."""

    environment: str
    """Spec file stem."""

    layout_success: RateChange
    """Change in layout success rate."""

    valid_candidates: RateChange
    """Change in valid-candidate rate."""

    significant: bool = False
    """Whether either change stays significant after a Holm adjustment over every environment and rate."""


@dataclass
class RunComparison:
    """Comparison of a candidate run against a baseline run."""

    suites: list[SuiteComparison]
    """One entry per suite with environments compared."""

    environments: list[EnvironmentComparison]
    """One entry per environment scored in both runs with the same checks."""

    unpaired: list[tuple[str, str, str, str]]
    """(suite, environment, baseline status, candidate status) for environments not scored in both runs.

    Environments skipped in both runs are left out.
    """

    settings_differences: list[tuple[str, object, object]]
    """(setting, baseline value, candidate value) for each setting that differs."""

    check_differences: list[tuple[str, str, dict[str, bool], dict[str, bool]]]
    """(suite, environment, baseline checks, candidate checks) for environments scored in both runs where a check that
    rejected candidates in either run is missing from the other run or differs in being required. Their success means
    something different in each run, so they are not compared."""


def summarize_suites(run: PlacementBenchmarkRun) -> list[SuiteSummary]:
    """Return one summary per suite, in order of first appearance.

    A suite's rates are None when any of its environments failed, since the failed environment could have lowered them.

    Args:
        run: Run to summarize.
    """
    records_per_suite: dict[str, list[EnvironmentRecord]] = {}
    for record in run.environments:
        records_per_suite.setdefault(record.suite, []).append(record)
    summaries = []
    for suite, records in records_per_suite.items():
        scored = [record for record in records if record.status == EnvironmentStatus.SCORED]
        failed = sum(record.status == EnvironmentStatus.FAILED for record in records)
        summaries.append(
            SuiteSummary(
                suite=suite,
                scored=len(scored),
                skipped=sum(record.status == EnvironmentStatus.SKIPPED for record in records),
                failed=failed,
                layout_success_rate=None if failed else _mean([record.layout_success_rate for record in scored]),
                valid_candidate_rate=None if failed else _mean([record.valid_candidate_rate for record in scored]),
                wall_time_s=sum(record.wall_time_s for record in records),
            )
        )
    return summaries


def compare_runs(baseline: PlacementBenchmarkRun, candidate: PlacementBenchmarkRun) -> RunComparison:
    """Return a comparison of two runs over the environments scored in both with the same checks.

    Each environment's interval is the Agresti-Caffo interval for a difference of two proportions. A suite's
    interval combines its environments' variances, treating the set of environments as fixed and their samples as
    independent. Environments are flagged as significant after a Holm adjustment over every environment and rate,
    at a 5% family-wise error rate.

    Args:
        baseline: Run to compare against.
        candidate: Run measured with the change.
    """
    baseline_records = {(record.suite, record.environment): record for record in baseline.environments}
    candidate_records = {(record.suite, record.environment): record for record in candidate.environments}
    environments = []
    unpaired = []
    check_differences = []
    for key in sorted(baseline_records.keys() | candidate_records.keys()):
        before, after = baseline_records.get(key), candidate_records.get(key)
        if _has_status(before, EnvironmentStatus.SCORED) and _has_status(after, EnvironmentStatus.SCORED):
            if _success_criteria_differ(before, after):
                check_differences.append((*key, before.checks, after.checks))
                continue
            environments.append(
                EnvironmentComparison(
                    suite=key[0],
                    environment=key[1],
                    layout_success=_compare_proportions(
                        before.layout_successes, before.num_layouts, after.layout_successes, after.num_layouts
                    ),
                    valid_candidates=_compare_proportions(
                        before.candidate_successes,
                        before.num_candidates,
                        after.candidate_successes,
                        after.num_candidates,
                    ),
                )
            )
        elif not (_has_status(before, EnvironmentStatus.SKIPPED) and _has_status(after, EnvironmentStatus.SKIPPED)):
            unpaired.append((*key, _status_name(before), _status_name(after)))

    _flag_significant_environments(environments)

    comparisons_per_suite: dict[str, list[EnvironmentComparison]] = {}
    for comparison in environments:
        comparisons_per_suite.setdefault(comparison.suite, []).append(comparison)
    suites = []
    for suite, comparisons in comparisons_per_suite.items():
        suites.append(
            SuiteComparison(
                suite=suite,
                environments=len(comparisons),
                layout_success=_combine([comparison.layout_success for comparison in comparisons]),
                valid_candidates=_combine([comparison.valid_candidates for comparison in comparisons]),
            )
        )
    settings_differences = []
    for key in sorted(baseline.settings.keys() | candidate.settings.keys()):
        if baseline.settings.get(key) != candidate.settings.get(key):
            settings_differences.append((key, baseline.settings.get(key), candidate.settings.get(key)))
    return RunComparison(suites, environments, unpaired, settings_differences, check_differences)


def format_run(run: PlacementBenchmarkRun) -> str:
    """Return a Markdown table of suite results, followed by the environments that were not scored.

    Args:
        run: Run to format.
    """
    lines = [
        f"Placement success at {format_provenance(run)}",
        "",
        "| suite | scored | skipped | failed | layout success | valid candidates | time |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summarize_suites(run):
        lines.append(
            f"| {summary.suite} | {summary.scored} | {summary.skipped} | {summary.failed} |"
            f" {_format_percent(summary.layout_success_rate)} | {_format_percent(summary.valid_candidate_rate)} |"
            f" {summary.wall_time_s:.0f} s |"
        )
    not_scored = [record for record in run.environments if record.status != EnvironmentStatus.SCORED]
    if not_scored:
        lines += ["", "Not scored:"]
        for record in not_scored:
            lines.append(f"- {record.suite}/{record.environment} ({record.status}): {record.reason}")
    return "\n".join(lines)


def format_comparison(comparison: RunComparison) -> str:
    """Return a Markdown table of suite changes, after any settings and check warnings and before the significant and
    unpaired environments.

    Args:
        comparison: Comparison from compare_runs().
    """
    lines = []
    if comparison.settings_differences:
        lines.append("Warning: the runs used different settings:")
        for key, before, after in comparison.settings_differences:
            lines.append(f"- {key}: baseline {before}, candidate {after}")
        lines.append("")
    if comparison.check_differences:
        lines.append(
            "Warning: these environments' checks or required checks differ between the runs, so their success"
            " criteria changed. They are left out of the suite rates and significance tests:"
        )
        for suite, environment, before, after in comparison.check_differences:
            lines.append(
                f"- {suite}/{environment}: baseline {_format_checks(before)}; candidate {_format_checks(after)}"
            )
        lines.append("")
    lines += [
        (
            "| suite | environments compared | layout success, baseline → candidate | change (95% interval) |"
            " valid candidates, baseline → candidate | change (95% interval) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    if not comparison.suites:
        lines.append("| no environment was scored in both runs with the same checks | | | | | |")
    for suite in comparison.suites:
        lines.append(
            f"| {suite.suite} | {suite.environments} | {_format_rates(suite.layout_success)} |"
            f" {_format_change(suite.layout_success)} | {_format_rates(suite.valid_candidates)} |"
            f" {_format_change(suite.valid_candidates)} |"
        )
    significant = [environment for environment in comparison.environments if environment.significant]
    if significant:
        lines += ["", "Environments with a significant change (Holm-adjusted over all environments and rates):"]
        for environment in significant:
            layout_success = environment.layout_success
            valid_candidates = environment.valid_candidates
            lines.append(
                f"- {environment.suite}/{environment.environment}:"
                f" layout success {_format_rates(layout_success)} {_format_change(layout_success)},"
                f" valid candidates {_format_rates(valid_candidates)} {_format_change(valid_candidates)}"
            )
    if comparison.unpaired:
        lines += ["", "Not scored in both runs:"]
        for suite, environment, before, after in comparison.unpaired:
            lines.append(f"- {suite}/{environment}: baseline {before}, candidate {after}")
    return "\n".join(lines)


def format_provenance(run: PlacementBenchmarkRun) -> str:
    """Return the run's commit, branch, dirty flag and start time on one line.

    Args:
        run: Run to describe.
    """
    dirty = {True: ", uncommitted changes", False: "", None: ", unknown changes"}[run.dirty]
    return f"{run.commit or 'unknown commit'} ({run.branch or 'unknown branch'}{dirty}), started {run.start_time}"


def _metric(name: str, value: float | int, tags: dict[str, str]) -> dict[str, object]:
    return {"name": name, "value": value, "tags": tags}


def _success_criteria_differ(before: EnvironmentRecord, after: EnvironmentRecord) -> bool:
    """Return whether a check that rejected candidates in either run is missing from the other or differently required.

    A check that never rejected a candidate in either run leaves success unchanged, so it is ignored.
    """
    active_checks = set()
    for record in (before, after):
        for check, failures in record.candidate_failures_per_check.items():
            if failures:
                active_checks.add(check)
    return any(before.checks.get(check) != after.checks.get(check) for check in active_checks)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _has_status(record: EnvironmentRecord | None, status: EnvironmentStatus) -> bool:
    return record is not None and record.status == status


def _status_name(record: EnvironmentRecord | None) -> str:
    return "missing" if record is None else str(record.status)


def _compare_proportions(baseline_successes: int, baseline_trials: int, successes: int, trials: int) -> RateChange:
    """Return observed rates and the Agresti-Caffo 95% interval for their difference."""
    adjusted_baseline = (baseline_successes + 1) / (baseline_trials + 2)
    adjusted = (successes + 1) / (trials + 2)
    variance = _adjusted_variance(baseline_successes, baseline_trials) + _adjusted_variance(successes, trials)
    standard_error = math.sqrt(variance)
    difference = adjusted - adjusted_baseline
    margin = _NORMAL_QUANTILE_95 * standard_error
    return RateChange(
        baseline_successes / baseline_trials,
        successes / trials,
        difference - margin,
        difference + margin,
        _two_sided_p_value(difference, standard_error),
    )


def _adjusted_variance(successes: int, trials: int) -> float:
    adjusted = (successes + 1) / (trials + 2)
    return adjusted * (1 - adjusted) / (trials + 2)


def _combine(changes: list[RateChange]) -> RateChange:
    """Average per-environment changes, combining their interval half-widths as independent errors."""
    count = len(changes)
    baseline = sum(change.baseline for change in changes) / count
    candidate = sum(change.candidate for change in changes) / count
    centre = sum((change.low + change.high) / 2 for change in changes) / count
    half_width = math.sqrt(sum(((change.high - change.low) / 2) ** 2 for change in changes)) / count
    p_value = _two_sided_p_value(centre, half_width / _NORMAL_QUANTILE_95)
    return RateChange(baseline, candidate, centre - half_width, centre + half_width, p_value)


def _two_sided_p_value(difference: float, standard_error: float) -> float:
    """Return the normal two-sided p-value for a difference with the given standard error."""
    return math.erfc(abs(difference) / standard_error / math.sqrt(2))


def _flag_significant_environments(environments: list[EnvironmentComparison]) -> None:
    """Mark environments with a change that survives a Holm adjustment over every environment and rate."""
    tests = []
    for environment in environments:
        tests.append((environment.layout_success.p_value, environment))
        tests.append((environment.valid_candidates.p_value, environment))
    tests.sort(key=lambda test: test[0])
    for rank, (p_value, environment) in enumerate(tests):
        if p_value > _FAMILY_ALPHA / (len(tests) - rank):
            break
        environment.significant = True


def _format_checks(checks: dict[str, bool]) -> str:
    names = [check if required else f"{check} (optional)" for check, required in sorted(checks.items())]
    return ", ".join(names) or "none"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _format_rates(change: RateChange) -> str:
    return f"{_format_percent(change.baseline)} → {_format_percent(change.candidate)}"


def _format_change(change: RateChange) -> str:
    return f"[{100 * change.low:+.1f}, {100 * change.high:+.1f}] pts"
