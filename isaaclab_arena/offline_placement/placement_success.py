# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement-success benchmark results: settings, results files, suite summaries and run comparisons.

A results file holds the start time, the commit and branch, the settings and one record per environment, so that two
runs can be compared.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

from isaaclab_arena.relations.collision_mode import CollisionMode

_NORMAL_QUANTILE_95 = 1.959963984540054
"""Two-sided 95% quantile of the standard normal distribution."""

_FAMILY_ALPHA = 0.05
"""Family-wise error rate of the Holm adjustment over every environment and rate."""


class EnvironmentStatus(StrEnum):
    """Outcome of benchmarking one environment spec."""

    SCORED = "scored"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PlacementBenchmarkSettings:
    """Measurement settings shared by every environment of a run."""

    num_layouts: int = 50
    """Layouts placed per environment and seed."""

    placement_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    """Distinct nonnegative seeds; each seed solves its own candidates."""

    random_yaw_init: bool = False
    """Whether placed objects, including a relation-placed robot, start from a random yaw instead of each spec's
    setting. Objects with FaceTo or a tilted RotateAroundSolution keep their own orientation."""

    collision_mode: CollisionMode | None = None
    """Default collision mode for every spec, or None to keep each spec's own."""

    solver_device: str | None = None
    """Device the solver ran on, recorded by the run rather than chosen; None before the simulator starts."""

    def __post_init__(self):
        assert self.num_layouts > 0, "num_layouts must be positive"
        assert self.placement_seeds, "placement_seeds must not be empty"
        assert all(seed >= 0 for seed in self.placement_seeds), "placement_seeds must be nonnegative"
        # A repeated seed would count the same candidates twice.
        assert len(set(self.placement_seeds)) == len(self.placement_seeds), "placement_seeds must be distinct"


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

    num_successful_layouts: int = 0
    """Layouts whose best candidate passed every required check."""

    num_candidates: int = 0
    """Candidates solved."""

    num_valid_candidates: int = 0
    """Candidates that passed every required check."""

    checks: dict[str, bool] = field(default_factory=dict)
    """Checks that ran for this environment, mapped to whether they are required for success."""

    candidate_failures_per_check: dict[str, int] = field(default_factory=dict)
    """Candidates rejected by each check, including optional checks.

    As in PlacementValidationRunner, a check that runs only on candidates that pass the cheaper required checks, such
    as IK reachability, counts as failed on the candidates it skipped.
    """

    wall_time_s: float = 0.0
    """Time to prepare the placement inputs and place every seed."""

    @property
    def layout_success_rate(self) -> float:
        """Share of layouts whose best candidate is valid."""
        return self.num_successful_layouts / self.num_layouts

    @property
    def valid_candidate_rate(self) -> float:
        """Share of candidates that are valid."""
        return self.num_valid_candidates / self.num_candidates


@dataclass
class PlacementBenchmarkResults:
    """Results of one benchmark run: its provenance, settings and one record per environment."""

    start_time: str
    """UTC start time in ISO 8601 format."""

    commit: str | None
    """Git commit of the measured code, or None when unknown."""

    branch: str | None
    """Git branch of the measured code, or None when unknown."""

    dirty: bool | None
    """Whether the checkout had uncommitted changes, or None when unknown."""

    settings: PlacementBenchmarkSettings
    """Settings shared by every environment."""

    environments: list[EnvironmentRecord]
    """Records in discovery order."""

    def save(self, path: str | Path) -> None:
        """Write the results as JSON.

        Args:
            path: File to write, replacing any existing file.
        """
        # Write a sibling file and rename it, so a run stopped mid-write keeps the previous results.
        temporary_path = Path(path).with_name(Path(path).name + ".partial")
        content = json.dumps(asdict(self), default=_enum_to_value, allow_nan=False, indent=2)
        temporary_path.write_text(content + "\n", encoding="utf-8")
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> PlacementBenchmarkResults:
        """Read results written by save().

        Args:
            path: Results file to read.
        """
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = content["settings"]
        if settings["collision_mode"] is not None:
            settings["collision_mode"] = CollisionMode(settings["collision_mode"])
        content["settings"] = PlacementBenchmarkSettings(**settings)
        environments = []
        for record in content["environments"]:
            record["status"] = EnvironmentStatus(record["status"])
            environments.append(EnvironmentRecord(**record))
        content["environments"] = environments
        return cls(**content)


@dataclass
class SuiteSummary:
    """Suite-level counts and rates, with every scored environment weighted equally."""

    suite: str
    """Suite name."""

    num_scored: int
    """Environments scored."""

    num_skipped: int
    """Environments skipped because they place nothing other than anchors."""

    num_failed: int
    """Environments that failed to build or place."""

    layout_success_rate: float | None
    """Mean layout success rate over scored environments, or None when none were scored or any failed."""

    valid_candidate_rate: float | None
    """Mean valid-candidate rate over scored environments, or None when none were scored or any failed."""

    wall_time_s: float
    """Total time spent on the suite."""


@dataclass
class RateChange:
    """A rate before and after a change, and the 95% interval for the difference, after minus before."""

    before: float
    """Rate in the run before the change."""

    after: float
    """Rate in the run after the change."""

    low: float
    """Lower end of the 95% interval for the difference."""

    high: float
    """Upper end of the 95% interval for the difference."""

    p_value: float
    """Two-sided p-value for a zero difference."""


@dataclass
class SuiteComparison:
    """Rate changes over the environments of a suite that both runs scored with the same success criteria."""

    suite: str
    """Suite name."""

    num_environments: int
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
class ExcludedEnvironment:
    """An environment left out of a comparison, with its record in each run."""

    suite: str
    """Suite name."""

    environment: str
    """Spec file stem."""

    before: EnvironmentRecord | None
    """Record in the run before the change, or None when the run has none."""

    after: EnvironmentRecord | None
    """Record in the run after the change, or None when the run has none."""


@dataclass
class ResultsComparison:
    """Comparison of the results after a change against the results before it."""

    suites: list[SuiteComparison]
    """One entry per suite with environments compared."""

    environments: list[EnvironmentComparison]
    """One entry per environment scored in both runs with the same success criteria."""

    unpaired: list[ExcludedEnvironment]
    """Environments not scored in both runs. Environments skipped in both runs are left out."""

    settings_differences: dict[str, tuple[object, object]]
    """Each setting that differs, mapped to its values before and after."""

    check_differences: list[ExcludedEnvironment]
    """Environments scored in both runs whose success criteria differ, so they are not compared.

    Criteria differ when a check required in only one run rejected candidates in that run.
    """


def summarize_suites(results: PlacementBenchmarkResults) -> list[SuiteSummary]:
    """Return one summary per suite, in order of first appearance.

    A suite's rates are None when any of its environments failed, since the failed environment could have lowered them.

    Args:
        results: Results to summarize.
    """
    records_per_suite: dict[str, list[EnvironmentRecord]] = {}
    for record in results.environments:
        records_per_suite.setdefault(record.suite, []).append(record)
    summaries = []
    for suite, records in records_per_suite.items():
        scored = [record for record in records if record.status == EnvironmentStatus.SCORED]
        num_failed = sum(record.status == EnvironmentStatus.FAILED for record in records)
        summaries.append(
            SuiteSummary(
                suite=suite,
                num_scored=len(scored),
                num_skipped=sum(record.status == EnvironmentStatus.SKIPPED for record in records),
                num_failed=num_failed,
                layout_success_rate=None if num_failed else _mean([record.layout_success_rate for record in scored]),
                valid_candidate_rate=None if num_failed else _mean([record.valid_candidate_rate for record in scored]),
                wall_time_s=sum(record.wall_time_s for record in records),
            )
        )
    return summaries


def compare_results(before: PlacementBenchmarkResults, after: PlacementBenchmarkResults) -> ResultsComparison:
    """Return a comparison of two runs over the environments both scored with the same success criteria.

    Each environment's interval is the Agresti-Caffo interval for a difference of two proportions. A suite's
    interval combines its environments' variances, treating the set of environments as fixed and their samples as
    independent. Environments are flagged as significant after a Holm adjustment over every environment and rate,
    at a 5% family-wise error rate.

    Args:
        before: Results measured before the change.
        after: Results measured after the change.
    """
    records_before = {(record.suite, record.environment): record for record in before.environments}
    records_after = {(record.suite, record.environment): record for record in after.environments}
    environments = []
    unpaired = []
    check_differences = []
    for key in sorted(records_before.keys() | records_after.keys()):
        record_before, record_after = records_before.get(key), records_after.get(key)
        if _has_status(record_before, EnvironmentStatus.SCORED) and _has_status(record_after, EnvironmentStatus.SCORED):
            if _success_criteria_differ(record_before, record_after):
                check_differences.append(ExcludedEnvironment(*key, record_before, record_after))
                continue
            environments.append(
                EnvironmentComparison(
                    suite=key[0],
                    environment=key[1],
                    layout_success=_compare_proportions(
                        record_before.num_successful_layouts,
                        record_before.num_layouts,
                        record_after.num_successful_layouts,
                        record_after.num_layouts,
                    ),
                    valid_candidates=_compare_proportions(
                        record_before.num_valid_candidates,
                        record_before.num_candidates,
                        record_after.num_valid_candidates,
                        record_after.num_candidates,
                    ),
                )
            )
        elif not (
            _has_status(record_before, EnvironmentStatus.SKIPPED)
            and _has_status(record_after, EnvironmentStatus.SKIPPED)
        ):
            unpaired.append(ExcludedEnvironment(*key, record_before, record_after))

    _flag_significant_environments(environments)

    comparisons_per_suite: dict[str, list[EnvironmentComparison]] = {}
    for comparison in environments:
        comparisons_per_suite.setdefault(comparison.suite, []).append(comparison)
    suites = []
    for suite, comparisons in comparisons_per_suite.items():
        suites.append(
            SuiteComparison(
                suite=suite,
                num_environments=len(comparisons),
                layout_success=_average_rate_changes([comparison.layout_success for comparison in comparisons]),
                valid_candidates=_average_rate_changes([comparison.valid_candidates for comparison in comparisons]),
            )
        )
    settings_before, settings_after = asdict(before.settings), asdict(after.settings)
    settings_differences = {}
    for key, value_before in settings_before.items():
        if value_before != settings_after[key]:
            settings_differences[key] = (value_before, settings_after[key])
    return ResultsComparison(suites, environments, unpaired, settings_differences, check_differences)


def format_results(results: PlacementBenchmarkResults) -> str:
    """Return a Markdown table of suite results, followed by the environments that were not scored.

    Args:
        results: Results to format.
    """
    lines = [
        f"Placement success at {format_provenance(results)}",
        "",
        "| suite | scored | skipped | failed | layout success | valid candidates | time |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summarize_suites(results):
        lines.append(
            f"| {summary.suite} | {summary.num_scored} | {summary.num_skipped} | {summary.num_failed} |"
            f" {_format_percent(summary.layout_success_rate)} | {_format_percent(summary.valid_candidate_rate)} |"
            f" {summary.wall_time_s:.0f} s |"
        )
    not_scored = [record for record in results.environments if record.status != EnvironmentStatus.SCORED]
    if not_scored:
        lines += ["", "Not scored:"]
        for record in not_scored:
            lines.append(f"- {record.suite}/{record.environment} ({record.status}): {record.reason}")
    return "\n".join(lines)


def format_comparison(comparison: ResultsComparison) -> str:
    """Return the comparison as Markdown: warnings, a table of suite changes, then the significant and unpaired
    environments.

    Args:
        comparison: Comparison from compare_results().
    """
    lines = []
    if comparison.settings_differences:
        lines.append("Warning: the runs used different settings:")
        for key, (value_before, value_after) in comparison.settings_differences.items():
            lines.append(f"- {key}: before {value_before}, after {value_after}")
        lines.append("")
    if comparison.check_differences:
        lines.append(
            "Warning: a check required in only one run rejected candidates in these environments, so their success"
            " criteria differ. They are left out of the suite rates and significance tests:"
        )
        for excluded in comparison.check_differences:
            lines.append(
                f"- {excluded.suite}/{excluded.environment}: before {_format_checks(excluded.before.checks)};"
                f" after {_format_checks(excluded.after.checks)}"
            )
        lines.append("")
    lines += [
        (
            "| suite | environments compared | layout success, before → after | change (95% interval) |"
            " valid candidates, before → after | change (95% interval) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    if not comparison.suites:
        lines.append("| no environment was scored in both runs with the same success criteria | | | | | |")
    for suite in comparison.suites:
        lines.append(
            f"| {suite.suite} | {suite.num_environments} | {_format_rates(suite.layout_success)} |"
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
        for excluded in comparison.unpaired:
            lines.append(
                f"- {excluded.suite}/{excluded.environment}: before {_format_status(excluded.before)},"
                f" after {_format_status(excluded.after)}"
            )
    return "\n".join(lines)


def format_provenance(results: PlacementBenchmarkResults) -> str:
    """Return the measured commit, branch, dirty flag and start time on one line.

    Args:
        results: Results to describe.
    """
    dirty = {True: ", uncommitted changes", False: "", None: ", unknown changes"}[results.dirty]
    commit = results.commit or "unknown commit"
    return f"{commit} ({results.branch or 'unknown branch'}{dirty}), started {results.start_time}"


def _success_criteria_differ(before: EnvironmentRecord, after: EnvironmentRecord) -> bool:
    """Return whether a check required in only one run rejected candidates in that run.

    A missing or optional check does not affect success, and neither does a required check that rejected nothing.
    """
    for check in before.checks.keys() | after.checks.keys():
        required_before = before.checks.get(check, False)
        required_after = after.checks.get(check, False)
        if required_before != required_after:
            record_requiring_check = before if required_before else after
            if record_requiring_check.candidate_failures_per_check.get(check, 0) > 0:
                return True
    return False


def _enum_to_value(value: object) -> object:
    """Return an enum's value, the form in which results files store enums."""
    assert isinstance(value, Enum), f"Results files cannot store {value!r}"
    return value.value


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _has_status(record: EnvironmentRecord | None, status: EnvironmentStatus) -> bool:
    return record is not None and record.status == status


def _compare_proportions(
    successes_before: int, trials_before: int, successes_after: int, trials_after: int
) -> RateChange:
    """Return observed rates and the Agresti-Caffo 95% interval for their difference."""
    adjusted_before = (successes_before + 1) / (trials_before + 2)
    adjusted_after = (successes_after + 1) / (trials_after + 2)
    variance = _adjusted_variance(successes_before, trials_before) + _adjusted_variance(successes_after, trials_after)
    standard_error = math.sqrt(variance)
    difference = adjusted_after - adjusted_before
    margin = _NORMAL_QUANTILE_95 * standard_error
    return RateChange(
        successes_before / trials_before,
        successes_after / trials_after,
        difference - margin,
        difference + margin,
        _two_sided_p_value(difference, standard_error),
    )


def _adjusted_variance(successes: int, trials: int) -> float:
    adjusted = (successes + 1) / (trials + 2)
    return adjusted * (1 - adjusted) / (trials + 2)


def _average_rate_changes(changes: list[RateChange]) -> RateChange:
    """Average per-environment changes, combining their interval half-widths as independent errors.

    The interval is centered on the observed change. The Agresti-Caffo centers shift every environment the same way when
    the runs' sample sizes differ, so averaging them would report a change between identical rates.
    """
    count = len(changes)
    before = sum(change.before for change in changes) / count
    after = sum(change.after for change in changes) / count
    center = after - before
    half_width = math.sqrt(sum(((change.high - change.low) / 2) ** 2 for change in changes)) / count
    p_value = _two_sided_p_value(center, half_width / _NORMAL_QUANTILE_95)
    return RateChange(before, after, center - half_width, center + half_width, p_value)


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


def _format_status(record: EnvironmentRecord | None) -> str:
    return "missing" if record is None else str(record.status)


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _format_rates(change: RateChange) -> str:
    return f"{_format_percent(change.before)} → {_format_percent(change.after)}"


def _format_change(change: RateChange) -> str:
    return f"[{100 * change.low:+.1f}, {100 * change.high:+.1f}] pts"
