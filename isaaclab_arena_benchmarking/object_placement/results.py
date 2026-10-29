# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement success benchmark results: settings, environment records, results files and suite summaries.

A results file holds the start time, the commit and branch, the settings and one record per environment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, StrEnum
from pathlib import Path

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.placement_diagnostics import ValidationCheckDiagnostics


class EnvironmentStatus(StrEnum):
    """Outcome of benchmarking one environment spec."""

    SCORED = "scored"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PlacementBenchmarkSettings:
    """Measurement settings shared by every environment of a run."""

    num_envs: int = 10
    """Environments in each build. A build's first pool batch solves num_envs * min_unique_layouts_per_env *
    max_placement_attempts candidates."""

    placement_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    """Distinct nonnegative seeds; each seed builds every spec once, solving its own candidates."""

    random_yaw_init: bool = False
    """Whether placed objects, including a relation-placed robot, start from a random yaw instead of each spec's
    setting. Objects with FaceTo or a tilted RotateAroundSolution keep their own orientation."""

    collision_mode: CollisionMode | None = None
    """Default collision mode for every spec, or None to keep each spec's own."""

    solver_device: str | None = None
    """Device the solver ran on, recorded by the run rather than chosen; None before the simulator starts."""

    def __post_init__(self):
        assert self.num_envs > 0, "num_envs must be positive"
        assert self.placement_seeds, "placement_seeds must not be empty"
        assert all(seed >= 0 for seed in self.placement_seeds), "placement_seeds must be nonnegative"
        # A repeated seed would count the same candidates twice.
        assert len(set(self.placement_seeds)) == len(self.placement_seeds), "placement_seeds must be distinct"


@dataclass
class EnvironmentRecord:
    """Placement outcome for one environment spec, accumulated over one build per seed."""

    suite: str
    """Suite the spec belongs to, such as robolab or kitchen_bench."""

    environment: str
    """Spec file stem."""

    status: EnvironmentStatus
    """Whether the environment was scored, skipped or failed to run."""

    reason: str = ""
    """Why the environment was skipped or failed; empty when it was scored."""

    num_builds: int = 0
    """Builds of the spec, one per seed."""

    num_candidates: int = 0
    """Candidates in the first pool batch of every build."""

    num_valid_candidates: int = 0
    """Candidates in those first batches that passed every required check."""

    num_envs: int = 0
    """Environments placed over all builds."""

    num_fallback_envs: int = 0
    """Environments whose pool fell back to best-loss layouts because none of their candidates was valid."""

    num_batches: int = 0
    """Pool batches solved over all builds; a build needs more than one when an environment lacks valid layouts."""

    checks: dict[str, ValidationCheckDiagnostics] = field(default_factory=dict)
    """Counts for each check over the first pool batches of every build. A check that runs only on candidates that
    passed the cheaper required checks, such as ik_reachable, counts only those candidates."""

    wall_time_s: float = 0.0
    """Time to build every seed."""

    @property
    def valid_candidate_rate(self) -> float:
        """Share of first-batch candidates that are valid."""
        return self.num_valid_candidates / self.num_candidates

    @property
    def valid_env_rate(self) -> float:
        """Share of environments whose pool holds only valid layouts, without a best-loss fallback."""
        return 1 - self.num_fallback_envs / self.num_envs

    @property
    def batches_per_build(self) -> float:
        """Mean number of pool batches a build solved."""
        return self.num_batches / self.num_builds


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
    """Whether the checkout had changes to tracked files or new files in the benchmarked packages, or None when
    unknown."""

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
        _assert_known_fields(cls, content, path)
        settings = content["settings"]
        _assert_known_fields(PlacementBenchmarkSettings, settings, path)
        if settings["collision_mode"] is not None:
            settings["collision_mode"] = CollisionMode(settings["collision_mode"])
        content["settings"] = PlacementBenchmarkSettings(**settings)
        environments = []
        for record in content["environments"]:
            _assert_known_fields(EnvironmentRecord, record, path)
            record["status"] = EnvironmentStatus(record["status"])
            checks = {}
            for check, check_values in record["checks"].items():
                _assert_known_fields(ValidationCheckDiagnostics, check_values, path)
                checks[check] = ValidationCheckDiagnostics(**check_values)
            record["checks"] = checks
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
    """Environments skipped because their builds solve no relations."""

    num_failed: int
    """Environments that failed to build or place."""

    valid_candidate_rate: float | None
    """Mean valid-candidate rate over scored environments, or None when none were scored or any failed."""

    valid_env_rate: float | None
    """Mean valid-environment rate over scored environments, or None when none were scored or any failed."""

    batches_per_build: float | None
    """Mean pool batches per build over scored environments, or None when none were scored or any failed."""

    wall_time_s: float
    """Total time spent on the suite."""


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
                valid_candidate_rate=None if num_failed else _mean([record.valid_candidate_rate for record in scored]),
                valid_env_rate=None if num_failed else _mean([record.valid_env_rate for record in scored]),
                batches_per_build=None if num_failed else _mean([record.batches_per_build for record in scored]),
                wall_time_s=sum(record.wall_time_s for record in records),
            )
        )
    return summaries


def format_results(results: PlacementBenchmarkResults) -> str:
    """Return a Markdown table of suite results, followed by the environments that were not scored.

    Args:
        results: Results to format.
    """
    lines = [
        f"Placement success at {format_provenance(results)}",
        "",
        "| suite | scored | skipped | failed | valid candidates | valid environments | batches per build | time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary in summarize_suites(results):
        lines.append(
            f"| {summary.suite} | {summary.num_scored} | {summary.num_skipped} | {summary.num_failed} |"
            f" {_format_percent(summary.valid_candidate_rate)} | {_format_percent(summary.valid_env_rate)} |"
            f" {_format_number(summary.batches_per_build)} | {summary.wall_time_s:.0f} s |"
        )
    not_scored = [record for record in results.environments if record.status != EnvironmentStatus.SCORED]
    if not_scored:
        lines += ["", "Not scored:"]
        for record in not_scored:
            lines.append(f"- {record.suite}/{record.environment} ({record.status}): {record.reason}")
    return "\n".join(lines)


def format_provenance(results: PlacementBenchmarkResults) -> str:
    """Return the measured commit, branch, dirty flag and start time on one line.

    Args:
        results: Results to describe.
    """
    dirty = {True: ", uncommitted changes", False: "", None: ", unknown changes"}[results.dirty]
    commit = results.commit or "unknown commit"
    return f"{commit} ({results.branch or 'unknown branch'}{dirty}), started {results.start_time}"


def _assert_known_fields(cls: type, values: dict[str, object], path: str | Path) -> None:
    """Assert that a results file's values name only fields of cls, which fails for files from other versions."""
    unknown = sorted(set(values) - {class_field.name for class_field in fields(cls)})
    assert not unknown, (
        f"{path} has {cls.__name__} fields that this benchmark does not have: {unknown}. Another version of the"
        " benchmark wrote it; rerun the benchmark with this version."
    )


def _enum_to_value(value: object) -> object:
    """Return an enum's value, the form in which results files store enums."""
    assert isinstance(value, Enum), f"Results files cannot store {value!r}"
    return value.value


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"
