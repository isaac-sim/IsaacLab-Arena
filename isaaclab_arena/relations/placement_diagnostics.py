# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Counts that describe how relation placement went: per validated batch, per fill of a placement pool, and summed
over several fills."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RejectionLog:
    """Rejections one check made on one batch of candidate layouts, counted by cause."""

    check: str
    """Name of the check whose rejections this log holds."""

    verbose: bool = False
    """Whether to print each rejection, with its measured detail, as it is added."""

    num_rejected_by_cause: dict[str, int] = field(default_factory=dict)
    """Rejections counted by cause, such as "'red_bowl' next_to(wooden_bowl, negative_y): on the wrong side". A
    check that inspects every object pair, such as no_overlap, can attribute one candidate to several causes."""

    def add(self, cause: str, count: int = 1, detail: str = "") -> None:
        """Add candidates rejected for cause to its count, and print it with detail when verbose.

        Args:
            cause: What failed and how, naming the objects and relation, such as
                "'bowl' next_to(plate, positive_x): on the wrong side". Holds no measured values, so every
                rejection of the same kind has the same cause and is counted together.
            count: Candidates rejected for this cause.
            detail: Measured violation, such as "by 0.150 m". Printed when verbose but not kept.
        """
        self.num_rejected_by_cause[cause] = self.num_rejected_by_cause.get(cause, 0) + count
        if self.verbose:
            suffix = f" ({detail})" if detail else ""
            print(f"  [{self.check}] {cause} x{count}{suffix}")


@dataclass
class ValidationCheckDiagnostics:
    """How one pre-physics check did on a batch of candidate layouts."""

    num_evaluated: int
    """Candidates the check ran on. A check that runs after the cheaper required checks skips the candidates that
    failed them."""

    num_passed: int
    """Candidates the check ran on and passed."""

    required: bool
    """Whether a candidate must pass this check to be valid."""

    num_rejected_by_cause: dict[str, int] = field(default_factory=dict)
    """Rejections counted by cause, such as "'red_bowl' next_to(wooden_bowl, negative_y): on the wrong side"."""

    @property
    def num_rejected(self) -> int:
        """Candidates the check ran on and failed."""
        return self.num_evaluated - self.num_passed


@dataclass
class BatchValidationDiagnostics:
    """Validation outcome of one batch of candidate layouts."""

    num_candidates: int
    """Candidates in the batch."""

    num_valid: int
    """Candidates that passed every required check."""

    checks: dict[str, ValidationCheckDiagnostics]
    """Counts for each check that ran on the batch."""

    def format_passed_per_check(self) -> str:
        """Return each check's passed and evaluated counts on one line, such as "on_relation=49/50, no_overlap=43/50"."""
        return ", ".join(
            f"{check}={check_diagnostics.num_passed}/{check_diagnostics.num_evaluated}"
            for check, check_diagnostics in self.checks.items()
        )


@dataclass
class PlacementDiagnostics:
    """How one fill of a placement pool went: every batch it validated, in order, and the environments whose
    remaining slots it filled with best-loss layouts because too few of their candidates were valid."""

    num_envs: int
    """Number of environments the pool fills."""

    batches: list[BatchValidationDiagnostics] = field(default_factory=list)
    """Validated batches, in solve order. A batch after the first runs only while some environment lacks
    layouts."""

    fallback_envs: list[int] = field(default_factory=list)
    """Environments given best-loss layouts, which fail at least one required check."""

    def get_num_rejected_by_cause_per_check(self) -> dict[str, dict[str, int]]:
        """Return each check's rejections summed over every batch, counted by cause, most frequent first."""
        num_rejected_by_cause_per_check: dict[str, dict[str, int]] = {}
        for batch in self.batches:
            for check, check_diagnostics in batch.checks.items():
                check_totals = num_rejected_by_cause_per_check.setdefault(check, {})
                for cause, count in check_diagnostics.num_rejected_by_cause.items():
                    check_totals[cause] = check_totals.get(cause, 0) + count
        return {
            check: dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
            for check, counts in num_rejected_by_cause_per_check.items()
            if counts
        }

    def format_summary(self, max_causes_per_check: int | None = 5) -> str:
        """Return one line per batch with its valid and per-check counts, then the most frequent rejection causes
        of each check over all batches, then the fallback environments, if any.

        Args:
            max_causes_per_check: Most rejection causes listed for each check. None lists every cause.
        """
        lines = []
        for index, batch in enumerate(self.batches):
            lines.append(
                f"batch {index}: {batch.num_valid}/{batch.num_candidates} valid; passed per check:"
                f" {batch.format_passed_per_check()}"
            )
        num_rejected_by_cause_per_check = self.get_num_rejected_by_cause_per_check()
        if num_rejected_by_cause_per_check:
            lines.append("rejections by cause over all batches:")
        for check, counts in num_rejected_by_cause_per_check.items():
            for cause, count in list(counts.items())[:max_causes_per_check]:
                lines.append(f"  {check}: {cause} x{count}")
            if max_causes_per_check is not None and len(counts) > max_causes_per_check:
                lines.append(f"  {check}: {len(counts) - max_causes_per_check} more cause(s)")
        if self.fallback_envs:
            lines.append(f"best-loss fallback for envs: {self.fallback_envs}")
        return "\n".join(lines)


@dataclass
class AggregatedPlacementDiagnostics:
    """Placement diagnostics summed over several pool fills, such as the builds of a Run."""

    num_builds: int
    """Fills whose diagnostics are summed, one per build."""

    num_envs: int
    """Sum of the fills' num_envs."""

    num_fallback_envs: int
    """Environments given best-loss layouts, summed over the fills."""

    num_batches: int
    """Batches the fills validated, summed over the fills."""

    batches: list[BatchValidationDiagnostics]
    """Counts summed by batch position: entry i sums the i-th batch of every fill that validated one."""

    def save(self, path: str | Path) -> None:
        """Write the diagnostics to a JSON file, replacing any existing file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AggregatedPlacementDiagnostics:
        """Return the diagnostics that save() wrote to path."""
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        batches = []
        for batch in content["batches"]:
            checks = {check: ValidationCheckDiagnostics(**counts) for check, counts in batch["checks"].items()}
            batches.append(
                BatchValidationDiagnostics(
                    num_candidates=batch["num_candidates"], num_valid=batch["num_valid"], checks=checks
                )
            )
        return cls(
            num_builds=content["num_builds"],
            num_envs=content["num_envs"],
            num_fallback_envs=content["num_fallback_envs"],
            num_batches=content["num_batches"],
            batches=batches,
        )


def aggregate_placement_diagnostics(
    diagnostics_per_build: list[PlacementDiagnostics],
) -> AggregatedPlacementDiagnostics:
    """Sum the diagnostics of several pool fills, adding each fill's batches by their position in it.

    Args:
        diagnostics_per_build: Diagnostics of each fill, such as each build of a Run that solved relation placement.
    """
    assert diagnostics_per_build, "diagnostics_per_build must contain at least one fill"
    batches: list[BatchValidationDiagnostics] = []
    for diagnostics in diagnostics_per_build:
        for position, batch in enumerate(diagnostics.batches):
            if position == len(batches):
                batches.append(BatchValidationDiagnostics(num_candidates=0, num_valid=0, checks={}))
            _add_batch_counts(batches[position], batch)
    return AggregatedPlacementDiagnostics(
        num_builds=len(diagnostics_per_build),
        num_envs=sum(diagnostics.num_envs for diagnostics in diagnostics_per_build),
        num_fallback_envs=sum(len(diagnostics.fallback_envs) for diagnostics in diagnostics_per_build),
        num_batches=sum(len(diagnostics.batches) for diagnostics in diagnostics_per_build),
        batches=batches,
    )


def _add_batch_counts(total: BatchValidationDiagnostics, batch: BatchValidationDiagnostics) -> None:
    """Add a batch's candidate counts and per-check counts to total, in place."""
    total.num_candidates += batch.num_candidates
    total.num_valid += batch.num_valid
    for check, check_diagnostics in batch.checks.items():
        total_check = total.checks.setdefault(
            check, ValidationCheckDiagnostics(num_evaluated=0, num_passed=0, required=check_diagnostics.required)
        )
        assert total_check.required == check_diagnostics.required, f"Check '{check}' is required in only some batches."
        total_check.num_evaluated += check_diagnostics.num_evaluated
        total_check.num_passed += check_diagnostics.num_passed
        for cause, count in check_diagnostics.num_rejected_by_cause.items():
            total_check.num_rejected_by_cause[cause] = total_check.num_rejected_by_cause.get(cause, 0) + count
