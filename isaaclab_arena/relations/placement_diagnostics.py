# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Counts and rejections that describe how relation placement went: per validated batch, per fill of a placement
pool, and over several fills."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Rejection:
    """Candidates one check rejected for one cause, with the measured violation."""

    cause: str
    """What failed and how, naming the objects and relation, such as
    "'bowl' next_to(plate, positive_x): on the wrong side". Holds no measured values, so every rejection of the same
    kind has the same cause and can be counted together."""

    count: int = 1
    """Candidates rejected for this cause."""

    detail: str = ""
    """Measured violation, such as "by 0.150 m". Empty when the cause says everything."""

    def format(self) -> str:
        """Return the rejection on one line, such as "'bowl' next_to(plate, positive_x): on the wrong side x1 (by
        0.150 m)"."""
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.cause} x{self.count}{suffix}"


@dataclass
class RejectionLog:
    """Rejections one check made on one batch of candidate layouts, in the order the check made them."""

    check: str
    """Name of the check whose rejections this log holds."""

    verbose: bool = False
    """Whether to print each rejection as it is added."""

    rejections: list[Rejection] = field(default_factory=list)
    """Every rejection added, in order. A check that inspects every object pair, such as no_overlap, can reject one
    candidate for several causes."""

    def add(self, cause: str, count: int = 1, detail: str = "") -> None:
        """Add candidates rejected for cause, and print the rejection when verbose.

        Args:
            cause: What failed and how, naming the objects and relation, such as
                "'bowl' next_to(plate, positive_x): on the wrong side". Holds no measured values.
            count: Candidates rejected for this cause.
            detail: Measured violation, such as "by 0.150 m".
        """
        rejection = Rejection(cause, count, detail)
        self.rejections.append(rejection)
        if self.verbose:
            print(f"  [{self.check}] {rejection.format()}")


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

    rejections: list[Rejection] = field(default_factory=list)
    """Every rejection the check made, in order, with its measured violation."""

    @property
    def num_rejected(self) -> int:
        """Candidates the check ran on and failed."""
        return self.num_evaluated - self.num_passed

    def get_num_rejected_by_cause(self) -> dict[str, int]:
        """Return the check's rejections counted by cause, in the order each cause first occurred."""
        num_rejected_by_cause: dict[str, int] = {}
        for rejection in self.rejections:
            num_rejected_by_cause[rejection.cause] = num_rejected_by_cause.get(rejection.cause, 0) + rejection.count
        return num_rejected_by_cause


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
                for cause, count in check_diagnostics.get_num_rejected_by_cause().items():
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
    """Diagnostics of several pool fills, such as the builds of a Run, with their counts summed."""

    fills: list[PlacementDiagnostics]
    """Diagnostics of each fill, in order, with every rejection the checks made."""

    @property
    def num_builds(self) -> int:
        """Fills whose diagnostics are held: one per build, and a failed refill when one ends a Run."""
        return len(self.fills)

    @property
    def num_envs(self) -> int:
        """Sum of the fills' num_envs."""
        return sum(fill.num_envs for fill in self.fills)

    @property
    def num_fallback_envs(self) -> int:
        """Environments given best-loss layouts, summed over the fills."""
        return sum(len(fill.fallback_envs) for fill in self.fills)

    @property
    def num_batches(self) -> int:
        """Batches the fills validated, summed over the fills."""
        return sum(len(fill.batches) for fill in self.fills)

    @property
    def batches(self) -> list[BatchValidationDiagnostics]:
        """Counts summed by batch position: entry i sums the i-th batch of every fill that validated one."""
        batches: list[BatchValidationDiagnostics] = []
        for fill in self.fills:
            for position, batch in enumerate(fill.batches):
                if position == len(batches):
                    batches.append(BatchValidationDiagnostics(num_candidates=0, num_valid=0, checks={}))
                _add_batch_counts(batches[position], batch)
        return batches

    def save(self, path: str | Path) -> None:
        """Write the diagnostics of every fill to a JSON file, replacing any existing file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AggregatedPlacementDiagnostics:
        """Return the diagnostics that save() wrote to path."""
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        fills = []
        for fill in content["fills"]:
            batches = []
            for batch in fill["batches"]:
                checks = {}
                for check, check_content in batch["checks"].items():
                    rejections = [Rejection(**rejection) for rejection in check_content["rejections"]]
                    checks[check] = ValidationCheckDiagnostics(
                        num_evaluated=check_content["num_evaluated"],
                        num_passed=check_content["num_passed"],
                        required=check_content["required"],
                        rejections=rejections,
                    )
                batches.append(
                    BatchValidationDiagnostics(
                        num_candidates=batch["num_candidates"], num_valid=batch["num_valid"], checks=checks
                    )
                )
            fills.append(
                PlacementDiagnostics(num_envs=fill["num_envs"], batches=batches, fallback_envs=fill["fallback_envs"])
            )
        return cls(fills=fills)


def _add_batch_counts(total: BatchValidationDiagnostics, batch: BatchValidationDiagnostics) -> None:
    """Add a batch's candidate counts and per-check counts and rejections to total, in place."""
    total.num_candidates += batch.num_candidates
    total.num_valid += batch.num_valid
    for check, check_diagnostics in batch.checks.items():
        total_check = total.checks.setdefault(
            check, ValidationCheckDiagnostics(num_evaluated=0, num_passed=0, required=check_diagnostics.required)
        )
        assert total_check.required == check_diagnostics.required, f"Check '{check}' is required in only some batches."
        total_check.num_evaluated += check_diagnostics.num_evaluated
        total_check.num_passed += check_diagnostics.num_passed
        total_check.rejections.extend(check_diagnostics.rejections)
