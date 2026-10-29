# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Counts that describe how relation placement went: per validated batch, and per fill of a placement pool."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def format_check_counts(self) -> str:
        """Return each check's passed and evaluated counts on one line, such as "on_relation=49/50, no_overlap=43/50"."""
        return ", ".join(f"{check}={counts.num_passed}/{counts.num_evaluated}" for check, counts in self.checks.items())


@dataclass
class PlacementDiagnostics:
    """How one fill of a placement pool went: every batch it validated, in order, and the environments it filled
    with best-loss layouts because none of their candidates was valid."""

    batches: list[BatchValidationDiagnostics] = field(default_factory=list)
    """Validated batches, in solve order. A batch after the first runs only while some environment lacks
    layouts."""

    fallback_envs: list[int] = field(default_factory=list)
    """Environments given best-loss layouts, which fail at least one required check."""

    def format_summary(self) -> str:
        """Return one line per batch with its valid and per-check counts, then the fallback environments, if any."""
        lines = []
        for index, batch in enumerate(self.batches):
            lines.append(
                f"batch {index}: {batch.num_valid}/{batch.num_candidates} valid; passed per check:"
                f" {batch.format_check_counts()}"
            )
        if self.fallback_envs:
            lines.append(f"best-loss fallback for envs: {self.fallback_envs}")
        return "\n".join(lines)
