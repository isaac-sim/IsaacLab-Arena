# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Counts that describe how relation placement went: per validated batch, and per fill of a placement pool."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


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

    def format_passed_per_check(self) -> str:
        """Return each check's passed and evaluated counts on one line, such as "on_relation=49/50, no_overlap=43/50"."""
        return ", ".join(
            f"{check}={check_diagnostics.num_passed}/{check_diagnostics.num_evaluated}"
            for check, check_diagnostics in self.checks.items()
        )


@dataclass
class PlacementDiagnostics:
    """How one fill of a placement pool went: every batch it validated, in order, and the environments it filled
    with best-loss layouts because none of their candidates was valid."""

    num_envs: int
    """Environments the pool fills."""

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
                f" {batch.format_passed_per_check()}"
            )
        if self.fallback_envs:
            lines.append(f"best-loss fallback for envs: {self.fallback_envs}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write the diagnostics to a JSON file, replacing any existing file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PlacementDiagnostics:
        """Read diagnostics written by save()."""
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        batches = []
        for batch in content["batches"]:
            checks = {check: ValidationCheckDiagnostics(**counts) for check, counts in batch["checks"].items()}
            batches.append(BatchValidationDiagnostics(batch["num_candidates"], batch["num_valid"], checks))
        return cls(content["num_envs"], batches, content["fallback_envs"])
