# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Counts that describe how relation placement went: per validated batch, and per fill of a placement pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlacementBounds:
    """Serializable world-axis bounds used by a placement check."""

    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, list[float]]:
        """Return JSON-serializable bounds."""
        return {"minimum_m": list(self.minimum_m), "maximum_m": list(self.maximum_m)}


@dataclass(frozen=True)
class RelationRejectionDiagnostics:
    """Geometry details for one failed On or ClutterOn candidate relation."""

    check: str
    failing_test: str
    relation_type: str
    subject: str
    support: str
    env_id: int
    candidate_id: int
    child_bounds: PlacementBounds
    support_bounds: PlacementBounds
    signed_height_error_m: float
    requested_edge_margin_m: float
    feasible_edge_margin_m: float
    footprint_margins_m: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable rejection record."""
        return {
            "check": self.check,
            "failing_test": self.failing_test,
            "relation_type": self.relation_type,
            "subject": self.subject,
            "support": self.support,
            "env_id": self.env_id,
            "candidate_id": self.candidate_id,
            "child_bounds": self.child_bounds.to_dict(),
            "support_bounds": self.support_bounds.to_dict(),
            "signed_height_error_m": self.signed_height_error_m,
            "requested_edge_margin_m": self.requested_edge_margin_m,
            "feasible_edge_margin_m": self.feasible_edge_margin_m,
            "footprint_margins_m": self.footprint_margins_m,
        }

    def format(self) -> str:
        """Return a compact human-readable rejection explanation."""
        margins = ", ".join(f"{key}={value:.4f}" for key, value in self.footprint_margins_m.items())
        return (
            f"{self.check}/{self.failing_test}: env={self.env_id} candidate={self.candidate_id} "
            f"{self.subject} {self.relation_type}({self.support}); "
            f"height_error={self.signed_height_error_m:.4f} m, "
            f"edge_margin={self.requested_edge_margin_m:.4f} m "
            f"(feasible={self.feasible_edge_margin_m:.4f} m), footprint=[{margins}]"
        )


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

    relation_rejections: list[RelationRejectionDiagnostics] = field(default_factory=list)
    """Bounded details for failed On and ClutterOn candidates."""

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

    batches: list[BatchValidationDiagnostics] = field(default_factory=list)
    """Validated batches, in solve order. A batch after the first runs only while some environment lacks
    layouts."""

    fallback_envs: list[int] = field(default_factory=list)
    """Environments given best-loss layouts, which fail at least one required check."""

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate counts and bounded rejection details as JSON-compatible data."""
        return {
            "batches": [
                {
                    "num_candidates": batch.num_candidates,
                    "num_valid": batch.num_valid,
                    "checks": {
                        name: {
                            "num_evaluated": check.num_evaluated,
                            "num_passed": check.num_passed,
                            "num_rejected": check.num_rejected,
                            "required": check.required,
                        }
                        for name, check in batch.checks.items()
                    },
                    "relation_rejections": [rejection.to_dict() for rejection in batch.relation_rejections],
                }
                for batch in self.batches
            ],
            "fallback_envs": list(self.fallback_envs),
        }

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
        for index, batch in enumerate(self.batches):
            lines.extend(f"batch {index} rejection: {rejection.format()}" for rejection in batch.relation_rejections)
        return "\n".join(lines)
