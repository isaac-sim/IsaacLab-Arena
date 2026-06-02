# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass(frozen=True)
class ValidationReport:
    """Per-check outcome of placement validation.

    checks maps each check name (e.g. "no_overlap", "on_relations") to its pass/fail result.
    """

    checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        # Read-only snapshot: neither the caller's original dict nor report.checks[...] can mutate it.
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))

    def __reduce__(self):
        # MappingProxyType can't be pickled/deepcopied, and Isaac Lab deepcopies the EventTermCfg
        # params that carry this report; rebuild from a plain dict so copy/pickle round-trip.
        return (self.__class__, (dict(self.checks),))

    @property
    def passed(self) -> bool:
        """True only when at least one check ran and every check passed (empty fails closed)."""
        return bool(self.checks) and all(self.checks.values())

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of the checks that failed, in insertion order."""
        return tuple(name for name, ok in self.checks.items() if not ok)


LayoutFilter = Callable[[ValidationReport], bool]
"""Acceptance predicate: given a layout's ValidationReport, whether the layout is kept."""


def default_layout_filter(report: ValidationReport) -> bool:
    """Default acceptance: keep a layout iff every built-in check passed."""
    return report.passed


@dataclass
class PlacementResult:
    """Result of an ObjectPlacer.place() call."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""

    validation: ValidationReport
    """Per-check validation outcome; success is derived from it."""

    orientations: dict[ObjectBase, float] = field(default_factory=dict)
    """Per-object yaw (radians) about the world up (Z) axis, composed on top of each object's
    base rotation. Keyed by object, like positions. Empty when unrotated."""

    @property
    def success(self) -> bool:
        """Whether placement passed validation checks."""
        return self.validation.passed


@dataclass
class MultiEnvPlacementResult:
    """Result of an ObjectPlacer.place() call for multiple environments."""

    results: list[PlacementResult]
    """One PlacementResult per environment (same length as num_envs)."""

    @property
    def success(self) -> bool:
        """True if every environment's placement succeeded."""
        return all(r.success for r in self.results)

    @property
    def attempts(self) -> int:
        """Number of attempts (same for all envs in the batched run)."""
        return self.results[0].attempts if self.results else 0
