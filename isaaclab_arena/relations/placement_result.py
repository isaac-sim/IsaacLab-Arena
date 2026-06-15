# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults


@dataclass
class PlacementResult:
    """Result of an ObjectPlacer.place() call (single-env or multi-env).

    Use fields directly for single-env; iterate results for multi-env.
    On wrappers built by from_per_env, top-level fields mirror results[0].
    dataclasses.replace() does not preserve the wrapper; use
    from_per_env(original.results) to copy one.
    """

    validation_results: PlacementValidationResults
    """Validation checklist for the placement."""

    positions: dict[ObjectBase, tuple[float, float, float]]
    """Final positions for each object."""

    final_loss: float
    """Loss value of the final placement."""

    attempts: int
    """Number of attempts made."""

    orientations: dict[ObjectBase, float] = field(default_factory=dict)
    """Per-object yaw (radians) about the world up (Z) axis, composed on top of each object's
    base rotation. Keyed by object, like positions. Empty when unrotated."""

    _per_env_results: list[PlacementResult] | None = field(default=None, init=False, repr=False, compare=False)
    # dataclasses.replace() resets this to None; use from_per_env(self.results) to copy a wrapper.

    @classmethod
    def from_per_env(cls, results: list[PlacementResult]) -> PlacementResult:
        """Wrap per-env leaf results into a single PlacementResult.

        Args:
            results: One leaf PlacementResult per environment (not itself a wrapper).

        Returns:
            A PlacementResult whose top-level fields mirror results[0];
            iterate results to access all per-env layouts.
        """
        assert results, "from_per_env requires at least one result"
        assert all(
            r._per_env_results is None for r in results
        ), "from_per_env requires bare PlacementResult leaves; wrapping a wrapper is not supported"
        first = results[0]
        obj = cls(
            validation_results=first.validation_results,
            positions=first.positions,
            final_loss=first.final_loss,
            attempts=first.attempts,
            orientations=first.orientations,
        )
        obj._per_env_results = results
        return obj

    @property
    def results(self) -> list[PlacementResult]:
        """Per-env result list; [self] for a directly-constructed leaf."""
        return self._per_env_results if self._per_env_results is not None else [self]

    @property
    def success(self) -> bool:
        """True when every env passed all required validation checks.

        place() always returns a best-ranked layout per env even when validation fails;
        check this to distinguish a validated layout from a lowest-loss fallback.
        """
        if self._per_env_results is not None:
            return all(r.success for r in self._per_env_results)
        return self.validation_results.do_all_required_validation_checks_pass()
