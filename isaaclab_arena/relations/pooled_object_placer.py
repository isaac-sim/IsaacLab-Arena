# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class PooledObjectPlacer:
    """Object placer that keeps a pool of optimized placement candidates.

    Wraps :class:`ObjectPlacer` and solves candidate layouts in batches of
    ``pool_size``, keeping only those that pass validation. The pool is refilled
    automatically when consumed candidates run out.

    * :meth:`sample_without_replacement` — returns the next *count* candidates
      sequentially.  Auto-refills when exhausted.
    * :meth:`sample_with_replacement` — picks *count* candidates at random
      (non-consuming).  Used for static initial positions.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of placement candidates to solve per batch.
        candidate_validator: Optional callback invoked on every sampled candidate.
    """

    def __init__(
        self,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
        candidate_validator: Callable[[PlacementResult], None] | None = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")

        self._objects = objects
        self._placer = ObjectPlacer(params=placer_params)
        self._pool_size = pool_size
        self._candidate_validator = candidate_validator
        self._candidates: list[PlacementResult] = []
        self._next_idx: int = 0

        # Pre-solve the initial batch (runs the gradient solver, no simulation is needed).
        self._solve_and_store(pool_size)
        if not self._candidates:
            raise RuntimeError(
                f"Pooled object placer failed to produce any valid candidates from {pool_size} attempts. "
                "Check object relations and constraints."
            )

    def _compact(self) -> None:
        """Drop consumed candidates and reset the read index to free memory."""
        self._candidates = self._candidates[self._next_idx :]
        self._next_idx = 0

    def _solve_and_store(self, num_candidates: int) -> None:
        """Solve *num_candidates* placements and append valid ones to the pool.

        When no candidates pass strict validation, the best-loss candidates are
        accepted with a warning (matching pre-pool behaviour where validation
        failures were non-fatal).
        """
        self._compact()

        # place() runs: random init → gradient solve → validate → rank.
        # It returns up to num_candidates results; some may fail validation.
        with torch.inference_mode(False):
            result = self._placer.place(self._objects, num_envs=num_candidates, result_per_env=True)

        # TODO(@zhx06): Simplify once ObjectPlacer.place() always returns MultiEnvPlacementResult.
        all_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        valid_results = [r for r in all_results if r.success]

        if len(valid_results) < num_candidates:
            print(
                f"Pooled object placer: solved {num_candidates} candidates,"
                f" {len(valid_results)} valid, {num_candidates - len(valid_results)} failed validation"
            )

        if valid_results:
            self._candidates.extend(valid_results)
        else:
            print("Warning: No candidates passed strict validation. Accepting best-loss candidates as fallback.")
            self._candidates.extend(all_results)

    def _validate_sampled_candidates(self, candidates: list[PlacementResult]) -> None:
        """Run the optional validation callback on each sampled candidate."""
        if self._candidate_validator is None:
            return
        for candidate in candidates:
            self._candidate_validator(candidate)

    def _ensure_candidates_available(self, count: int) -> None:
        """Refill the pool when fewer than *count* candidates are available."""
        if self.remaining < count:
            self._solve_and_store(max(self._pool_size, count))

        if self.remaining < count:
            raise RuntimeError(
                f"Pooled object placer has {self.remaining} valid candidates but {count} were requested. "
                "The solver is not producing enough valid placements."
            )

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        """Return the next *count* candidates sequentially (without replacement).

        Auto-refills the pool when there are not enough candidates ahead of the read index.

        Raises:
            RuntimeError: If the pool cannot provide *count* candidates after refilling.
        """
        self._ensure_candidates_available(count)

        start = self._next_idx
        self._next_idx += count
        candidates = self._candidates[start : self._next_idx]
        self._validate_sampled_candidates(candidates)
        return candidates

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick *count* candidates at random with replacement (non-consuming).

        Used by ``resolve_on_reset=False`` to assign initial positions
        that persist across resets.
        """
        candidates = random.choices(self._candidates, k=count)
        self._validate_sampled_candidates(candidates)
        return candidates

    @property
    def remaining(self) -> int:
        """Number of candidates not yet consumed by :meth:`sample_without_replacement`."""
        return len(self._candidates) - self._next_idx

    @property
    def pool_size(self) -> int:
        """Number of candidates requested for each pool refill solve."""
        return self._pool_size
