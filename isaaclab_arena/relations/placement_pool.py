# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class PlacementPool:
    """Pre-solved pool of valid placement layouts.

    Solves ``pool_size`` layouts at build time and keeps only those that
    pass validation.  Valid layouts are shuffled so successive draws
    provide variety across resets.

    * :meth:`acquire` — returns the next *count* layouts sequentially
      (without replacement).  Auto-refills when exhausted.
    * :meth:`sample` — picks *count* layouts at random with replacement
      (non-consuming).  Used for static initial positions.

    Internally the pool is a single list with a read cursor, following the
    same pattern as ``stable-baselines3.ReplayBuffer`` (single storage +
    position index).

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of layouts to solve per batch.
    """

    def __init__(
        self,
        objects: Sequence[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
    ) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")

        self._objects = list(objects)
        self._placer = ObjectPlacer(params=placer_params)
        self._pool_size = pool_size
        self._layouts: list[PlacementResult] = []
        self._cursor: int = 0

        self._solve_and_store(pool_size)
        if not self._layouts:
            raise RuntimeError(
                f"Placement pool failed to produce any valid layouts from {pool_size} attempts. "
                "Check object relations and constraints."
            )

    def _compact(self) -> None:
        """Drop consumed layouts and reset the cursor to free memory."""
        self._layouts = self._layouts[self._cursor :]
        self._cursor = 0

    def _solve_and_store(self, num_layouts: int) -> None:
        """Solve *num_layouts* placements and append valid ones to the pool."""
        self._compact()

        with torch.inference_mode(False):
            result = self._placer.place(self._objects, num_envs=num_layouts, result_per_env=True)

        # TODO(@zhx06): Simplify once ObjectPlacer.place() always returns MultiEnvPlacementResult.
        all_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        valid_results = [r for r in all_results if r.success]

        if len(valid_results) < num_layouts:
            print(
                f"[WARNING] Placement pool: solved {num_layouts}, got {len(valid_results)} valid."
                f" {num_layouts - len(valid_results)} layouts failed validation."
            )

        random.shuffle(valid_results)
        self._layouts.extend(valid_results)

    def acquire(self, count: int) -> list[PlacementResult]:
        """Return the next *count* layouts sequentially (without replacement).

        Auto-refills the pool when there are not enough layouts ahead of the cursor.

        Raises:
            RuntimeError: If the pool cannot provide *count* layouts after refilling.
        """
        remaining = len(self._layouts) - self._cursor
        if remaining < count:
            self._solve_and_store(max(self._pool_size, count))

        remaining = len(self._layouts) - self._cursor
        if remaining < count:
            raise RuntimeError(
                f"Placement pool has {remaining} valid layouts but {count} were requested. "
                "The solver is not producing enough valid placements."
            )

        start = self._cursor
        self._cursor += count
        return self._layouts[start : self._cursor]

    def sample(self, count: int) -> list[PlacementResult]:
        """Pick *count* layouts at random with replacement (non-consuming).

        Used by ``resolve_on_reset=False`` to assign initial positions
        that persist across resets.
        """
        return random.choices(self._layouts, k=count)

    def __len__(self) -> int:
        """Number of layouts ahead of the cursor (available for :meth:`acquire`)."""
        return len(self._layouts) - self._cursor
