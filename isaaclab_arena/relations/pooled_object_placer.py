# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class PooledObjectPlacer:
    """Object placer that keeps a pool of optimized layouts.

    Wraps :class:`ObjectPlacer` and solves object layouts in batches of
    ``pool_size``, keeping only those that pass validation. The pool is refilled
    automatically when consumed layouts run out.

    Layouts are always stored in per-env pools.  The public sampling methods
    expose only the replacement strategy; internally, samples are drawn in
    env-index order.

    * :meth:`sample_without_replacement` — returns the next *count* layouts
      sequentially.  Auto-refills when exhausted.
    * :meth:`sample_with_replacement` — picks *count* layouts at random
      (non-consuming).  Used for static initial positions.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of layouts to solve per batch.
        num_envs: Total number of simulation environments.  Required when
            layouts use env-specific object variants and defaults to 1 otherwise.
    """

    def __init__(
        self,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
        num_envs: int | None = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")

        self._objects = list(objects)
        self._placer = ObjectPlacer(params=placer_params)
        self._pool_size = pool_size
        self._uses_env_specific_bboxes = any(getattr(obj, "heterogeneous_bbox", False) for obj in objects)

        self._num_envs = num_envs if num_envs is not None else 1
        if self._num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self._num_envs}")
        if self._uses_env_specific_bboxes:
            assert num_envs is not None, "num_envs is required when layouts use env-specific object variants."
        self._layout_pools: dict[int, list[PlacementResult]] = {cur_env: [] for cur_env in range(self._num_envs)}
        self._layout_cursors: dict[int, int] = {cur_env: 0 for cur_env in range(self._num_envs)}

        self._solve_and_store(pool_size)
        for cur_env, pool in self._layout_pools.items():
            if not pool:
                raise RuntimeError(
                    f"Placement pool failed to produce any valid layouts for env {cur_env} "
                    f"from {pool_size} attempts. Check object relations and constraints."
                )

    # ------------------------------------------------------------------
    # Pool storage internals
    # ------------------------------------------------------------------

    def _discard_consumed_layouts(self) -> None:
        """Drop consumed layouts from every env pool before appending new layouts."""
        for cur_env in self._layout_pools:
            idx = self._layout_cursors[cur_env]
            self._layout_pools[cur_env] = self._layout_pools[cur_env][idx:]
            self._layout_cursors[cur_env] = 0

    def _solve_and_store(self, num_layouts: int) -> None:
        """Solve layouts and append complete per-env rounds to the pools."""
        self._discard_consumed_layouts()
        target_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        max_solve_batches = max(1, self._placer.params.max_placement_attempts)

        for _ in range(max_solve_batches):
            missing_per_env = [
                target_per_env - (len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env])
                for cur_env in range(self._num_envs)
            ]
            max_missing = max(missing_per_env)
            if max_missing <= 0:
                return

            batch_size = max_missing * self._num_envs
            if self._uses_env_specific_bboxes:
                all_results, layouts_per_env = self._solve_layouts_with_env_bboxes(batch_size)
                self._store_env_matched_results(all_results, layouts_per_env)
            else:
                layouts = self._solve_reusable_layouts(batch_size)
                self._store_reusable_results(layouts)

            available = [
                len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env] for cur_env in range(self._num_envs)
            ]
            if min(available) >= target_per_env:
                return

        available = [
            len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env] for cur_env in range(self._num_envs)
        ]
        raise RuntimeError(
            f"Placement pool could not fill {target_per_env} layouts per env after "
            f"{max_solve_batches} solve batches. Available per env: {available}."
        )

    def _solve_reusable_layouts(self, num_layouts: int) -> list[PlacementResult]:
        """Solve layouts that can be used by any env pool.

        When no layouts pass strict validation, the best-loss layouts are
        accepted with a warning (matching pre-pool behaviour where validation
        failures were non-fatal).
        """
        with torch.inference_mode(False):
            result = self._placer.place(self._objects, num_envs=num_layouts, result_per_env=True)

        # TODO(@zhx06): Simplify once ObjectPlacer.place() always returns MultiEnvPlacementResult.
        all_layouts = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        valid_layouts = [layout for layout in all_layouts if layout.success]

        if len(valid_layouts) < num_layouts:
            print(
                f"Pooled object placer: solved {num_layouts} layouts,"
                f" {len(valid_layouts)} valid, {num_layouts - len(valid_layouts)} failed validation"
            )

        if valid_layouts:
            return valid_layouts

        print("Warning: No candidates passed strict validation. Accepting best-loss layouts as fallback.")
        return all_layouts

    def _store_reusable_results(self, layouts: list[PlacementResult]) -> None:
        """Distribute reusable layouts across env pools without dropping valid results."""
        if not layouts:
            return

        for layout in layouts:
            cur_env = min(
                range(self._num_envs),
                key=lambda cur_env: len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env],
            )
            self._layout_pools[cur_env].append(layout)

    def _solve_layouts_with_env_bboxes(self, num_layouts: int) -> tuple[list[PlacementResult], int]:
        """Solve layouts tied to each env's actual object geometry.

        Computes bounding boxes for the real ``num_envs`` once, tiles them
        to ``num_layouts`` entries, and solves everything in **one** batched
        ``place()`` call.  Result ``i`` is mapped back to real env
        ``i % num_envs`` for pool storage.
        """
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

        layouts_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        total_layouts = layouts_per_env * self._num_envs

        real_bboxes: dict = {obj: obj.get_bounding_box_per_env(self._num_envs) for obj in self._objects}

        tiled_bboxes: dict = {}
        for obj, bbox in real_bboxes.items():
            # (num_envs, 3) -> repeat each env's row layouts_per_env times -> (total_layouts, 3)
            min_pt = bbox.min_point.repeat_interleave(layouts_per_env, dim=0)
            max_pt = bbox.max_point.repeat_interleave(layouts_per_env, dim=0)
            tiled_bboxes[obj] = AxisAlignedBoundingBox(min_point=min_pt, max_point=max_pt)

        with torch.inference_mode(False):
            result = self._placer.place(
                self._objects,
                num_envs=total_layouts,
                result_per_env=True,
                env_bboxes=tiled_bboxes,
            )

        all_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        return all_results, layouts_per_env

    def _store_env_matched_results(self, all_results: list[PlacementResult], layouts_per_env: int) -> None:
        """Store layouts into the env pools they were solved for."""
        total_valid = 0
        for i, r in enumerate(all_results):
            cur_env = i // layouts_per_env
            if r.success:
                self._layout_pools[cur_env].append(r)
                total_valid += 1

        total_solved = len(all_results)
        if total_valid < total_solved:
            failed_envs = [cur_env for cur_env in self._layout_pools if not self._layout_pools[cur_env]]
            msg = (
                f"Placement pool (env-matched layouts): solved {total_solved} candidates,"
                f" {total_valid} valid, {total_solved - total_valid} failed validation"
            )
            if failed_envs:
                msg += f". Envs with zero valid layouts: {failed_envs}"
            print(msg)

        for cur_env in range(self._num_envs):
            best: PlacementResult | None = None
            had_valid = False
            start = cur_env * layouts_per_env
            end = start + layouts_per_env
            for r in all_results[start:end]:
                if r.success:
                    had_valid = True
                    continue
                if best is None or r.final_loss < best.final_loss:
                    best = r
            if not had_valid and best is not None:
                print(
                    f"Warning: env {cur_env} had too few valid layouts; "
                    f"accepting best-loss fallback (loss={best.final_loss:.6f})."
                )
                self._layout_pools[cur_env].append(best)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        """Return the next *count* layouts sequentially (without replacement).

        Auto-refills any env pool that does not have enough layouts ahead of
        its read cursor.

        Args:
            count: Number of layouts to return.

        Raises:
            ValueError: If *count* is not a complete env round.
            RuntimeError: If the pool cannot provide *count* layouts after refilling.
        """
        if count % self._num_envs != 0:
            raise ValueError(f"count must be a multiple of num_envs ({self._num_envs}), got {count}")

        sample_env_order = [i % self._num_envs for i in range(count)]
        layouts_per_env = count // self._num_envs
        needs_refill = any(
            len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env] < layouts_per_env
            for cur_env in range(self._num_envs)
        )

        if needs_refill:
            self._solve_and_store(max(self._pool_size, count))

        results: list[PlacementResult] = []
        for cur_env in sample_env_order:
            idx = self._layout_cursors[cur_env]
            if idx >= len(self._layout_pools[cur_env]):
                raise RuntimeError(
                    f"Placement pool: env {cur_env} has no more valid layouts. "
                    "The solver is not producing enough valid placements."
                )
            results.append(self._layout_pools[cur_env][idx])
            self._layout_cursors[cur_env] = idx + 1

        return results

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick *count* layouts at random with replacement (non-consuming).

        Each returned layout is drawn from the per-env pool corresponding to
        its position in the requested batch.

        Used by ``resolve_on_reset=False`` to assign initial positions
        that persist across resets.
        """
        sample_env_order = [i % self._num_envs for i in range(count)]
        results: list[PlacementResult] = []
        for cur_env in sample_env_order:
            pool = self._layout_pools[cur_env]
            assert pool, f"Env {cur_env} has no valid layouts to sample from."
            results.append(random.choice(pool))
        return results

    @property
    def remaining(self) -> int:
        """Number of layouts not yet consumed by :meth:`sample_without_replacement`.

        Reports the minimum available count across env pools so every env has
        the same without-replacement capacity.
        """
        return min(len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env] for cur_env in self._layout_pools)

    @property
    def pool_size(self) -> int:
        """Number of layouts to solve per batch. When the pool runs low, it will solve at least this number of layouts so future samples can reuse the buffer."""
        return self._pool_size
