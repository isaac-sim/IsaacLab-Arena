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
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class PooledObjectPlacer:
    """Object placer that maintains a pool of valid placement layouts.

    Storage: ``num_envs`` independent layout pools, each with its own read
    cursor (this replaces the single ``_layouts`` list + ``_next_idx`` cursor
    used before heterogeneous placement). Env-specific layouts are solved
    against a fixed env's object geometry and must be sampled in complete env
    rounds. Reusable layouts can be consumed one at a time.

    The pool is refilled automatically when an env's queue runs out.

    * :meth:`sample_without_replacement` — returns the next *count* layouts.
      Env-specific layouts require ``count`` to be a multiple of ``num_envs``.
    * :meth:`sample_with_replacement` — picks *count* layouts at random per
      env-slot (non-consuming). Used for static initial positions.

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
        # 1. Validate params.
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        # ``has_env_specific_bboxes`` is duck-typed (set on RigidObjectSet / DummyObject
        # but not declared on the abstract ObjectBase), so read it via getattr.
        self._uses_env_specific_bboxes = any(getattr(obj, "has_env_specific_bboxes", False) for obj in objects)
        if self._uses_env_specific_bboxes:
            assert num_envs is not None, "num_envs is required when layouts use env-specific object variants."
        self._num_envs = num_envs if num_envs is not None else 1
        if self._num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self._num_envs}")

        # 2. Configure dependencies and per-env storage.
        self._objects = objects
        self._placer = ObjectPlacer(params=placer_params)
        self._pool_size = pool_size
        self._layout_pools: dict[int, list[PlacementResult]] = {cur_env: [] for cur_env in range(self._num_envs)}
        self._layout_cursors: dict[int, int] = {cur_env: 0 for cur_env in range(self._num_envs)}

        # 3. Solve the initial pool and assert every env has at least one layout.
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

    def _available_per_env(self) -> list[int]:
        """Number of unread layouts in each env's pool (length ``num_envs``)."""
        return [len(self._layout_pools[cur_env]) - self._layout_cursors[cur_env] for cur_env in range(self._num_envs)]

    def _total_available(self) -> int:
        """Total unread layouts across all env pools."""
        return sum(self._available_per_env())

    def _discard_consumed_layouts(self) -> None:
        """Drop consumed layouts from every env pool before appending new layouts."""
        for cur_env in self._layout_pools:
            idx = self._layout_cursors[cur_env]
            self._layout_pools[cur_env] = self._layout_pools[cur_env][idx:]
            self._layout_cursors[cur_env] = 0

    def _solve_and_store(self, num_layouts: int) -> None:
        """Solve layouts in batches until every env has ``target_per_env`` unread layouts.

        Each batch contributes (roughly) one round of layouts per env. The
        outer loop is bounded by ``max_placement_attempts`` to avoid an
        unbounded refill in pathological configurations.
        """
        self._discard_consumed_layouts()
        target_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        max_solve_batches = max(1, self._placer.params.max_placement_attempts)

        for _ in range(max_solve_batches):
            max_missing = target_per_env - min(self._available_per_env())
            if max_missing <= 0:
                return

            batch_size = max_missing * self._num_envs
            if self._uses_env_specific_bboxes:
                all_results, layouts_per_env = self._solve_layouts_with_env_bboxes(batch_size)
                self._store_env_matched_results(all_results, layouts_per_env)
            else:
                layouts = self._solve_reusable_layouts(batch_size)
                self._store_reusable_results(layouts)

            if min(self._available_per_env()) >= target_per_env:
                return

        raise RuntimeError(
            f"Placement pool could not fill {target_per_env} layouts per env after "
            f"{max_solve_batches} solve batches. Available per env: {self._available_per_env()}."
        )

    def _solve_reusable_layouts(self, num_layouts: int) -> list[PlacementResult]:
        """Solve layouts that can be used by any env pool.

        When no candidates pass strict validation, the best-loss candidates are
        accepted with a warning (matching pre-pool behaviour where validation
        failures were non-fatal).
        """
        with torch.inference_mode(False):
            result = self._placer.place(self._objects, num_envs=num_layouts, result_per_env=True)

        all_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        valid_results = [r for r in all_results if r.success]

        if len(valid_results) < num_layouts:
            print(
                f"Placement pool: solved {num_layouts} candidates,"
                f" {len(valid_results)} valid, {num_layouts - len(valid_results)} failed validation"
            )

        if valid_results:
            return valid_results

        print("Warning: No candidates passed strict validation. Accepting best-loss layouts as fallback.")
        return all_results

    def _store_reusable_results(self, layouts: list[PlacementResult]) -> None:
        """Distribute reusable layouts across env pools using greedy shortest-first.

        Layouts produced by ``_solve_reusable_layouts`` are interchangeable
        across envs, so we place each one into whichever pool currently has
        the fewest unread layouts. This keeps reusable capacity balanced
        across env pools.
        """
        if not layouts:
            return

        available = self._available_per_env()
        for layout in layouts:
            cur_env = min(range(self._num_envs), key=available.__getitem__)
            self._layout_pools[cur_env].append(layout)
            available[cur_env] += 1

    def _solve_layouts_with_env_bboxes(self, num_layouts: int) -> tuple[list[PlacementResult], int]:
        """Solve layouts tied to each env's actual object geometry.

        Computes bounding boxes for the real ``num_envs`` once, tiles them
        to ``num_layouts`` entries, and solves everything in **one** batched
        ``place()`` call.  Result ``i`` is mapped back to real env
        ``i // layouts_per_env`` for pool storage.
        """
        layouts_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        total_layouts = layouts_per_env * self._num_envs

        real_bboxes = {obj: obj.get_bounding_box_per_env(self._num_envs) for obj in self._objects}

        # (num_envs, 3) -> repeat each env's row layouts_per_env times -> (total_layouts, 3).
        tiled_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] = {
            obj: AxisAlignedBoundingBox(
                min_point=bbox.min_point.repeat_interleave(layouts_per_env, dim=0),
                max_point=bbox.max_point.repeat_interleave(layouts_per_env, dim=0),
            )
            for obj, bbox in real_bboxes.items()
        }

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
        """Store env-matched results into per-env pools, with a best-loss fallback.

        Two passes:
        1. Append every successful result to its env's pool.
        2. For any env whose block produced zero successful results, append
           the block's best-loss candidate (with a warning).
        """
        # Pass 1: store successful layouts.
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
                f"Placement pool (env-specific bbox layouts): solved {total_solved} candidates,"
                f" {total_valid} valid, {total_solved - total_valid} failed validation"
            )
            if failed_envs:
                msg += f". Envs with zero valid layouts: {failed_envs}"
            print(msg)

        # Pass 2: best-loss fallback for empty env blocks.
        for cur_env in range(self._num_envs):
            start = cur_env * layouts_per_env
            env_block = all_results[start : start + layouts_per_env]
            if any(r.success for r in env_block):
                continue
            best = min(env_block, key=lambda r: r.final_loss, default=None)
            if best is None:
                continue
            print(
                f"Warning: env {cur_env} had too few valid layouts; "
                f"accepting best-loss fallback (loss={best.final_loss:.6f})."
            )
            self._layout_pools[cur_env].append(best)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        """Return the next *count* layouts.

        Env-specific layouts are returned as complete rounds of
        ``[env_0, env_1, ..., env_{num_envs-1}]`` so each result still maps
        to the absolute environment it was solved for. Reusable layouts are
        interchangeable and consume only ``count`` entries.

        Args:
            count: Number of layouts to return.

        Raises:
            ValueError: If env-specific layouts are requested without a complete env round.
            RuntimeError: If the pool cannot provide *count* layouts after refilling.
        """
        if self._uses_env_specific_bboxes:
            return self._sample_env_indexed_without_replacement(count)
        return self._sample_reusable_without_replacement(count)

    def _sample_env_indexed_without_replacement(self, count: int) -> list[PlacementResult]:
        """Consume complete env rounds for layouts tied to absolute env ids."""
        if count % self._num_envs != 0:
            raise ValueError(f"count must be a multiple of num_envs ({self._num_envs}), got {count}")

        layouts_per_env = count // self._num_envs
        if min(self._available_per_env()) < layouts_per_env:
            self._solve_and_store(max(self._pool_size, count))

        results: list[PlacementResult] = []
        for _ in range(layouts_per_env):
            for cur_env in range(self._num_envs):
                idx = self._layout_cursors[cur_env]
                if idx >= len(self._layout_pools[cur_env]):
                    raise RuntimeError(
                        f"Placement pool: env {cur_env} has no more valid layouts. "
                        "The solver is not producing enough valid placements."
                    )
                results.append(self._layout_pools[cur_env][idx])
                self._layout_cursors[cur_env] = idx + 1
        return results

    def _sample_reusable_without_replacement(self, count: int) -> list[PlacementResult]:
        """Consume exactly ``count`` interchangeable layouts."""
        if self._total_available() < count:
            self._solve_and_store(max(self._pool_size, count))

        available = self._available_per_env()
        if sum(available) < count:
            raise RuntimeError(
                f"Placement pool has {sum(available)} reusable layouts but {count} were requested. "
                "The solver is not producing enough valid placements."
            )

        results: list[PlacementResult] = []
        for _ in range(count):
            cur_env = max(range(self._num_envs), key=available.__getitem__)
            idx = self._layout_cursors[cur_env]
            if idx >= len(self._layout_pools[cur_env]):
                raise RuntimeError(
                    f"Placement pool: env {cur_env} has no more valid layouts. "
                    "The solver is not producing enough valid placements."
                )
            results.append(self._layout_pools[cur_env][idx])
            self._layout_cursors[cur_env] = idx + 1
            available[cur_env] -= 1
        return results

    @property
    def requires_env_indexed_layouts(self) -> bool:
        """Whether sampled layouts must be matched back to absolute env ids."""
        return self._uses_env_specific_bboxes

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick *count* layouts at random per env-slot (non-consuming).

        Slot ``i`` is filled by a random pick from env ``i % num_envs``'s
        pool, so a length-``count`` request walks env slots in order. Used
        by ``resolve_on_reset=False`` to assign initial positions that persist
        across resets.
        """
        results: list[PlacementResult] = []
        for i in range(count):
            cur_env = i % self._num_envs
            pool = self._layout_pools[cur_env]
            assert pool, f"Env {cur_env} has no valid layouts to sample from."
            results.append(random.choice(pool))
        return results

    @property
    def remaining(self) -> int:
        """Number of complete env rounds available to :meth:`sample_without_replacement`.

        Returns the minimum unread count across env pools (the previous
        ``remaining`` was a total across one shared list; under per-env
        storage a single round consumes one layout from every env, so the
        minimum is what limits without-replacement capacity).
        """
        return min(self._available_per_env())

    @property
    def total_remaining(self) -> int:
        """Total unread layouts across all env pools."""
        return self._total_available()
