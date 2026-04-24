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
    """Object placer that maintains a pool of valid placement layouts.

    Wraps :class:`ObjectPlacer` and solves layouts in batches of ``pool_size``,
    keeping only those that pass validation.  The pool is refilled automatically
    when consumed layouts run out.

    **Homogeneous mode** (default): all objects have the same geometry in
    every environment.  Layouts are stored in a single flat list and any
    layout can serve any environment.

    **Heterogeneous mode** (activated when any object has
    ``heterogeneous_bbox = True``, e.g. ``RigidObjectSet``): each
    environment has its own fixed set of object variants, assigned at
    build time.  Layouts are stored per ``env_id`` so that resets always
    return a layout solved for that environment's actual object geometry.

    * :meth:`sample_without_replacement` — returns the next *count* layouts
      sequentially.  Auto-refills when exhausted.  In heterogeneous mode,
      pass ``env_ids`` so each environment receives a matching layout.
    * :meth:`sample_with_replacement` — picks *count* layouts at random
      (non-consuming).  Used for static initial positions.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of layouts to solve per batch.
        num_envs: Total number of simulation environments.  Required for
            heterogeneous placement so per-env pools can be created.
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

        self._objects = objects
        self._placer = ObjectPlacer(params=placer_params)
        self._pool_size = pool_size
        self._heterogeneous = any(getattr(obj, "heterogeneous_bbox", False) for obj in objects)

        if self._heterogeneous:
            assert (
                num_envs is not None
            ), "num_envs is required for heterogeneous placement so per-env pools can be created."
            self._num_envs = num_envs

            self._layout_pools: dict[int, list[PlacementResult]] = {env_id: [] for env_id in range(num_envs)}
            self._layout_cursors: dict[int, int] = {env_id: 0 for env_id in range(num_envs)}

            self._solve_and_store_heterogeneous(pool_size)
            for env_id, pool in self._layout_pools.items():
                if not pool:
                    raise RuntimeError(
                        f"Placement pool failed to produce any valid layouts for env {env_id} "
                        f"from {pool_size} attempts. Check object relations and constraints."
                    )
        else:
            self._layouts: list[PlacementResult] = []
            self._next_idx: int = 0

            self._solve_and_store(pool_size)
            if not self._layouts:
                raise RuntimeError(
                    f"Placement pool failed to produce any valid layouts from {pool_size} attempts. "
                    "Check object relations and constraints."
                )

    @property
    def is_heterogeneous(self) -> bool:
        """Whether this pool operates in heterogeneous (per-env) mode."""
        return self._heterogeneous

    # ------------------------------------------------------------------
    # Homogeneous (flat pool) internals
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        """Drop consumed layouts and reset the read index to free memory."""
        self._layouts = self._layouts[self._next_idx :]
        self._next_idx = 0

    def _solve_and_store(self, num_layouts: int) -> None:
        """Solve *num_layouts* placements and append valid ones to the pool.

        When no candidates pass strict validation, the best-loss candidates are
        accepted with a warning (matching pre-pool behaviour where validation
        failures were non-fatal).
        """
        self._compact()

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
            self._layouts.extend(valid_results)
        else:
            print("Warning: No candidates passed strict validation. Accepting best-loss layouts as fallback.")
            self._layouts.extend(all_results)

    # ------------------------------------------------------------------
    # Heterogeneous (per-env pool) internals
    # ------------------------------------------------------------------

    def _compact_env_pool(self, env_id: int) -> None:
        """Drop consumed layouts for a single env and reset its cursor."""
        idx = self._layout_cursors[env_id]
        self._layout_pools[env_id] = self._layout_pools[env_id][idx:]
        self._layout_cursors[env_id] = 0

    def _solve_and_store_heterogeneous(self, num_layouts: int) -> None:
        """Solve layouts and store valid results into per-env pools.

        Each round solves ``num_envs`` layouts in one batched call.
        Result ``i`` is solved with env ``i``'s actual object geometry
        (from ``get_bounding_box_per_env(num_envs)``) and is stored
        directly into ``_layout_pools[i]``.  Multiple rounds are run
        until each env has enough candidates.
        """
        for env_id in self._layout_pools:
            self._compact_env_pool(env_id)

        num_rounds = max(1, num_layouts // self._num_envs)
        total_valid = 0
        total_solved = 0
        all_round_results: list[list[PlacementResult]] = []

        for _ in range(num_rounds):
            with torch.inference_mode(False):
                result = self._placer.place(self._objects, num_envs=self._num_envs, result_per_env=True)

            round_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
            all_round_results.append(round_results)
            total_solved += len(round_results)

            for env_id, r in enumerate(round_results):
                if r.success:
                    self._layout_pools[env_id].append(r)
                    total_valid += 1

        if total_valid < total_solved:
            failed_envs = [e for e in self._layout_pools if not self._layout_pools[e]]
            msg = (
                f"Placement pool (heterogeneous): solved {total_solved} candidates"
                f" across {num_rounds} rounds,"
                f" {total_valid} valid, {total_solved - total_valid} failed validation"
            )
            if failed_envs:
                msg += f". Envs with zero valid layouts: {failed_envs}"
            print(msg)

        # Per-env fallback: for any env that still has zero valid layouts,
        # accept the best-loss (lowest loss) result from all rounds.
        for env_id in range(self._num_envs):
            if self._layout_pools[env_id]:
                continue
            best: PlacementResult | None = None
            for round_results in all_round_results:
                if env_id < len(round_results):
                    r = round_results[env_id]
                    if best is None or r.final_loss < best.final_loss:
                        best = r
            if best is not None:
                print(
                    f"Warning: env {env_id} had no valid layouts; "
                    f"accepting best-loss fallback (loss={best.final_loss:.6f})."
                )
                self._layout_pools[env_id].append(best)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_without_replacement(
        self, count: int, env_ids: list[int] | torch.Tensor | None = None
    ) -> list[PlacementResult]:
        """Return the next *count* layouts sequentially (without replacement).

        Auto-refills the pool when there are not enough layouts ahead of the
        read index.

        In **heterogeneous mode** ``env_ids`` must be provided so each
        environment receives a layout matching its object geometry.

        Args:
            count: Number of layouts to return.
            env_ids: Environment indices being reset.  Required when the
                pool is heterogeneous; ignored otherwise.

        Raises:
            RuntimeError: If the pool cannot provide *count* layouts after refilling.
        """
        if self._heterogeneous:
            return self._sample_without_replacement_heterogeneous(count, env_ids)

        remaining = len(self._layouts) - self._next_idx
        if remaining < count:
            self._solve_and_store(max(self._pool_size, count))

        remaining = len(self._layouts) - self._next_idx
        if remaining < count:
            raise RuntimeError(
                f"Placement pool has {remaining} valid layouts but {count} were requested. "
                "The solver is not producing enough valid placements."
            )

        start = self._next_idx
        self._next_idx += count
        return self._layouts[start : self._next_idx]

    def _sample_without_replacement_heterogeneous(
        self, count: int, env_ids: list[int] | torch.Tensor | None
    ) -> list[PlacementResult]:
        """Draw one layout per requested env, refilling any depleted per-env pool."""
        assert env_ids is not None, "env_ids must be provided for heterogeneous placement pools."

        if isinstance(env_ids, torch.Tensor):
            ids: list[int] = [int(x) for x in env_ids]
        else:
            ids = list(env_ids)
        assert len(ids) == count

        # Refill any env pool that doesn't have enough layouts.
        demand_per_env: dict[int, int] = {}
        for env_id in ids:
            demand_per_env[env_id] = demand_per_env.get(env_id, 0) + 1

        needs_refill = False
        for env_id, demand in demand_per_env.items():
            available = len(self._layout_pools[env_id]) - self._layout_cursors[env_id]
            if available < demand:
                needs_refill = True
                break

        if needs_refill:
            max_demand = max(demand_per_env.values())
            self._solve_and_store_heterogeneous(max(self._pool_size, max_demand * self._num_envs))

        results: list[PlacementResult] = []
        for env_id in ids:
            idx = self._layout_cursors[env_id]
            if idx >= len(self._layout_pools[env_id]):
                raise RuntimeError(
                    f"Placement pool: env {env_id} has no more valid layouts. "
                    "The solver is not producing enough valid placements."
                )
            results.append(self._layout_pools[env_id][idx])
            self._layout_cursors[env_id] = idx + 1

        return results

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick *count* layouts at random with replacement (non-consuming).

        In **heterogeneous mode**, each position ``i`` in the returned
        list corresponds to env ``i`` and is drawn from that env's pool.

        Used by ``resolve_on_reset=False`` to assign initial positions
        that persist across resets.
        """
        if self._heterogeneous:
            return self._sample_with_replacement_heterogeneous(count)
        return random.choices(self._layouts, k=count)

    def _sample_with_replacement_heterogeneous(self, count: int) -> list[PlacementResult]:
        """Pick one random layout per env from its pool (non-consuming)."""
        results: list[PlacementResult] = []
        for env_id in range(count):
            pool = self._layout_pools[env_id]
            assert pool, f"Env {env_id} has no valid layouts to sample from."
            results.append(random.choice(pool))
        return results

    @property
    def remaining(self) -> int:
        """Number of layouts not yet consumed by :meth:`sample_without_replacement`.

        For heterogeneous pools, returns the minimum across all envs.
        """
        if self._heterogeneous:
            return min(len(self._layout_pools[e]) - self._layout_cursors[e] for e in self._layout_pools)
        return len(self._layouts) - self._next_idx
