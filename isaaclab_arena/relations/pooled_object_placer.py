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
    ``heterogeneous_bbox = True``, e.g. ``RigidObjectSet``): each layout
    is tied to a specific *variant index* (``env_idx % num_variants``).
    Layouts are bucketed into per-variant sub-pools so that
    :meth:`sample_without_replacement` and :meth:`sample_with_replacement`
    always return a layout that matches the requesting environment's
    variant geometry.

    * :meth:`sample_without_replacement` — returns the next *count* layouts
      sequentially.  Auto-refills when exhausted.  In heterogeneous mode,
      pass ``env_ids`` so each environment receives a layout matching its
      variant.
    * :meth:`sample_with_replacement` — picks *count* layouts at random
      (non-consuming).  Used for static initial positions.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of layouts to solve per batch.
        num_envs: Total number of simulation environments.  Required for
            heterogeneous placement so variant indices can be resolved.
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
        self._heterogeneous = any(getattr(obj, "heterogeneous_bbox", False) for obj in objects)

        if self._heterogeneous:
            assert (
                num_envs is not None
            ), "num_envs is required for heterogeneous placement pools so variant indices can be resolved."
            self._num_envs = num_envs
            self._num_variants = self._detect_num_variants(objects)

            self._variant_layouts: dict[int, list[PlacementResult]] = {v: [] for v in range(self._num_variants)}
            self._variant_next_idx: dict[int, int] = {v: 0 for v in range(self._num_variants)}

            self._solve_and_store_heterogeneous(pool_size)
            for v in range(self._num_variants):
                if not self._variant_layouts[v]:
                    raise RuntimeError(
                        f"Placement pool failed to produce any valid layouts for variant {v} "
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
        """Whether this pool operates in heterogeneous (per-variant) mode."""
        return self._heterogeneous

    # ------------------------------------------------------------------
    # Variant helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_num_variants(objects: list[ObjectBase]) -> int:
        """Return the number of unique object variants across heterogeneous objects."""
        for obj in objects:
            if getattr(obj, "heterogeneous_bbox", False):
                return len(obj.objects)  # type: ignore[attr-defined]
        return 1

    def _variant_for_env(self, env_id: int) -> int:
        """Map an environment index to its variant index."""
        return env_id % self._num_variants

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
    # Heterogeneous (per-variant sub-pool) internals
    # ------------------------------------------------------------------

    def _compact_variant(self, variant: int) -> None:
        """Drop consumed layouts for a single variant and reset its read index."""
        idx = self._variant_next_idx[variant]
        self._variant_layouts[variant] = self._variant_layouts[variant][idx:]
        self._variant_next_idx[variant] = 0

    def _solve_and_store_heterogeneous(self, num_layouts: int) -> None:
        """Solve layouts and bucket valid results by variant index.

        Each result ``i`` from the solver corresponds to variant
        ``i % num_variants`` because ``get_bounding_box_per_env`` assigns
        variants in round-robin order.
        """
        for v in range(self._num_variants):
            self._compact_variant(v)

        with torch.inference_mode(False):
            result = self._placer.place(self._objects, num_envs=num_layouts, result_per_env=True)

        all_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]

        total_valid = 0
        for i, r in enumerate(all_results):
            if r.success:
                variant = i % self._num_variants
                self._variant_layouts[variant].append(r)
                total_valid += 1

        if total_valid < num_layouts:
            print(
                f"Placement pool (heterogeneous): solved {num_layouts} candidates,"
                f" {total_valid} valid, {num_layouts - total_valid} failed validation"
            )

        for v in range(self._num_variants):
            random.shuffle(self._variant_layouts[v])

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
        environment receives a layout matching its variant geometry.

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
        assert env_ids is not None, "env_ids must be provided for heterogeneous placement pools."

        if isinstance(env_ids, torch.Tensor):
            ids: list[int] = [int(x) for x in env_ids]
        else:
            ids = list(env_ids)
        assert len(ids) == count

        variant_demand: dict[int, int] = {}
        for eid in ids:
            v = self._variant_for_env(eid)
            variant_demand[v] = variant_demand.get(v, 0) + 1

        for v, demand in variant_demand.items():
            avail = len(self._variant_layouts[v]) - self._variant_next_idx[v]
            if avail < demand:
                refill = max(self._pool_size, demand * self._num_variants)
                self._solve_and_store_heterogeneous(refill)

        results: list[PlacementResult] = []
        for eid in ids:
            v = self._variant_for_env(eid)
            idx = self._variant_next_idx[v]
            if idx >= len(self._variant_layouts[v]):
                raise RuntimeError(
                    f"Placement pool: variant {v} has no more valid layouts "
                    f"(needed for env {eid}). The solver is not producing enough valid placements."
                )
            results.append(self._variant_layouts[v][idx])
            self._variant_next_idx[v] = idx + 1

        return results

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick *count* layouts at random with replacement (non-consuming).

        In **heterogeneous mode**, each position ``i`` in the returned
        list corresponds to env ``i`` and is drawn from the sub-pool
        matching that env's variant (``i % num_variants``).

        Used by ``resolve_on_reset=False`` to assign initial positions
        that persist across resets.
        """
        if self._heterogeneous:
            return self._sample_with_replacement_heterogeneous(count)
        return random.choices(self._layouts, k=count)

    def _sample_with_replacement_heterogeneous(self, count: int) -> list[PlacementResult]:
        results: list[PlacementResult] = []
        for env_idx in range(count):
            v = self._variant_for_env(env_idx)
            pool = self._variant_layouts[v]
            assert pool, f"Variant {v} has no valid layouts to sample from."
            results.append(random.choice(pool))
        return results

    @property
    def remaining(self) -> int:
        """Number of layouts not yet consumed by :meth:`sample_without_replacement`.

        For heterogeneous pools, returns the minimum across all variants.
        """
        if self._heterogeneous:
            return min(len(self._variant_layouts[v]) - self._variant_next_idx[v] for v in range(self._num_variants))
        return len(self._layouts) - self._next_idx
