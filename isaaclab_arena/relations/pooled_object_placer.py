# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.utils.random import get_rngs

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject


@dataclass
class EnvLayoutPool:
    """Unread layout queue for one absolute environment."""

    layouts: list[PlacementResult]
    cursor: int = 0

    @property
    def available(self) -> int:
        return len(self.layouts) - self.cursor

    def discard_consumed(self) -> None:
        self.layouts = self.layouts[self.cursor :]
        self.cursor = 0

    def append(self, layout: PlacementResult) -> None:
        self.layouts.append(layout)

    def extend(self, layouts: list[PlacementResult]) -> None:
        self.layouts.extend(layouts)

    def next(self) -> PlacementResult:
        assert self.cursor < len(self.layouts), "No unread layouts remain in this env pool."
        layout = self.layouts[self.cursor]
        self.cursor += 1
        return layout


class PooledObjectPlacer:
    """Object placer that maintains solved placement layouts.

    Storage is organized as one queue per environment: every layout is solved
    against its env's geometry and bound to that absolute env id.

    Strictly valid layouts are preferred. On the final retry batch, best-loss
    solver results may be kept as a fallback.

    The pool is refilled automatically when an env's queue runs out.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ObjectPlacer for the batched solve.
        pool_size: Number of layouts to solve per batch.
        num_envs: Number of simulation environments.
        collision_objects: Fixed background obstacles avoided during placement but never
            optimized or relation-constrained.
    """

    def __init__(
        self,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
        num_envs: int | None = None,
        collision_objects: list[CollisionObject] | None = None,
    ) -> None:
        assert pool_size >= 1, f"pool_size must be >= 1, got {pool_size}"
        assert not (
            has_heterogeneous_objects(objects) and num_envs is None
        ), "num_envs is required for heterogeneous scenes."
        self._num_envs = num_envs if num_envs is not None else 1
        assert self._num_envs >= 1, f"num_envs must be >= 1, got {self._num_envs}"

        self._objects = list(objects)
        self._collision_objects = list(collision_objects) if collision_objects else []
        # Pool construction ranks several candidate layouts per env and applies
        # poses only when a sampled layout is used.
        self._placer = ObjectPlacer(params=replace(placer_params, apply_positions_to_objects=False))
        self._pool_size = pool_size
        self._had_fallbacks = False
        self._allow_best_loss_fallbacks = placer_params.allow_best_loss_fallbacks
        self._base_placement_seed = placer_params.placement_seed
        self._next_seed_offset = 0
        # Per-env sampling RNG keyed by (placement_seed, env_id): env i's draws are reproducible
        # and independent of other envs.
        self._env_rngs = get_rngs(self._num_envs, placer_params.placement_seed)
        self._env_pools: list[EnvLayoutPool] = [EnvLayoutPool([]) for _ in range(self._num_envs)]

        self._solve_and_store(pool_size)
        for cur_env, pool in enumerate(self._env_pools):
            if not pool.layouts:
                raise RuntimeError(
                    f"Placement pool failed to produce any valid layouts for env {cur_env} "
                    f"from {pool_size} attempts. Check object relations and constraints."
                )

    # ------------------------------------------------------------------
    # Pool storage internals
    # ------------------------------------------------------------------

    def _available_per_env(self) -> list[int]:
        """Number of unread layouts in each env's pool (length num_envs)."""
        return [pool.available for pool in self._env_pools]

    def _total_available(self) -> int:
        """Total unread layouts across all env pools."""
        return sum(self._available_per_env())

    def _discard_consumed_layouts(self) -> None:
        """Drop consumed layouts from every env pool before appending new layouts."""
        for pool in self._env_pools:
            pool.discard_consumed()

    def _prepare_seeded_solve(self, num_candidates: int) -> None:
        """Avoid replaying the same candidate sequence on seeded refills."""
        if self._base_placement_seed is None:
            return
        self._placer.params.placement_seed = self._base_placement_seed + self._next_seed_offset
        self._next_seed_offset += num_candidates

    def _solve_and_store(self, num_layouts: int) -> None:
        """Solve and store layouts until every env has target_num_layouts_per_env unread layouts.

        Bounded by max_placement_attempts; raises if the target cannot be met.
        """
        self._discard_consumed_layouts()
        target_num_layouts_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        max_solve_batches = max(1, self._placer.params.max_placement_attempts)

        for batch_idx in range(max_solve_batches):
            max_missing = target_num_layouts_per_env - min(self._available_per_env())
            if max_missing <= 0:
                return

            batch_size = max_missing * self._num_envs
            allow_fallback = self._allow_best_loss_fallbacks and batch_idx == max_solve_batches - 1
            ranked_results_per_env, layouts_per_env = self._solve_env_ranked_layouts(batch_size)
            self._store_env_matched_results(
                ranked_results_per_env,
                layouts_per_env,
                allow_fallback=allow_fallback,
                target_num_layouts_per_env=target_num_layouts_per_env,
            )

            if min(self._available_per_env()) >= target_num_layouts_per_env:
                return

        raise RuntimeError(
            f"Placement pool could not fill {target_num_layouts_per_env} layouts per env after "
            f"{max_solve_batches} solve batches. Available per env: {self._available_per_env()}."
        )

    def _solve_env_ranked_layouts(self, num_layouts: int) -> tuple[list[list[PlacementResult]], int]:
        """Solve ranked layouts tied to each env's actual object geometry.

        Returns ranked candidate lists per real env so the pool can store
        multiple layouts for each env without treating candidate rows as
        environments.
        """
        layouts_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        # ObjectPlacer seeds each candidate as placement_seed + candidate_idx.
        # Keep this count aligned with place_ranked_per_env's candidate layout.
        num_candidates = self._num_envs * layouts_per_env * self._placer.params.max_placement_attempts
        self._prepare_seeded_solve(num_candidates)

        with torch.inference_mode(False):
            ranked_results_per_env = self._placer.place_ranked_per_env(
                self._objects,
                num_envs=self._num_envs,
                results_per_env=layouts_per_env,
                collision_objects=self._collision_objects,
            )

        return ranked_results_per_env, layouts_per_env

    def _store_env_matched_results(
        self,
        ranked_results_per_env: list[list[PlacementResult]],
        layouts_per_env: int,
        target_num_layouts_per_env: int,
        allow_fallback: bool = False,
    ) -> None:
        """Store each env's results into its pool, up to target_num_layouts_per_env unread layouts.

        Valid layouts are preferred; when allow_fallback is set, an env with no
        valid layout keeps its best-loss results instead of staying empty.
        An env that has at least one valid layout never falls back to best-loss,
        even if it has fewer valid layouts than target_num_layouts_per_env.
        """
        total_valid = 0
        fallback_envs = []
        for cur_env in range(self._num_envs):
            env_results = ranked_results_per_env[cur_env][:layouts_per_env]
            valid_results = [r for r in env_results if r.success]
            missing = target_num_layouts_per_env - self._env_pools[cur_env].available
            if valid_results:
                if missing > 0:
                    enqueued = valid_results[:missing]
                    total_valid += len(enqueued)
                    self._env_pools[cur_env].extend(enqueued)
                else:
                    total_valid += len(valid_results)
            elif allow_fallback and missing > 0:
                fallback = env_results[:missing]
                if fallback:
                    self._env_pools[cur_env].extend(fallback)
                    fallback_envs.append(cur_env)
                    self._had_fallbacks = True

        total_solved = sum(min(len(env_results), layouts_per_env) for env_results in ranked_results_per_env)
        if total_valid < total_solved or fallback_envs:
            msg = (
                f"Placement pool solved {total_solved} candidates,"
                f" {total_valid} valid, {total_solved - total_valid} failed validation"
            )
            if fallback_envs:
                msg += f". Falling back to best-loss layouts for envs: {fallback_envs}"
            print(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        """Return the next count layouts as complete env rounds.

        Layouts are returned as complete rounds of
        [env_0, env_1, ..., env_{num_envs-1}] so each result maps to the
        absolute environment it was solved for.

        Args:
            count: Number of layouts to return. Must be a multiple of num_envs.
        """
        if count % self._num_envs != 0:
            raise ValueError(f"count must be a multiple of num_envs ({self._num_envs}), got {count}")

        layouts_per_env = count // self._num_envs
        if min(self._available_per_env()) < layouts_per_env:
            self._solve_and_store(max(self._pool_size, count))

        results: list[PlacementResult] = []
        for _ in range(layouts_per_env):
            for cur_env in range(self._num_envs):
                pool = self._env_pools[cur_env]
                if pool.available <= 0:
                    raise RuntimeError(
                        f"Placement pool: env {cur_env} has no more valid layouts. "
                        "The solver is not producing enough valid placements."
                    )
                results.append(pool.next())
        return results

    def sample_for_envs(self, env_ids: list[int]) -> dict[int, PlacementResult]:
        """Consume one layout for each requested absolute env id."""
        if any(env_id < 0 or env_id >= self._num_envs for env_id in env_ids):
            raise ValueError(f"env_ids must be in [0, {self._num_envs}); got {env_ids}")

        if any(self._env_pools[env_id].available < 1 for env_id in env_ids):
            self._solve_and_store(max(self._pool_size, len(env_ids)))

        results: dict[int, PlacementResult] = {}
        for env_id in env_ids:
            pool = self._env_pools[env_id]
            if pool.available <= 0:
                raise RuntimeError(
                    f"Placement pool: env {env_id} has no more valid layouts. "
                    "The solver is not producing enough valid placements."
                )
            results[env_id] = pool.next()
        return results

    @property
    def num_envs(self) -> int:
        """Number of environment pools managed by this placer."""
        return self._num_envs

    @property
    def had_fallbacks(self) -> bool:
        """Whether any pool refill accepted best-loss layouts that failed strict validation."""
        return self._had_fallbacks

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick count layouts at random with replacement (non-consuming).

        Slot i picks from env i % num_envs's pool so each result matches its
        absolute env, using that env's RNG for a reproducible draw.
        """
        # Reads pool.layouts directly, ignoring the consumption cursor.
        results: list[PlacementResult] = []
        for i in range(count):
            cur_env = i % self._num_envs
            pool = self._env_pools[cur_env].layouts
            assert pool, f"Env {cur_env} has no valid layouts to sample from."
            results.append(self._env_rngs[cur_env].choice(pool))
        return results

    @property
    def remaining(self) -> int:
        """Number of complete env rounds available, i.e. the minimum unread count across env pools."""
        return min(self._available_per_env())

    @property
    def pool_size(self) -> int:
        """Number of layouts solved per batch when the pool is refilled."""
        return self._pool_size

    @property
    def total_remaining(self) -> int:
        """Total unread layouts across all env pools."""
        return self._total_available()

    # ------------------------------------------------------------------
    # Pool introspection for the offline layout validator (sim-free)
    # ------------------------------------------------------------------

    @property
    def objects(self) -> list[ObjectBase]:
        """All objects (including anchors) participating in relation solving."""
        return self._objects

    def layouts_per_env(self) -> list[list[PlacementResult]]:
        """Flattened list of every stored layout, grouped by env pool index."""
        return [list(pool.layouts) for pool in self._env_pools]
