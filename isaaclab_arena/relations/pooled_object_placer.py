# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import torch
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
from isaaclab_arena.relations.layout_pool_serialization import (
    PoolDocument,
    deserialize_layout,
    read_pool_document,
    serialize_layout,
    write_pool_document,
)
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import (
    LayoutFilter,
    MultiEnvPlacementResult,
    PlacementResult,
    default_layout_filter,
)
from isaaclab_arena.utils.random import get_rngs

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass
class PooledLayout:
    """A stored layout plus its per-slot use_count.

    use_count lives here, not on PlacementResult, because the same result object is handed out
    by multiple slots and repeatedly by sample_with_replacement.
    """

    result: PlacementResult
    use_count: int = 0
    """Number of times this layout has been served (consuming or not)."""

    def mark_used(self) -> None:
        """Record that this layout was served to a caller."""
        self.use_count += 1


@dataclass
class EnvLayoutPool:
    """Unread layout queue for one absolute environment."""

    layouts: list[PooledLayout]
    cursor: int = 0

    @property
    def available(self) -> int:
        return len(self.layouts) - self.cursor

    def discard_consumed(self) -> None:
        self.layouts = self.layouts[self.cursor :]
        self.cursor = 0

    def append(self, result: PlacementResult) -> None:
        self.layouts.append(PooledLayout(result))

    def extend(self, results: list[PlacementResult]) -> None:
        self.layouts.extend(PooledLayout(result) for result in results)

    def next(self) -> PlacementResult:
        assert self.cursor < len(self.layouts), "No unread layouts remain in this env pool."
        layout = self.layouts[self.cursor]
        self.cursor += 1
        layout.mark_used()
        return layout.result


class PooledObjectPlacer:
    """Object placer that maintains solved placement layouts.

    Storage is organized as one queue per environment. Env-specific layouts
    are solved for fixed env geometry. sample_without_replacement consumes
    complete env rounds; sample_for_envs consumes only the requested absolute
    env ids. Reusable layouts are interchangeable and can be consumed one at a
    time from the pooled queues.

    Accepted layouts are preferred. On the final retry batch, best-loss
    solver results may be kept as a fallback.

    The pool is refilled automatically when an env's queue runs out.

    * sample_without_replacement(count) consumes count layouts. For
      env-specific layouts, count must be a multiple of num_envs and
      may cover multiple complete env rounds.
    * sample_for_envs(env_ids) consumes one layout for each requested
      absolute env id (used for partial resets).
    * sample_with_replacement(count) is non-consuming. Env-specific layouts
      are sampled from matching env slots; reusable layouts are drawn from the
      flattened pool, with the per-env RNG selecting the stream.

    save(path) / load(path, ...) persist solved layouts to JSON so a scene can reuse
    pre-existing object poses without re-solving.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ObjectPlacer for the batched solve.
        pool_size: Number of layouts to solve per batch.
        num_envs: Total number of simulation environments.  Required when
            layouts use env-specific object variants and defaults to 1 otherwise.
        layout_filter: Predicate over a layout's ValidationReport deciding which layouts to keep.
            Defaults to accepting layouts whose checks all pass. A custom predicate must tolerate
            missing keys (e.g. report.checks.get(name, False)), since checks can change.
    """

    def __init__(
        self,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
        num_envs: int | None = None,
        layout_filter: LayoutFilter | None = None,
        _skip_initial_solve: bool = False,
    ) -> None:
        assert pool_size >= 1, f"pool_size must be >= 1, got {pool_size}"
        self._layout_filter: LayoutFilter = layout_filter or default_layout_filter
        self._uses_env_specific_bboxes = has_heterogeneous_objects(objects)
        assert not (
            self._uses_env_specific_bboxes and num_envs is None
        ), "num_envs is required when layouts use env-specific object variants."
        self._num_envs = num_envs if num_envs is not None else 1
        assert self._num_envs >= 1, f"num_envs must be >= 1, got {self._num_envs}"

        self._objects = list(objects)
        # The placer ranks by the same filter the pool stores by, so its best layout is one we accept.
        # Poses are applied only when a sampled layout is used.
        self._placer = ObjectPlacer(
            params=replace(placer_params, apply_positions_to_objects=False), layout_filter=self._layout_filter
        )
        self._pool_size = pool_size
        self._had_fallbacks = False
        # Why the most recent solve rejected layouts (per check, plus "layout_filter"); shown in messages.
        self._last_rejection_summary: dict[str, int] = {}
        self._base_placement_seed = placer_params.placement_seed
        self._next_seed_offset = 0
        # Per-env sampling RNG keyed by (placement_seed, env_id): env i's draws are reproducible
        # and independent of other envs.
        self._env_rngs = get_rngs(self._num_envs, placer_params.placement_seed)
        self._env_pools: list[EnvLayoutPool] = [EnvLayoutPool([]) for _ in range(self._num_envs)]

        # load() fills the pools from disk instead of solving, so skip the upfront solve there.
        if _skip_initial_solve:
            return

        # _solve_and_store fills every env to >= 1 layout or raises with per-env diagnostics, so a
        # populated pool is an invariant here rather than a user-facing failure mode.
        self._solve_and_store(pool_size)
        assert all(pool.layouts for pool in self._env_pools), "Placement pool is empty after solving."

    def accepts(self, result: PlacementResult) -> bool:
        """Whether a layout passes the pool's layout_filter.

        Reporting consults this, not result.success, so fallbacks track the predicate the pool
        stores by even under a custom filter.
        """
        return self._layout_filter(result.validation)

    # ------------------------------------------------------------------
    # Pool storage internals
    # ------------------------------------------------------------------

    def _summarize_rejections(self, layouts: list[PlacementResult]) -> dict[str, int]:
        """Count rejection causes: each failed named check, or "layout_filter".

        Only filter-rejected layouts are counted. One that passes every built-in check but the
        filter still rejects is counted under "layout_filter" rather than vanishing.
        """
        counts: dict[str, int] = {}
        for layout in layouts:
            if self.accepts(layout):
                continue
            failed_checks = layout.validation.failed_checks
            for name in failed_checks or ("layout_filter",):
                counts[name] = counts.get(name, 0) + 1
        return counts

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
        """Solve layouts in batches until every env has target_per_env unread layouts.

        Each batch contributes one or more layout rounds per env. The outer
        loop is bounded by max_placement_attempts to avoid an
        unbounded refill in pathological configurations.
        """
        self._discard_consumed_layouts()
        target_per_env = max(1, (num_layouts + self._num_envs - 1) // self._num_envs)
        max_solve_batches = max(1, self._placer.params.max_placement_attempts)

        for batch_idx in range(max_solve_batches):
            # Reset each iteration so a fresh solve never carries a prior solve's stale counts and
            # terminal diagnostics report only the batch that ultimately failed.
            self._last_rejection_summary = {}
            max_missing = target_per_env - min(self._available_per_env())
            if max_missing <= 0:
                return

            batch_size = max_missing * self._num_envs
            allow_fallback = batch_idx == max_solve_batches - 1
            if self._uses_env_specific_bboxes:
                ranked_results_per_env, layouts_per_env = self._solve_env_ranked_layouts(batch_size)
                self._store_env_matched_results(
                    ranked_results_per_env,
                    layouts_per_env,
                    allow_fallback=allow_fallback,
                    target_per_env=target_per_env,
                )
            else:
                layouts = self._solve_reusable_layouts(batch_size, allow_fallback=allow_fallback)
                self._store_reusable_results(layouts)

            if min(self._available_per_env()) >= target_per_env:
                return

        raise RuntimeError(
            f"Placement pool could not fill {target_per_env} layouts per env after "
            f"{max_solve_batches} solve batches. Available per env: {self._available_per_env()}. "
            f"Most recent rejection reasons: {self._last_rejection_summary or 'none recorded'}."
        )

    def _solve_reusable_layouts(self, num_layouts: int, allow_fallback: bool = False) -> list[PlacementResult]:
        """Solve layouts that can be used by any env pool.

        Rejected candidates are discarded when at least one accepted layout exists.
        If no candidate is accepted on the final retry batch, fall back to best-loss
        results so environments with imperfect validation can still run.
        """
        self._prepare_seeded_solve(num_layouts * self._placer.params.max_placement_attempts)
        with torch.inference_mode(False):
            result = self._placer.place(self._objects, num_envs=num_layouts)

        # place() returns a single PlacementResult only when num_envs == 1.
        all_layouts = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        accepted_layouts = [layout for layout in all_layouts if self.accepts(layout)]

        if len(accepted_layouts) < num_layouts:
            self._last_rejection_summary = self._summarize_rejections(all_layouts)
            print(
                f"Pooled object placer: solved {num_layouts} layouts,"
                f" {len(accepted_layouts)} accepted, {num_layouts - len(accepted_layouts)} rejected"
                f" (rejection reasons: {self._last_rejection_summary})"
            )

        if accepted_layouts:
            return accepted_layouts

        if not allow_fallback:
            return []

        self._had_fallbacks = True
        print(
            "WARNING: No candidates met the pool's acceptance criteria. Accepting best-loss layouts as "
            f"fallback. Rejection reasons across {len(all_layouts)} candidates: {self._last_rejection_summary}"
        )
        return all_layouts

    def _store_reusable_results(self, layouts: list[PlacementResult]) -> None:
        """Distribute reusable layouts across env pools using greedy shortest-first.

        Layouts produced by _solve_reusable_layouts are interchangeable
        across envs, so we place each one into whichever pool currently has
        the fewest unread layouts. This keeps reusable capacity balanced
        across env pools.
        """
        if not layouts:
            return

        available = self._available_per_env()
        for layout in layouts:
            cur_env = min(range(self._num_envs), key=available.__getitem__)
            self._env_pools[cur_env].append(layout)
            available[cur_env] += 1

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
            )

        return ranked_results_per_env, layouts_per_env

    def _store_env_matched_results(
        self,
        ranked_results_per_env: list[list[PlacementResult]],
        layouts_per_env: int,
        target_per_env: int,
        allow_fallback: bool = False,
    ) -> None:
        """Store env-matched results into their corresponding pools.

        Each env is filled only up to target_per_env unread layouts, so envs
        that already met the target are not overfilled. Accepted layouts are
        preferred; if allow_fallback is set and an env has no accepted layouts,
        fall back to its best-loss results so environments with imperfect
        validation can still run.
        """
        total_accepted = 0
        fallback_envs = []
        for cur_env in range(self._num_envs):
            env_results = ranked_results_per_env[cur_env][:layouts_per_env]
            accepted_results = [r for r in env_results if self.accepts(r)]
            missing = target_per_env - self._env_pools[cur_env].available
            if accepted_results:
                if missing > 0:
                    enqueued = accepted_results[:missing]
                    total_accepted += len(enqueued)
                    self._env_pools[cur_env].extend(enqueued)
                else:
                    total_accepted += len(accepted_results)
            elif allow_fallback and missing > 0:
                self._env_pools[cur_env].extend(env_results[:missing])
                fallback_envs.append(cur_env)
                self._had_fallbacks = True

        total_solved = sum(min(len(env_results), layouts_per_env) for env_results in ranked_results_per_env)
        if total_accepted < total_solved:
            considered = [r for env_results in ranked_results_per_env for r in env_results[:layouts_per_env]]
            self._last_rejection_summary = self._summarize_rejections(considered)
            msg = (
                f"Placement pool (env-specific bbox layouts) solved {total_solved} candidates,"
                f" {total_accepted} accepted, {total_solved - total_accepted} rejected"
                f" (rejection reasons: {self._last_rejection_summary})"
            )
            if fallback_envs:
                msg += f". Falling back to best-loss layouts for envs: {fallback_envs}"
            print(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        """Return the next count layouts.

        Env-specific layouts are returned as complete rounds of
        [env_0, env_1, ..., env_{num_envs-1}] so each result still maps
        to the absolute environment it was solved for. Reusable layouts consume
        exactly count interchangeable entries.

        Args:
            count: Number of layouts to return.

        Raises:
            ValueError: If env-specific layouts are requested without a complete env round.
            RuntimeError: If the pool cannot provide count layouts after refilling.
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
                pool = self._env_pools[cur_env]
                if pool.available <= 0:
                    raise RuntimeError(
                        f"Placement pool: env {cur_env} has no more accepted layouts. "
                        "The solver is not producing enough accepted placements."
                    )
                results.append(pool.next())
        return results

    def sample_for_envs(self, env_ids: list[int]) -> dict[int, PlacementResult]:
        """Consume one layout for each requested absolute env id."""
        if not self._uses_env_specific_bboxes:
            layouts = self._sample_reusable_without_replacement(len(env_ids))
            return dict(zip(env_ids, layouts))

        if any(env_id < 0 or env_id >= self._num_envs for env_id in env_ids):
            raise ValueError(f"env_ids must be in [0, {self._num_envs}); got {env_ids}")

        if any(self._env_pools[env_id].available < 1 for env_id in env_ids):
            self._solve_and_store(max(self._pool_size, len(env_ids)))

        results: dict[int, PlacementResult] = {}
        for env_id in env_ids:
            pool = self._env_pools[env_id]
            if pool.available <= 0:
                raise RuntimeError(
                    f"Placement pool: env {env_id} has no more accepted layouts. "
                    "The solver is not producing enough accepted placements."
                )
            results[env_id] = pool.next()
        return results

    def _sample_reusable_without_replacement(self, count: int) -> list[PlacementResult]:
        """Consume exactly count interchangeable layouts."""
        if self._total_available() < count:
            self._solve_and_store(max(self._pool_size, count))

        available = self._available_per_env()
        if sum(available) < count:
            raise RuntimeError(
                f"Placement pool has {sum(available)} reusable layouts but {count} were requested. "
                "The solver is not producing enough accepted placements."
            )

        results: list[PlacementResult] = []
        for _ in range(count):
            cur_env = max(range(self._num_envs), key=available.__getitem__)
            pool = self._env_pools[cur_env]
            if pool.available <= 0:
                raise RuntimeError(
                    f"Placement pool: env {cur_env} has no more accepted layouts. "
                    "The solver is not producing enough accepted placements."
                )
            results.append(pool.next())
            available[cur_env] -= 1
        return results

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick count layouts at random with replacement (non-consuming).

        For env-specific layouts, slot i picks from env i % num_envs's pool
        so each result matches its absolute env. For reusable layouts, each slot
        draws from the flattened pool, with the per-env RNG only selecting the stream.
        """
        # Non-consuming (reads pool.layouts, ignoring the cursor). Slot i draws from env
        # (i % num_envs)'s RNG, so given identical pool contents each env's sequence replays
        # under (placement_seed, env_id), independent of other envs' draws.
        if self._uses_env_specific_bboxes:
            results: list[PlacementResult] = []
            for i in range(count):
                cur_env = i % self._num_envs
                pooled = self._env_pools[cur_env].layouts
                assert pooled, f"Env {cur_env} has no accepted layouts to sample from."
                results.append(self._draw(self._env_rngs[cur_env], pooled))
            return results
        # Reusable layouts are interchangeable: draw each slot from the flattened pool, with the
        # per-env RNG only selecting which stream the slot draws from.
        all_layouts = [layout for pool in self._env_pools for layout in pool.layouts]
        return [self._draw(self._env_rngs[i % self._num_envs], all_layouts) for i in range(count)]

    @staticmethod
    def _draw(rng: random.Random, pooled_layouts: list[PooledLayout]) -> PlacementResult:
        """Pick a stored layout, record the use, and return its result."""
        layout = rng.choice(pooled_layouts)
        layout.mark_used()
        return layout.result

    @property
    def requires_env_indexed_layouts(self) -> bool:
        """Whether sampled layouts must be matched back to absolute env ids."""
        return self._uses_env_specific_bboxes

    @property
    def num_envs(self) -> int:
        """Number of environment pools managed by this placer."""
        return self._num_envs

    @property
    def had_fallbacks(self) -> bool:
        """Whether any refill served best-loss fallback layouts."""
        return self._had_fallbacks

    @property
    def remaining(self) -> int:
        """Complete env rounds available to sample_without_replacement.

        One round consumes a layout from every env, so the per-env minimum is the limit.
        """
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
    # Persistence: save/load solved layouts to reuse poses without re-solving
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write all stored layouts to path as JSON for reuse without re-solving.

        Layouts are keyed by object name and include already-consumed ones (unread again on load).
        Cursor, use_count, and refill offset are not persisted; placement_seed and had_fallbacks
        are, so a loaded pool samples like a fresh pool with the same seed. See
        layout_pool_serialization for the on-disk schema and the atomic, fail-loud write.
        """
        assert len({obj.name for obj in self._objects}) == len(
            self._objects
        ), "Object names must be unique to save a layout pool keyed by name."
        document = PoolDocument(
            placement_seed=self._base_placement_seed,
            num_envs=self._num_envs,
            uses_env_specific_bboxes=self._uses_env_specific_bboxes,
            had_fallbacks=self._had_fallbacks,
            env_pools=[[serialize_layout(pooled.result) for pooled in pool.layouts] for pool in self._env_pools],
        )
        write_pool_document(Path(path), document)

    @classmethod
    def load(
        cls,
        path: str | Path,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
        *,
        num_envs: int | None = None,
        layout_filter: LayoutFilter | None = None,
    ) -> PooledObjectPlacer:
        """Rebuild a pool from a save() file, reusing stored poses instead of solving.

        objects must contain, by name, every object referenced in the file. Malformed files and
        env-count/heterogeneity/object-name mismatches fail loudly (see layout_pool_serialization
        for the structural checks). The saved placement_seed is restored (placer_params.placement_seed is
        ignored for seeding) so sampling matches the saved run; refill offset is not persisted, so
        a refill restarts from the first solve batch. pool_size becomes the loaded layout count,
        governing refill size.
        """
        path = Path(path)
        document = read_pool_document(path)
        assert (
            num_envs is None or num_envs == document.num_envs
        ), f"num_envs={num_envs} does not match the {document.num_envs} envs saved in {path}."

        name_to_obj = {obj.name: obj for obj in objects}
        assert len(name_to_obj) == len(objects), "Object names must be unique to load a layout pool by name."

        loaded_count = sum(len(env_layouts) for env_layouts in document.env_pools)
        placer = cls(
            objects=objects,
            placer_params=placer_params,
            pool_size=max(1, loaded_count),
            num_envs=document.num_envs,
            layout_filter=layout_filter,
            _skip_initial_solve=True,
        )
        assert placer._uses_env_specific_bboxes == document.uses_env_specific_bboxes, (
            "Loaded objects' heterogeneity does not match the saved pool; re-solve instead of loading this cache:"
            f" {path}"
        )

        # Reproduce the saved run rather than the freshly-passed seed.
        placer._base_placement_seed = document.placement_seed
        placer._had_fallbacks = document.had_fallbacks
        placer._env_rngs = get_rngs(document.num_envs, document.placement_seed)
        placer._env_pools = [
            EnvLayoutPool([PooledLayout(deserialize_layout(layout, name_to_obj)) for layout in env_layouts])
            for env_layouts in document.env_pools
        ]
        for cur_env, pool in enumerate(placer._env_pools):
            assert pool.layouts, f"Loaded layout pool has no layouts for env {cur_env}: {path}"
        return placer
