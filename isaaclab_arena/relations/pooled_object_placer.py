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
from isaaclab_arena.utils.random import derive_seed, get_rngs

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


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

    With draw_by_episode, layouts are instead indexed by run-wide episode: episode j gets layout j, solved in the
    fixed group j // layouts_per_solve. The same episodes then get the same layouts on any number of envs. This
    requires the same objects in every env; the per-env queues are only built if a per-env method is called.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ObjectPlacer for the batched solve.
        pool_size: Number of layouts to solve per batch.
        num_envs: Number of simulation environments.
        collision_objects: Fixed background obstacles avoided during placement but never
            optimized or relation-constrained.
        draw_by_episode: Whether resets draw layouts by run-wide episode index rather than from per-env queues.
    """

    def __init__(
        self,
        objects: list[PlaceableAsset],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
        num_envs: int | None = None,
        collision_objects: list[CollisionObject] | None = None,
        draw_by_episode: bool = False,
    ) -> None:
        assert pool_size >= 1, f"pool_size must be >= 1, got {pool_size}"
        assert not (
            has_heterogeneous_objects(objects) and num_envs is None
        ), "num_envs is required for heterogeneous scenes."
        assert not (
            draw_by_episode and has_heterogeneous_objects(objects)
        ), "Drawing layouts by episode requires the same objects in every env."
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
        self._num_seeded_solves = 0
        # Per-env sampling RNG keyed by (placement_seed, env_id): env i's draws are reproducible
        # and independent of other envs.
        self._env_rngs = get_rngs(self._num_envs, placer_params.placement_seed)
        self._env_pools: list[EnvLayoutPool] = [EnvLayoutPool([]) for _ in range(self._num_envs)]
        self._env_pools_filled = False
        self._draw_by_episode = draw_by_episode
        self._episode_layouts: dict[int, PlacementResult] = {}
        self._solved_layout_groups: set[int] = set()

        if draw_by_episode:
            # Solve the first group now so impossible constraints fail at construction.
            self._solve_layout_group(0)
        else:
            self._ensure_env_pools()

    def _ensure_env_pools(self) -> None:
        """Fill the per-env queues on first use."""
        if self._env_pools_filled:
            return
        self._env_pools_filled = True
        self._solve_and_store(self._pool_size)
        for cur_env, pool in enumerate(self._env_pools):
            if not pool.layouts:
                raise RuntimeError(
                    f"Placement pool failed to produce any valid layouts for env {cur_env} "
                    f"from {self._pool_size} attempts. Check object relations and constraints."
                )

    def _solve_layout_group(self, group: int) -> None:
        """Solve the episode layouts of one fixed group, retrying invalid layouts with unused candidates.

        Retries are bounded by max_placement_attempts; the final retry may keep best-loss layouts as a fallback.
        """
        params = self._placer.params
        if self._base_placement_seed is not None:
            # Keep episode layouts apart from the per-env queues' (seed, env_id, candidate_id) keys.
            params.placement_seed = derive_seed(self._base_placement_seed, "episode_layouts")
        first_layout = group * params.layouts_per_solve
        pending = list(range(first_layout, first_layout + params.layouts_per_solve))
        max_solve_batches = max(1, params.max_placement_attempts)
        fallback_layouts = []
        for batch_idx in range(max_solve_batches):
            allow_fallback = self._allow_best_loss_fallbacks and batch_idx == max_solve_batches - 1
            results = self._placer.place_layouts(
                self._objects,
                pending,
                first_candidate_id=batch_idx * params.max_placement_attempts,
                collision_objects=self._collision_objects,
            )
            still_pending = []
            for layout_id, result in zip(pending, results, strict=True):
                if result.success or allow_fallback:
                    self._episode_layouts[layout_id] = result
                    if not result.success:
                        fallback_layouts.append(layout_id)
                else:
                    still_pending.append(layout_id)
            pending = still_pending
            if not pending:
                break
        if fallback_layouts:
            self._had_fallbacks = True
            print(f"Falling back to best-loss layouts for episode layouts: {fallback_layouts}")
        if pending:
            raise RuntimeError(
                f"Placement pool could not fill episode layouts {pending} after {max_solve_batches} solve batches."
            )
        self._solved_layout_groups.add(group)

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

    def _prepare_seeded_solve(self) -> None:
        """Give each seeded refill its own placement seed, so it neither replays this pool nor another seed's pool.

        The first solve keeps the base seed. Later solves derive the seed from (base seed, solve index) rather than
        offsetting it, which would reuse the first batch of a nearby base seed.
        """
        if self._base_placement_seed is None:
            return
        if self._num_seeded_solves == 0:
            self._placer.params.placement_seed = self._base_placement_seed
        else:
            self._placer.params.placement_seed = derive_seed(
                self._base_placement_seed, "pool_refill", self._num_seeded_solves
            )
        self._num_seeded_solves += 1

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
        self._prepare_seeded_solve()

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
        fallback_envs = []
        for cur_env in range(self._num_envs):
            env_results = ranked_results_per_env[cur_env][:layouts_per_env]
            valid_results = [r for r in env_results if r.success]
            missing = target_num_layouts_per_env - self._env_pools[cur_env].available
            if valid_results:
                if missing > 0:
                    self._env_pools[cur_env].extend(valid_results[:missing])
            elif allow_fallback and missing > 0:
                fallback = env_results[:missing]
                if fallback:
                    self._env_pools[cur_env].extend(fallback)
                    fallback_envs.append(cur_env)
                    self._had_fallbacks = True

        if fallback_envs:
            print(f"Falling back to best-loss layouts for envs: {fallback_envs}")

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
        self._ensure_env_pools()

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
        self._ensure_env_pools()

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

    def sample_for_episodes(self, episode_indices: list[int]) -> list[PlacementResult]:
        """Return layout j for each run-wide episode index j, solving the groups that hold them on first use.

        Drawing a layout does not consume it, so asking for the same episode again returns the same layout.
        """
        assert self._draw_by_episode, "This pool draws layouts per env; construct it with draw_by_episode=True."
        assert all(index >= 0 for index in episode_indices), f"Episode indices must be >= 0; got {episode_indices}"
        layouts_per_solve = self._placer.params.layouts_per_solve
        for group in sorted({index // layouts_per_solve for index in episode_indices}):
            if group not in self._solved_layout_groups:
                self._solve_layout_group(group)
        return [self._episode_layouts[index] for index in episode_indices]

    @property
    def draws_by_episode(self) -> bool:
        """Whether resets draw layouts by run-wide episode index rather than from per-env queues."""
        return self._draw_by_episode

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
        self._ensure_env_pools()
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
        self._ensure_env_pools()
        return min(self._available_per_env())

    @property
    def pool_size(self) -> int:
        """Number of layouts solved per batch when the pool is refilled."""
        return self._pool_size

    @property
    def total_remaining(self) -> int:
        """Total unread layouts across all env pools."""
        self._ensure_env_pools()
        return self._total_available()

    # ------------------------------------------------------------------
    # Pool introspection for the offline layout validator (sim-free)
    # ------------------------------------------------------------------

    @property
    def objects(self) -> list[PlaceableAsset]:
        """All objects (including anchors) participating in relation solving."""
        return self._objects

    def layouts_per_env(self) -> list[list[PlacementResult]]:
        """Flattened list of every stored layout, grouped by env pool index.

        A pool that draws by episode lists the episode layouts solved so far, episode j under env j % num_envs.
        """
        if self._draw_by_episode:
            layouts = [self._episode_layouts[index] for index in sorted(self._episode_layouts)]
            return [layouts[env_id :: self._num_envs] for env_id in range(self._num_envs)]
        self._ensure_env_pools()
        return [list(pool.layouts) for pool in self._env_pools]
