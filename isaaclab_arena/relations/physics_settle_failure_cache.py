# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from isaaclab_arena.relations.placement_result import PlacementResult


class PhysicsSettleFailureCache:
    """Tracks which solved layouts failed the post-reset physics settle check, persisted per scene/task.

    A layout is identified by a monotonic uid assigned in pool-enqueue order. A uid names the same
    layout across runs only when the whole solve sequence reproduces, which takes both a fixed
    ``placement_seed`` (same draw order) and the same object composition (same objects, geometry, and
    relations). The cache is bound to both: the seed and a ``layout_signature`` are stored in the file
    and re-checked on load, so swapping an object in/out under the same scene name invalidates the cache
    instead of skipping the wrong layouts. Persistence needs a cache key and a fixed seed -- without
    either, the cache still de-dupes failures in memory for the current run but writes nothing to disk.

    Args:
        cache_key: Scene/task identifier used as the on-disk file name. None disables persistence.
        placement_seed: The pool's base placement seed. None disables persistence; it also tags the
            cache file so a run under a different seed ignores stale (now-misaligned) failures.
        layout_signature: Opaque fingerprint of the object composition that fixes the solve order. Stored
            in the file and re-checked on load; a mismatch discards the cache, since the same uid would
            otherwise name a different layout once objects are swapped, resized, or re-related.
    """

    def __init__(
        self,
        cache_key: str | None,
        placement_seed: int | None,
        layout_signature: str | None = None,
    ) -> None:
        self._cache_key = cache_key
        self._placement_seed = placement_seed
        self._layout_signature = layout_signature
        self._uid_counter = 0
        # Keyed by id(layout): a layout object's uid for the lifetime of this run.
        self._uid_by_layout: dict[int, int] = {}
        self._failed_uids: set[int] = self._load()

    def register(self, layout: PlacementResult) -> bool:
        """Assign a layout its reproducible uid at enqueue time; return whether it's a known settle failure.

        Call once per layout, in enqueue order, so uids line up with the cached fail-set.
        """
        uid = self._uid_counter
        self._uid_counter += 1
        self._uid_by_layout[id(layout)] = uid
        return uid in self._failed_uids

    def record_failures(self, layouts: Iterable[PlacementResult]) -> None:
        """Record any of these layouts that aren't already known failures, persisting if anything is new.

        Layouts not previously registered (no uid) are ignored. No-op when nothing is new.
        """
        new_uids = {
            uid
            for layout in layouts
            if (uid := self._uid_by_layout.get(id(layout))) is not None and uid not in self._failed_uids
        }
        if not new_uids:
            return
        self._failed_uids |= new_uids
        self._save()

    # ------------------------------------------------------------------
    # Persistence (keyed by env, gated on placement_seed)
    # ------------------------------------------------------------------

    def _path(self) -> str | None:
        """Cache file for this env, or None when persistence is unavailable.

        Persistence needs both a cache key and a fixed placement_seed: uids only name the same layout
        across runs when the whole solve sequence reproduces.
        """
        if self._cache_key is None or self._placement_seed is None:
            return None
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "isaaclab_arena", "physics_settle")
        return os.path.join(cache_dir, f"{self._cache_key}.json")

    def _load(self) -> set[int]:
        """Load persisted settle failures for this Env. Ignores caches from a different seed."""
        path = self._path()
        if path is None or not os.path.exists(path):
            return set()
        with open(path) as cache_file:
            data = json.load(cache_file)
        # Discard the cache when either the placement seed or the layout signature
        # differs; stored uids would otherwise name different layouts than this run produces.
        if data.get("placement_seed") != self._placement_seed:
            return set()
        if data.get("layout_signature") != self._layout_signature:
            return set()
        return set(data.get("failed_uids", []))

    def _save(self) -> None:
        """Persist the accumulated settle failures so future runs skip them."""
        path = self._path()
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as cache_file:
            json.dump(
                {
                    "placement_seed": self._placement_seed,
                    "layout_signature": self._layout_signature,
                    "failed_uids": sorted(self._failed_uids),
                },
                cache_file,
            )
