# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from isaaclab_arena.assets.registries import AssetRegistry


@dataclass
class TraceEvent:
    """One step in the resolution pipeline — emitted to a structured log."""

    # Identifier for the resolution step, e.g. item.preferred_tags.substring.
    stage: str
    # The original query string that triggered this event.
    query: str
    # The registry key that was selected, or ``None`` when resolution failed.
    chosen: str | None
    # Human-readable annotation explaining why this choice was made.
    note: str = ""


class AssetMatcher:
    """Matches agent query strings to registered assets in :class:`AssetRegistry`."""

    _ERROR_TRACE_STAGES: frozenset[str] = frozenset({
        "item.required_tags.miss",
        "background.required_tags.miss",
        "embodiment.required_tags.miss",
    })

    def __init__(self, registry: AssetRegistry, trace: list[TraceEvent]) -> None:
        """Args:
        registry: Asset registry to look up asset classes in.
        trace: Mutable list that receives one :class:`TraceEvent` per
            matching decision; shared with the parent :class:`IntentCompiler`.
        """
        self.registry = registry
        self.trace = trace

    def resolve_name(
        self,
        query: str,
        trace_prefix: str,
        required_tags: list[str] | None = None,
        preferred_tags: list[str] | None = None,
    ) -> str | None:
        """Match ``query`` to a registered asset key using tag-narrowed pools.

        Tries :meth:`_best_match` on assets matching ``required_tags + preferred_tags``
        first, then relaxes to ``required_tags`` only.

        Args:
            query: Asset name as emitted by the agent.
            trace_prefix: Prefix for trace event stages (e.g. ``"item"``,
                ``"background"``, ``"embodiment"``).
            required_tags: Tags every candidate must carry (e.g. ``["object"]``).
            preferred_tags: Additional tags that narrow the first-pass pool
                (e.g. item ``category_tags`` or ``["ik"]`` for embodiments).

        Returns:
            A registered asset key, or ``None`` if no match was found.
        """
        required_tags = required_tags or []
        candidates = sorted(self.registry.get_assets_with_all_tags(required_tags))
        # Exact match
        if query in candidates:
            self.trace.append(TraceEvent(f"{trace_prefix}.exact", query, query, note=""))
            return query

        # Approximate match by preferred tags
        preferred_tags = preferred_tags or []
        if preferred_tags:
            preferred_candidates = sorted(self.registry.get_assets_with_all_tags(required_tags + preferred_tags))
            chosen = self._best_match(
                query,
                preferred_candidates,
                trace_prefix=f"{trace_prefix}.preferred_tags",
                note=f"tags={required_tags + preferred_tags}, pool size={len(preferred_candidates)}",
            )
            if chosen is not None:
                return chosen

        # Approximate match by required tags
        chosen = self._best_match(
            query,
            candidates,
            trace_prefix=f"{trace_prefix}.required_tags",
            note=f"tags={required_tags}, pool size={len(candidates)}",
        )
        if chosen is not None:
            return chosen

        return None

    def _best_match(
        self,
        query: str,
        pool: list[str],
        trace_prefix: str,
        note: str = "",
    ) -> str | None:
        """Match ``query`` within ``pool``: exact, then substring, then fuzzy."""
        if not pool:
            self.trace.append(TraceEvent(f"{trace_prefix}.empty_pool", query, None, note=note))
            return None

        # Substring match
        q = query.lower()
        substrs = [name for name in pool if q in name.lower()]
        if substrs:
            chosen = min(substrs, key=len)
            self.trace.append(
                TraceEvent(
                    f"{trace_prefix}.substring",
                    query,
                    chosen,
                    note=note,
                )
            )
            return chosen

        # Fuzzy match by difflib similarity
        matches = get_close_matches(query, pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent(f"{trace_prefix}.fuzzy", query, matches[0], note=note))
            return matches[0]

        self.trace.append(TraceEvent(f"{trace_prefix}.miss", query, None, note=note))
        return None
