# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from isaaclab_arena.assets.registries import AssetRegistry


@dataclass
class TraceEvent:
    """One step in the resolution pipeline — emitted to a structured log."""

    # Identifier for the resolution step, e.g. item.in_tags.substring or embodiment.miss.
    stage: str
    # The original query string that triggered this event.
    query: str
    # The registry key that was selected, or ``None`` when resolution failed.
    chosen: str | None
    # Up to _MAX_CANDIDATES near-miss names considered during fuzzy / tag matching.
    candidates: list[str] = field(default_factory=list)
    # Human-readable annotation explaining why this choice was made.
    note: str = ""


# Maximum number of near-miss candidate names stored in a TraceEvent.
_MAX_CANDIDATES = 10


class AssetMatcher:
    """Matches agent query strings to registered assets in :class:`AssetRegistry`."""

    _ERROR_TRACE_STAGES: set[str] = {
        "item.miss",  # no object asset matched the agent's query even after tag relaxation
        "background.wrong_tag",  # named asset exists but lacks the required background tag
        "background.miss",  # explicitly requested background not found in the registry
        "embodiment.miss",  # no embodiment asset matched even after preferred-tag relaxation
    }

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
        required_tags: list[str],
        preferred_tags: list[str] | None = None,
    ) -> str | None:
        """Match ``query`` to a registered asset key using tag-narrowed pools.

        Tries :meth:`_best_match` on ``required_tags + preferred_tags`` first.
        When that yields no match, relaxes to ``required_tags`` only.

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
        preferred_tags = preferred_tags or []

        if self.registry.is_registered(query):
            asset = self.registry.get_asset_by_name(query)
            tags = getattr(asset, "tags", None) or []
            if all(tag in tags for tag in required_tags):
                self.trace.append(TraceEvent(f"{trace_prefix}.exact", query, query))
                return query
            if trace_prefix == "background":
                self.trace.append(
                    TraceEvent(f"{trace_prefix}.wrong_tag", query, None, note=f"expected tags {required_tags!r}")
                )
                return None

        if preferred_tags:
            narrow_tags = required_tags + preferred_tags
            narrow_pool = self._pool_for(narrow_tags)
            if not narrow_pool:
                if trace_prefix == "item":
                    self.trace.append(
                        TraceEvent(
                            f"{trace_prefix}.tag_pool_empty",
                            query,
                            None,
                            note=f"no assets matched tags={preferred_tags}; relaxing to objects",
                        )
                    )
            else:
                preferred_stage_prefix = f"{trace_prefix}.in_tags" if trace_prefix == "item" else trace_prefix
                chosen = self._best_match(
                    query,
                    narrow_pool,
                    stage_prefix=preferred_stage_prefix,
                    note=f"tags={preferred_tags}",
                    ik_family=trace_prefix == "embodiment",
                )
                if chosen is not None:
                    return chosen
                if trace_prefix == "item":
                    self.trace.append(
                        TraceEvent(
                            f"{trace_prefix}.no_match_in_tags",
                            query,
                            None,
                            candidates=narrow_pool[:_MAX_CANDIDATES],
                            note=f"tags={preferred_tags}; relaxing to objects",
                        )
                    )

        relaxed_stage_prefix = f"{trace_prefix}.relaxed" if trace_prefix == "item" else trace_prefix
        relaxed_note = "closest object; category ignored" if trace_prefix == "item" else f"tags={required_tags}"
        relaxed_pool = self._pool_for(required_tags)
        chosen = self._best_match(query, relaxed_pool, stage_prefix=relaxed_stage_prefix, note=relaxed_note)
        if chosen is not None:
            return chosen

        self.trace.append(TraceEvent(f"{trace_prefix}.miss", query, None, candidates=relaxed_pool[:_MAX_CANDIDATES]))
        return None

    def _best_match(
        self,
        query: str,
        pool: list[str],
        *,
        stage_prefix: str,
        note: str = "",
        ik_family: bool = False,
    ) -> str | None:
        """Match ``query`` within ``pool``: exact, then substring, then fuzzy."""
        if not pool:
            return None

        if query in pool:
            self.trace.append(TraceEvent(f"{stage_prefix}.exact", query, query, note=note))
            return query

        q = query.lower()
        substrs = [name for name in pool if q in name.lower()]
        if substrs:
            chosen = min(substrs, key=len)
            if ik_family:
                base = stage_prefix.removesuffix(".in_tags")
                self.trace.append(
                    TraceEvent(
                        f"{base}.ik_family",
                        query,
                        chosen,
                        candidates=substrs[:_MAX_CANDIDATES],
                        note=f"bare family {query!r} → IK variant",
                    )
                )
            else:
                self.trace.append(
                    TraceEvent(
                        f"{stage_prefix}.substring",
                        query,
                        chosen,
                        candidates=substrs[:_MAX_CANDIDATES],
                        note=note,
                    )
                )
            return chosen

        matches = get_close_matches(query, pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent(f"{stage_prefix}.fuzzy", query, matches[0], candidates=matches, note=note))
            return matches[0]

        return None

    def _pool_for(self, tags: list[str]) -> list[str]:
        """Return registry keys whose assets carry every tag in ``tags``."""
        assets: set[str] | None = None
        for tag in tags:
            tagged = {a.name for a in self.registry.get_assets_by_tag(tag)}
            assets = tagged if assets is None else assets & tagged
        return sorted(assets or [])
