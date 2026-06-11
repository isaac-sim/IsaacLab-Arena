# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from isaaclab_arena.assets.registries import AssetRegistry


@dataclass
class IntentResolutionTraceEvent:
    """One step in the intent-resolution pipeline for debugging."""

    # Identifier for the resolution step, e.g. item.preferred_tags.substring.
    stage: str
    # The original query string that triggered this event.
    query: str
    # The registry key that was selected, or ``None`` when resolution failed.
    chosen: str | None
    # Human-readable annotation explaining why this choice was made.
    note: str = ""


ASSET_ERROR_STAGES: frozenset[str] = frozenset({
    # Required-tag pool is non-empty but no asset matched the query.
    "item.required_tags.miss",
    "background.required_tags.miss",
    "embodiment.required_tags.miss",
    # Required-tag pool is empty: the registry has no assets for this category.
    # The node is silently dropped in both cases, so both are resolution errors.
    "item.required_tags.empty_pool",
    "background.required_tags.empty_pool",
    "embodiment.required_tags.empty_pool",
})
"""Trace stage identifiers that indicate an asset-query failure."""


def match_asset(
    registry: AssetRegistry,
    query: str,
    trace_prefix: str,
    required_tags: list[str] | None = None,
    preferred_tags: list[str] | None = None,
) -> tuple[str | None, list[IntentResolutionTraceEvent]]:
    """Match a free-text ``query`` to a registered asset key.

    Resolution proceeds in three stages:

    1. **Exact match** in the pool of assets that carry ``required_tags``.
    2. **Fuzzy match** (substring, then difflib) in the pool narrowed by
       ``required_tags + preferred_tags`` when ``preferred_tags`` is non-empty.
    3. **Fuzzy match** in the ``required_tags`` pool only (relaxed fallback).

    Args:
        registry: Registry to look up asset names in.
        query: Asset name as emitted by the agent.
        trace_prefix: Prefix for trace event stages (e.g. ``"item"``,
            ``"background"``, ``"embodiment"``).

        required_tags: Tags every candidate must carry (e.g. ``["object"]``).
        preferred_tags: Additional tags that narrow the first-pass pool
            (e.g. item ``category_tags`` or ``["ik"]`` for embodiments).

    Returns:
        ``(chosen_key, events)`` — the resolved asset key (or ``None`` on
        miss) and all trace events produced during this call.
    """
    events: list[IntentResolutionTraceEvent] = []
    required_tags = required_tags or []
    candidates = sorted(registry.get_assets_with_all_tags(required_tags))
    # 1. Exact name match in a pool of assets with only the required tags.
    if query in candidates:
        events.append(IntentResolutionTraceEvent(f"{trace_prefix}.exact", query, query))
        return query, events

    # 2. Fuzzy matching in a pool narrowed by required + preferred tags.
    if preferred_tags:
        preferred_candidates = sorted(registry.get_assets_with_all_tags(required_tags + preferred_tags))
        chosen, sub_events = _fuzzy_match(
            preferred_candidates,
            query,
            trace_prefix=f"{trace_prefix}.preferred_tags",
            note=f"tags={required_tags + preferred_tags}, pool size={len(preferred_candidates)}",
        )
        events.extend(sub_events)
        if chosen is not None:
            return chosen, events

    # 3. Fuzzy matching in a pool of assets with only the required tags.
    chosen, sub_events = _fuzzy_match(
        candidates,
        query,
        trace_prefix=f"{trace_prefix}.required_tags",
        note=f"tags={required_tags}, pool size={len(candidates)}",
    )
    events.extend(sub_events)
    return chosen, events


def _fuzzy_match(
    pool: list[str],
    query: str,
    trace_prefix: str,
    note: str = "",
) -> tuple[str | None, list[IntentResolutionTraceEvent]]:
    """Match ``query`` within ``pool``: substring then difflib fuzzy.

    Returns a ``(chosen, events)`` pair.
    """
    if not pool:
        return None, [IntentResolutionTraceEvent(f"{trace_prefix}.empty_pool", query, None, note=note)]

    q = query.lower()
    substrs = [name for name in pool if q in name.lower()]
    if substrs:
        chosen = min(substrs, key=len)
        return chosen, [IntentResolutionTraceEvent(f"{trace_prefix}.substring", query, chosen, note=note)]

    matches = get_close_matches(query, pool, n=3, cutoff=0.5)
    if matches:
        # TODO(qianl): support object sets when there's multiple matching assets.
        chosen = matches[0]
        fuzzy_note = note
        if len(matches) > 1:
            ambiguity = f"multiple matches={matches}, taking first={chosen!r}"
            fuzzy_note = f"{note}; {ambiguity}" if note else ambiguity
        return chosen, [IntentResolutionTraceEvent(f"{trace_prefix}.fuzzy", query, chosen, note=fuzzy_note)]

    return None, [IntentResolutionTraceEvent(f"{trace_prefix}.miss", query, None, note=note)]
