# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item
from isaaclab_arena.assets.registries import AssetRegistry


@dataclass
class TraceEvent:
    """One step in the resolution pipeline — emitted to a structured log."""

    # Identifier for the resolution step, e.g.item.in_tags.substring or embodiment.miss.
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


class AssetResolver:
    """Resolves catalog query strings against :class:`AssetRegistry`."""

    _ERROR_TRACE_STAGES: set[str] = {
        "item.miss",  # no object asset matched the agent's query even after tag relaxation
        "name.wrong_tag",  # named asset exists but lacks the required tag (e.g. embodiment constraint)
        "name.miss",  # explicitly requested asset name not found in the registry
        "embodiment.miss",  # no embodiment asset matched even after family expansion and fuzzy matching
    }

    def __init__(self, registry: AssetRegistry, trace: list[TraceEvent]) -> None:
        """Args:
        registry: Asset registry to look up asset classes in.
        trace: Mutable list that receives one :class:`TraceEvent` per
            resolution decision; shared with the parent :class:`IntentResolver`.
        """
        self.registry = registry
        self.trace = trace

    def resolve_item(self, item: Item) -> type | None:
        """Resolve a scene item query to a registered asset class.

        Args:
            item: The agent-proposed item to look up with ``query`` and ``category_tags``.

        Returns:
            The best-matching asset class, or ``None`` if no match was found.
        """
        # 1. Exact name match against the registry.
        if self.registry.is_registered(item.query):
            self.trace.append(TraceEvent("item.exact", item.query, item.query))
            return self.registry.get_asset_by_name(item.query)

        object_pool = self._pool_for(["object"])

        # 2. Substring / fuzzy match within the tag-narrowed pool (item.category_tags).
        if item.category_tags:
            pool = self._pool_for(item.category_tags)
            if not pool:
                self.trace.append(
                    TraceEvent(
                        "item.tag_pool_empty",
                        item.query,
                        None,
                        note=f"no assets matched tags={item.category_tags}; relaxing to objects",
                    )
                )
            else:
                cls = self._best_match(item.query, pool, stage_prefix="item.in_tags", note=f"tags={item.category_tags}")
                if cls is not None:
                    return cls
                self.trace.append(
                    TraceEvent(
                        "item.no_match_in_tags",
                        item.query,
                        None,
                        candidates=pool[:_MAX_CANDIDATES],
                        note=f"tags={item.category_tags}; relaxing to objects",
                    )
                )

        # 3. Substring / fuzzy match across all "object"-tagged assets (tag relaxation).
        cls = self._best_match(
            item.query, object_pool, stage_prefix="item.relaxed", note="closest object; category ignored"
        )
        if cls is not None:
            return cls

        self.trace.append(TraceEvent("item.miss", item.query, None, candidates=object_pool[:_MAX_CANDIDATES]))
        return None

    def resolve_name(self, name: str, required_tag: str | None) -> type | None:
        """Resolve an explicit asset name, optionally requiring a specific tag.

        Args:
            name: Exact asset name to look up first.
            required_tag: If provided, the resolved asset must carry this tag.

        Returns:
            The matching asset class, or ``None``.
        """
        if self.registry.is_registered(name):
            cls = self.registry.get_asset_by_name(name)
            if required_tag and required_tag not in getattr(cls, "tags", []):
                self.trace.append(TraceEvent("name.wrong_tag", name, None, note=f"expected tag {required_tag!r}"))
                return None
            self.trace.append(TraceEvent("name.exact", name, name))
            return cls

        pool = self._pool_for([required_tag]) if required_tag else self.registry.get_all_keys()
        matches = get_close_matches(name, pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent("name.fuzzy", name, matches[0], candidates=matches))
            return self.registry.get_asset_by_name(matches[0])

        self.trace.append(TraceEvent("name.miss", name, None, candidates=pool[:_MAX_CANDIDATES]))
        return None

    def resolve_embodiment(self, name: str) -> str | None:
        """Resolve a robot embodiment name to a registered asset key.

        Resolution strategy (in order):
        1. Exact registry match.
        2. Bare family name expansion: find the shortest ``["embodiment", "ik"]``-tagged
           asset whose name starts with the family prefix
           (e.g. ``"franka"`` → ``"franka_ik"``).
        3. Fuzzy match within the ``"embodiment"``-tagged asset pool.

        Args:
            name: Robot name as emitted by the agent (may be a bare family name
                like ``"franka"`` or a full registered name like ``"franka_joint_pos"``).

        Returns:
            A registered embodiment asset key, or ``None`` if no match was found.
        """
        if self.registry.is_registered(name):
            self.trace.append(TraceEvent("embodiment.exact", name, name))
            return name

        lower = name.lower()
        ik_pool = self._pool_for(["embodiment", "ik"])
        family_matches = [n for n in ik_pool if n.startswith(lower + "_")]
        if family_matches:
            chosen = min(family_matches, key=len)
            self.trace.append(
                TraceEvent(
                    "embodiment.ik_family",
                    name,
                    chosen,
                    candidates=family_matches,
                    note=f"bare family {name!r} → IK variant",
                )
            )
            return chosen

        embodiment_pool = self._pool_for(["embodiment"])
        matches = get_close_matches(name, embodiment_pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent("embodiment.fuzzy", name, matches[0], candidates=matches))
            return matches[0]

        self.trace.append(TraceEvent("embodiment.miss", name, None, candidates=embodiment_pool[:_MAX_CANDIDATES]))
        return None

    def _best_match(self, query: str, pool: list[str], stage_prefix: str, note: str) -> type | None:
        """Prefer substring containment (e.g. 'bowl' → 'bowl_ycb_robolab'), then difflib fuzzy."""
        q = query.lower()
        substrs = [p for p in pool if q in p.lower()]
        if substrs:
            chosen = min(substrs, key=len)
            self.trace.append(
                TraceEvent(f"{stage_prefix}.substring", query, chosen, candidates=substrs[:_MAX_CANDIDATES], note=note)
            )
            return self.registry.get_asset_by_name(chosen)

        matches = get_close_matches(query, pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent(f"{stage_prefix}.fuzzy", query, matches[0], candidates=matches, note=note))
            return self.registry.get_asset_by_name(matches[0])
        return None

    def _pool_for(self, tags: list[str]) -> list[str]:
        """Get a list of assets by tags."""
        assets = None
        for tag in tags:
            tagged = {a.name for a in self.registry.get_assets_by_tag(tag)}
            assets = tagged if assets is None else assets & tagged
        return sorted(assets or [])
