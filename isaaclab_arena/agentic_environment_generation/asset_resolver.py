# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bind intent query strings to registered assets via :class:`AssetRegistry`."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import Item
from isaaclab_arena.assets.registries import AssetRegistry

# When the agent emits a bare robot family name, pick the IK variant.
IK_DEFAULTS: dict[str, str] = {
    "franka": "franka_ik",
    "droid": "droid_differential_ik",
    "g1": "g1_wbc_pink",
    "gr1": "gr1_pink",
}


@dataclass
class TraceEvent:
    """One step in the resolution pipeline — emitted to a structured log."""

    stage: str
    query: str
    chosen: str | None
    candidates: list[str] = field(default_factory=list)
    note: str = ""


class AssetResolver:
    """Resolves catalog query strings against :class:`AssetRegistry`.

    Appends trace events to the shared ``trace`` list passed at construction.
    Never raises on bad queries — callers inspect ``trace`` instead.
    """

    _ERROR_TRACE_STAGES: frozenset[str] = frozenset({
        "item.miss",
        "name.wrong_tag",
        "name.miss",
    })

    def __init__(self, registry: AssetRegistry, trace: list[TraceEvent]) -> None:
        self.registry = registry
        self.trace = trace

    def resolve_item(self, item: Item) -> type | None:
        if self.registry.is_registered(item.query):
            self.trace.append(TraceEvent("item.exact", item.query, item.query))
            return self.registry.get_asset_by_name(item.query)

        object_pool = self._pool_for(["object"])

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
                        candidates=pool[:10],
                        note=f"tags={item.category_tags}; relaxing to objects",
                    )
                )

        cls = self._best_match(
            item.query, object_pool, stage_prefix="item.relaxed", note="closest object; category ignored"
        )
        if cls is not None:
            return cls

        self.trace.append(TraceEvent("item.miss", item.query, None, candidates=object_pool[:10]))
        return None

    def resolve_name(self, name: str, required_tag: str | None) -> type | None:
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

        self.trace.append(TraceEvent("name.miss", name, None, candidates=pool[:10]))
        return None

    def resolve_embodiment(self, name: str) -> str:
        if self.registry.is_registered(name):
            self.trace.append(TraceEvent("embodiment.exact", name, name))
            return name

        lower = name.lower()
        if lower in IK_DEFAULTS:
            chosen = IK_DEFAULTS[lower]
            self.trace.append(
                TraceEvent("embodiment.ik_default", name, chosen, note=f"bare family {name!r} → IK variant")
            )
            return chosen

        embodiment_pool = self._pool_for(["embodiment"])
        matches = get_close_matches(name, embodiment_pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent("embodiment.fuzzy", name, matches[0], candidates=matches))
            return matches[0]
        self.trace.append(TraceEvent("embodiment.miss", name, None, note="falling back to franka_ik"))
        return "franka_ik"

    def _best_match(self, query: str, pool: list[str], stage_prefix: str, note: str) -> type | None:
        """Prefer substring containment (e.g. 'bowl' → 'bowl_ycb_robolab'), then difflib fuzzy."""
        q = query.lower()
        substrs = [p for p in pool if q in p.lower()]
        if substrs:
            chosen = min(substrs, key=len)
            self.trace.append(TraceEvent(f"{stage_prefix}.substring", query, chosen, candidates=substrs[:5], note=note))
            return self.registry.get_asset_by_name(chosen)

        matches = get_close_matches(query, pool, n=3, cutoff=0.5)
        if matches:
            self.trace.append(TraceEvent(f"{stage_prefix}.fuzzy", query, matches[0], candidates=matches, note=note))
            return self.registry.get_asset_by_name(matches[0])
        return None

    def _pool_for(self, tags: list[str]) -> list[str]:
        assets = None
        for tag in tags:
            tagged = {a.name for a in self.registry.get_assets_by_tag(tag)}
            assets = tagged if assets is None else assets & tagged
        return sorted(assets or [])
