# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic resolver that turns a SceneSpec into concrete asset classes.

The LLM emits a SceneSpec. Resolver.resolve() walks that spec, binds each
query string to a registered Asset (preferring exact name, then fuzzy match
filtered by tags), and records a step-by-step trace so the caller can see
*why* each binding was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

from isaaclab_arena.assets.registries import AssetRegistry

from .schema import Item, SceneSpec


@dataclass
class TraceEvent:
    """One step in the resolution pipeline — emitted to a structured log."""

    stage: str
    query: str
    chosen: str | None
    candidates: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ResolvedScene:
    background: type
    embodiment_name: str
    items: dict[str, type]
    relations: list[dict[str, Any]]
    trace: list[TraceEvent]


class Resolver:
    """Resolves SceneSpec fields against AssetRegistry.

    Design notes:
      * Never raises on LLM mistakes — instead records a trace event with
        chosen=None so the caller can decide (retry LLM, ask user, fall back).
      * Exact name match wins. Otherwise we search by tag intersection, then
        fuzzy-match within that pool. This keeps category constraints hard
        (a "vegetable" slot never gets resolved to a power drill) while still
        tolerating noisy LLM strings.
    """

    def __init__(self, registry: AssetRegistry | None = None):
        self.registry = registry or AssetRegistry()

    def resolve(self, spec: SceneSpec) -> ResolvedScene:
        trace: list[TraceEvent] = []

        background_cls = self._resolve_name(spec.background, required_tag="background", trace=trace)
        embodiment_name = self._resolve_embodiment(spec.embodiment, trace=trace)

        items: dict[str, type] = {}
        for item in spec.items:
            cls = self._resolve_item(item, trace=trace)
            if cls is not None:
                key = item.instance_name or item.query
                items[key] = cls

        relations = self._resolve_relations(spec, items, trace=trace)

        return ResolvedScene(
            background=background_cls,
            embodiment_name=embodiment_name,
            items=items,
            relations=relations,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_item(self, item: Item, trace: list[TraceEvent]) -> type | None:
        # 1. exact name hit
        if self.registry.is_registered(item.query):
            cls = self.registry.get_asset_by_name(item.query)
            trace.append(TraceEvent("item.exact", item.query, item.query))
            return cls

        # 2. tag-filtered pool
        pool = self._pool_for(item.category_tags) if item.category_tags else self.registry.get_all_keys()

        # 3. fuzzy match within the pool
        matches = get_close_matches(item.query, pool, n=3, cutoff=0.4)
        if matches:
            chosen = matches[0]
            trace.append(
                TraceEvent("item.fuzzy", item.query, chosen, candidates=matches, note=f"tags={item.category_tags}")
            )
            return self.registry.get_asset_by_name(chosen)

        trace.append(TraceEvent("item.miss", item.query, None, candidates=pool[:10]))
        return None

    def _pool_for(self, tags: list[str]) -> list[str]:
        # Intersection across tags — an item tagged {"vegetable", "graspable"}
        # must satisfy both.
        assets = None
        for tag in tags:
            tagged = {a.name for a in self.registry.get_assets_by_tag(tag)}
            assets = tagged if assets is None else assets & tagged
        return sorted(assets or [])

    def _resolve_name(self, name: str, required_tag: str | None, trace: list[TraceEvent]) -> type | None:
        if self.registry.is_registered(name):
            cls = self.registry.get_asset_by_name(name)
            if required_tag and required_tag not in getattr(cls, "tags", []):
                trace.append(
                    TraceEvent("name.wrong_tag", name, None, note=f"expected tag {required_tag!r}")
                )
                return None
            trace.append(TraceEvent("name.exact", name, name))
            return cls

        pool = self._pool_for([required_tag]) if required_tag else self.registry.get_all_keys()
        matches = get_close_matches(name, pool, n=3, cutoff=0.5)
        if matches:
            trace.append(TraceEvent("name.fuzzy", name, matches[0], candidates=matches))
            return self.registry.get_asset_by_name(matches[0])

        trace.append(TraceEvent("name.miss", name, None, candidates=pool[:10]))
        return None

    def _resolve_embodiment(self, name: str, trace: list[TraceEvent]) -> str:
        if self.registry.is_registered(name):
            trace.append(TraceEvent("embodiment.exact", name, name))
            return name
        # Fuzzy match against anything that starts with a known prefix —
        # franka, gr1, g1, droid, galbot, kuka_allegro, agibot.
        matches = get_close_matches(name, self.registry.get_all_keys(), n=3, cutoff=0.5)
        if matches:
            trace.append(TraceEvent("embodiment.fuzzy", name, matches[0], candidates=matches))
            return matches[0]
        trace.append(TraceEvent("embodiment.miss", name, None, note="falling back to franka_ik"))
        return "franka_ik"

    def _resolve_relations(
        self, spec: SceneSpec, items: dict[str, type], trace: list[TraceEvent]
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        known = set(items) | {spec.background}
        for rel in spec.relations:
            if rel.subject not in known:
                trace.append(TraceEvent("relation.unknown_subject", rel.subject, None, note=rel.kind))
                continue
            if rel.target is not None and rel.target not in known:
                trace.append(TraceEvent("relation.unknown_target", rel.target, None, note=rel.kind))
                continue
            resolved.append(
                {"kind": rel.kind, "subject": rel.subject, "target": rel.target, "params": rel.params}
            )
            trace.append(TraceEvent("relation.ok", rel.subject, rel.target, note=rel.kind))
        return resolved
