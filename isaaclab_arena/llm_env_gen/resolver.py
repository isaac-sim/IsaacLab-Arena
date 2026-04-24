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

# When the LLM emits a bare robot family name, pick the IK variant.
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


@dataclass
class ResolvedScene:
    background: type
    embodiment_name: str
    items: dict[str, type]
    initial_scene_graph: list[dict[str, Any]]
    final_scene_graph: list[dict[str, Any]]
    # Derived from the two graphs: relations that must become true to solve
    # the task (final − initial) and relations that must become false
    # (initial − final). The placement solver is expected to honor these as
    # negative constraints on the initial realization — see the TODO on
    # isaaclab_arena.relations.relation_solver.RelationSolver.
    goal_added: list[dict[str, Any]]
    goal_removed: list[dict[str, Any]]
    trace: list[TraceEvent]


class Resolver:
    """Resolves SceneSpec fields against AssetRegistry.

    Design notes:
      * Never raises on LLM mistakes — instead records a trace event with
        chosen=None so the caller can decide (retry LLM, ask user, fall back).
      * Exact name match wins. Otherwise substring containment, then difflib
        fuzzy, within a tag-filtered pool.
      * category_tags is a PREFERENCE, not a hard filter: if the tag pool is
        empty or yields no close match, we relax to the full object pool and
        record the relaxation in the trace.
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

        known = set(items) | {spec.background}
        initial_graph = self._resolve_graph(spec.initial_scene_graph, "initial", known, trace)
        final_graph = self._resolve_graph(spec.final_scene_graph, "final", known, trace)

        initial_keys = {(r["kind"], r["subject"], r["target"]) for r in initial_graph}
        final_keys = {(r["kind"], r["subject"], r["target"]) for r in final_graph}
        goal_added = [r for r in final_graph if (r["kind"], r["subject"], r["target"]) not in initial_keys]
        goal_removed = [r for r in initial_graph if (r["kind"], r["subject"], r["target"]) not in final_keys]

        for r in goal_added:
            trace.append(TraceEvent("diff.goal_added", r["subject"], r["target"], note=r["kind"]))
        for r in goal_removed:
            trace.append(TraceEvent("diff.goal_removed", r["subject"], r["target"], note=r["kind"]))

        return ResolvedScene(
            background=background_cls,
            embodiment_name=embodiment_name,
            items=items,
            initial_scene_graph=initial_graph,
            final_scene_graph=final_graph,
            goal_added=goal_added,
            goal_removed=goal_removed,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_item(self, item: Item, trace: list[TraceEvent]) -> type | None:
        if self.registry.is_registered(item.query):
            trace.append(TraceEvent("item.exact", item.query, item.query))
            return self.registry.get_asset_by_name(item.query)

        object_pool = self._pool_for(["object"])

        if item.category_tags:
            pool = self._pool_for(item.category_tags)
            if not pool:
                trace.append(
                    TraceEvent(
                        "item.tag_pool_empty",
                        item.query,
                        None,
                        note=f"no assets matched tags={item.category_tags}; relaxing to objects",
                    )
                )
            else:
                cls = self._best_match(
                    item.query, pool, trace, stage_prefix="item.in_tags", note=f"tags={item.category_tags}"
                )
                if cls is not None:
                    return cls
                trace.append(
                    TraceEvent(
                        "item.no_match_in_tags",
                        item.query,
                        None,
                        candidates=pool[:10],
                        note=f"tags={item.category_tags}; relaxing to objects",
                    )
                )

        cls = self._best_match(
            item.query, object_pool, trace, stage_prefix="item.relaxed", note="closest object; category ignored"
        )
        if cls is not None:
            return cls

        trace.append(TraceEvent("item.miss", item.query, None, candidates=object_pool[:10]))
        return None

    def _best_match(
        self, query: str, pool: list[str], trace: list[TraceEvent], stage_prefix: str, note: str
    ) -> type | None:
        """Prefer substring containment (e.g. 'bowl' → 'bowl_ycb_robolab'), then difflib fuzzy."""
        q = query.lower()
        substrs = [p for p in pool if q in p.lower()]
        if substrs:
            chosen = min(substrs, key=len)
            trace.append(TraceEvent(f"{stage_prefix}.substring", query, chosen, candidates=substrs[:5], note=note))
            return self.registry.get_asset_by_name(chosen)

        matches = get_close_matches(query, pool, n=3, cutoff=0.5)
        if matches:
            trace.append(TraceEvent(f"{stage_prefix}.fuzzy", query, matches[0], candidates=matches, note=note))
            return self.registry.get_asset_by_name(matches[0])
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
                trace.append(TraceEvent("name.wrong_tag", name, None, note=f"expected tag {required_tag!r}"))
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

        lower = name.lower()
        if lower in IK_DEFAULTS:
            chosen = IK_DEFAULTS[lower]
            trace.append(TraceEvent("embodiment.ik_default", name, chosen, note=f"bare family {name!r} → IK variant"))
            return chosen

        embodiment_pool = self._pool_for(["embodiment"])
        matches = get_close_matches(name, embodiment_pool, n=3, cutoff=0.5)
        if matches:
            trace.append(TraceEvent("embodiment.fuzzy", name, matches[0], candidates=matches))
            return matches[0]
        trace.append(TraceEvent("embodiment.miss", name, None, note="falling back to franka_ik"))
        return "franka_ik"

    def _resolve_graph(self, graph: list, phase: str, known: set[str], trace: list[TraceEvent]) -> list[dict[str, Any]]:
        """Validate one scene graph (initial or final) against the known item set."""
        resolved: list[dict[str, Any]] = []
        for rel in graph:
            stage_prefix = f"relation.{phase}"
            if rel.subject not in known:
                trace.append(TraceEvent(f"{stage_prefix}.unknown_subject", rel.subject, None, note=rel.kind))
                continue
            if rel.target is not None and rel.target not in known:
                trace.append(TraceEvent(f"{stage_prefix}.unknown_target", rel.target, None, note=rel.kind))
                continue
            if rel.kind == "in" and phase == "initial":
                trace.append(
                    TraceEvent(
                        f"{stage_prefix}.in_skipped",
                        rel.subject,
                        rel.target,
                        note="'in' has no initial-state semantics; move this to final_scene_graph.",
                    )
                )
                continue
            resolved.append({"kind": rel.kind, "subject": rel.subject, "target": rel.target, "params": rel.params})
            trace.append(TraceEvent(f"{stage_prefix}.ok", rel.subject, rel.target, note=rel.kind))
        return resolved
