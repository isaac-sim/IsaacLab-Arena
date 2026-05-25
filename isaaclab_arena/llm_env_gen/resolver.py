# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic resolver that turns a SceneSpec into an ArenaEnvGraphSpec.

The LLM emits a SceneSpec. Resolver.resolve() walks that spec, binds each
query string to a registered Asset (preferring exact name, then fuzzy match
filtered by tags), and emits a fully-formed :class:`ArenaEnvGraphSpec`:

  * ``nodes`` — background, embodiment, and objects.
  * ``state_specs`` — one initial state spec derived from
    ``SceneSpec.initial_scene_graph``, plus one empty success state spec
    per task as a placeholder for downstream synthesis.
  * ``tasks`` — one task per LLM task, wired to its initial / success
    state spec ids.

Per-step "why-this-binding" decisions accumulate on ``self.trace`` so the
caller can inspect resolution after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphSpatialConstraintSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskSpec,
)
from isaaclab_arena.environments.graph_spec_utils import assert_references_exist, assert_unique_ids

from .schema import Item, Relation, SceneSpec, Task

# When the LLM emits a bare robot family name, pick the IK variant.
IK_DEFAULTS: dict[str, str] = {
    "franka": "franka_ik",
    "droid": "droid_differential_ik",
    "g1": "g1_wbc_pink",
    "gr1": "gr1_pink",
}

# SceneSpec relation kinds that have no ArenaEnvGraphSpatialConstraintType
# counterpart yet. Open/closed are task-state goals on articulated assets and
# only become meaningful inside the task class, not as scene-graph edges.
_UNSUPPORTED_RELATION_KINDS: frozenset[str] = frozenset({"open", "closed"})

# id used for the single initial state spec the resolver emits.
_INITIAL_STATE_SPEC_ID = "state_initial"


# id pattern used for the per-task success state spec placeholders. Each is
# emitted as an empty ArenaEnvGraphStateSpec so that ArenaEnvGraphSpec
# reference-existence assertions hold; downstream task-graph synthesis is
# responsible for populating them.
def _success_state_spec_id(task_index: int) -> str:
    return f"state_success_{task_index}"


# Mapping from SceneSpec relation kinds to the spatial-constraint types used
# inside an ArenaEnvGraphStateSpec. Keys must stay in sync with
# isaaclab_arena.llm_env_gen.schema.RelationKind.
_RELATION_KIND_TO_CONSTRAINT_TYPE: dict[str, ArenaEnvGraphSpatialConstraintType] = {
    "on": ArenaEnvGraphSpatialConstraintType.ON,
    "in": ArenaEnvGraphSpatialConstraintType.IN,
    "next_to": ArenaEnvGraphSpatialConstraintType.NEXT_TO,
    "at_position": ArenaEnvGraphSpatialConstraintType.AT_POSITION,
    "is_anchor": ArenaEnvGraphSpatialConstraintType.IS_ANCHOR,
}

# Relation kinds whose semantic anchor is the *subject* (no child). For these
# we set parent=subject and leave child=None. Everything else uses
# parent=target, child=subject (e.g. on/in/next_to).
_SUBJECT_AS_PARENT_KINDS: frozenset[str] = frozenset({"is_anchor", "at_position"})


@dataclass
class TraceEvent:
    """One step in the resolution pipeline — emitted to a structured log."""

    stage: str
    query: str
    chosen: str | None
    candidates: list[str] = field(default_factory=list)
    note: str = ""


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
      * The trace lives on the Resolver instance (``self.trace``) and is
        cleared at the start of every ``resolve()`` call.
    """

    def __init__(self, registry: AssetRegistry | None = None):
        self.registry = registry or AssetRegistry()
        # Populated incrementally by every resolution call. Caller reads after
        # ``resolve()`` returns.
        self.trace: list[TraceEvent] = []

    def resolve(self, spec: SceneSpec, env_name: str | None = None) -> ArenaEnvGraphSpec:
        """Resolve a SceneSpec into a full :class:`ArenaEnvGraphSpec`.

        ``env_name`` is derived from the first task and background if not
        provided. The success state of each task is NOT derived here —
        downstream code is responsible for filling in the per-task success
        state specs that this resolver emits as empty placeholders.
        """
        self.trace = []

        nodes: list[ArenaEnvGraphNodeSpec] = []

        background_node = self._resolve_background_node(spec.background)
        if background_node is not None:
            nodes.append(background_node)

        nodes.append(self._resolve_embodiment_node(spec.embodiment))

        for item in spec.items:
            item_node = self._resolve_item_node(item)
            if item_node is not None:
                nodes.append(item_node)

        known_ids = {node.id for node in nodes}

        initial_state_spec = self._build_initial_state_spec(spec.initial_scene_graph, known_ids)
        success_state_specs = [ArenaEnvGraphStateSpec(id=_success_state_spec_id(i)) for i in range(len(spec.tasks))]
        state_specs = [initial_state_spec, *success_state_specs]
        tasks = self._build_task_specs(spec.tasks, known_ids)

        env_graph_spec = ArenaEnvGraphSpec(
            env_name=env_name or self._derive_env_name(spec),
            nodes=nodes,
            tasks=tasks,
            state_specs=state_specs,
        )

        # Defensive: the resolver owns every id and reference it emits, so
        # these invariants should always hold. Catching a violation here
        # surfaces resolver bugs eagerly rather than at downstream consumers.
        assert_unique_ids(env_graph_spec.nodes, env_graph_spec.tasks, env_graph_spec.state_specs)
        assert_references_exist(env_graph_spec.nodes, env_graph_spec.tasks, env_graph_spec.state_specs)

        return env_graph_spec

    @staticmethod
    def _derive_env_name(spec: SceneSpec) -> str:
        first_kind = spec.tasks[0].kind if spec.tasks else "task"
        return f"llm_gen_{spec.background}_{first_kind}"

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _resolve_background_node(self, query: str) -> ArenaEnvGraphNodeSpec | None:
        cls = self._resolve_name(query, required_tag="background")
        if cls is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=query,
            name=cls.name,
            type=ArenaEnvGraphNodeType.BACKGROUND,
        )

    def _resolve_embodiment_node(self, query: str) -> ArenaEnvGraphNodeSpec:
        embodiment_name = self._resolve_embodiment(query)
        return ArenaEnvGraphNodeSpec(
            id=embodiment_name,
            name=embodiment_name,
            type=ArenaEnvGraphNodeType.EMBODIMENT,
        )

    def _resolve_item_node(self, item: Item) -> ArenaEnvGraphNodeSpec | None:
        cls = self._resolve_item(item)
        if cls is None:
            return None
        params: dict = {}
        if item.scale is not None:
            params["scale"] = item.scale
        return ArenaEnvGraphNodeSpec(
            id=item.instance_name or item.query,
            name=cls.name,
            type=ArenaEnvGraphNodeType.OBJECT,
            params=params,
        )

    # ------------------------------------------------------------------
    # State spec + task spec construction
    # ------------------------------------------------------------------

    def _build_initial_state_spec(self, graph: list[Relation], known_ids: set[str]) -> ArenaEnvGraphStateSpec:
        """Translate the LLM's initial scene graph into an ArenaEnvGraphStateSpec."""
        constraints: list[ArenaEnvGraphSpatialConstraintSpec] = []
        for index, rel in enumerate(graph):
            constraint = self._build_spatial_constraint(rel, index, known_ids)
            if constraint is not None:
                constraints.append(constraint)
        return ArenaEnvGraphStateSpec(
            id=_INITIAL_STATE_SPEC_ID,
            spatial_constraints=constraints,
            task_constraints=[],
        )

    def _build_spatial_constraint(
        self, rel: Relation, index: int, known_ids: set[str]
    ) -> ArenaEnvGraphSpatialConstraintSpec | None:
        stage_prefix = "relation.initial"
        if rel.kind in _UNSUPPORTED_RELATION_KINDS:
            self.trace.append(
                TraceEvent(
                    f"{stage_prefix}.unsupported_kind",
                    rel.subject,
                    None,
                    note=f"kind={rel.kind!r} has no spatial-constraint counterpart; skipping",
                )
            )
            return None
        if rel.kind == "in":
            self.trace.append(
                TraceEvent(
                    f"{stage_prefix}.in_skipped",
                    rel.subject,
                    rel.target,
                    note="'in' has no initial-state semantics; specify placement changes via tasks instead.",
                )
            )
            return None
        if rel.kind not in _RELATION_KIND_TO_CONSTRAINT_TYPE:
            self.trace.append(
                TraceEvent(
                    f"{stage_prefix}.unknown_kind",
                    rel.subject,
                    None,
                    note=f"kind={rel.kind!r} has no constraint mapping; skipping",
                )
            )
            return None
        if rel.subject not in known_ids:
            self.trace.append(TraceEvent(f"{stage_prefix}.unknown_subject", rel.subject, None, note=rel.kind))
            return None
        if rel.target is not None and rel.target not in known_ids:
            self.trace.append(TraceEvent(f"{stage_prefix}.unknown_target", rel.target, None, note=rel.kind))
            return None

        constraint_type = _RELATION_KIND_TO_CONSTRAINT_TYPE[rel.kind]
        if rel.kind in _SUBJECT_AS_PARENT_KINDS:
            parent, child = rel.subject, None
        else:
            if rel.target is None:
                self.trace.append(
                    TraceEvent(
                        f"{stage_prefix}.missing_target",
                        rel.subject,
                        None,
                        note=f"kind={rel.kind!r} requires a target; skipping",
                    )
                )
                return None
            parent, child = rel.target, rel.subject

        child_part = f"_{child}" if child is not None else ""
        constraint_id = f"{_INITIAL_STATE_SPEC_ID}_{index}_{rel.kind}_{parent}{child_part}"
        self.trace.append(TraceEvent(f"{stage_prefix}.ok", rel.subject, rel.target, note=rel.kind))
        return ArenaEnvGraphSpatialConstraintSpec(
            id=constraint_id,
            type=constraint_type,
            parent=parent,
            child=child,
            params=dict(rel.params),
        )

    def _build_task_specs(self, tasks: list[Task], known_ids: set[str]) -> list[ArenaEnvGraphTaskSpec]:
        out: list[ArenaEnvGraphTaskSpec] = []
        for index, task in enumerate(tasks):
            self.trace.append(
                TraceEvent(
                    "task.resolve",
                    task.kind,
                    task.kind,
                    note=f"subject={task.subject}, target={task.target}",
                )
            )
            if task.subject not in known_ids:
                self.trace.append(TraceEvent("task.unknown_subject", task.subject, None, note=f"task kind={task.kind}"))
            if task.target is not None and task.target not in known_ids:
                self.trace.append(TraceEvent("task.unknown_target", task.target, None, note=f"task kind={task.kind}"))
            out.append(
                ArenaEnvGraphTaskSpec(
                    id=f"task_{index}_{task.kind}",
                    type=task.kind,
                    initial_state_spec_id=_INITIAL_STATE_SPEC_ID,
                    # Points at an empty placeholder state spec emitted by
                    # resolve(); downstream task-graph synthesis fills it in.
                    success_state_spec_id=_success_state_spec_id(index),
                    task_args={
                        "subject": task.subject,
                        "target": task.target,
                        "description": task.description,
                    },
                )
            )
        return out

    # ------------------------------------------------------------------
    # Asset binding helpers (use self.trace directly)
    # ------------------------------------------------------------------

    def _resolve_item(self, item: Item) -> type | None:
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
        # Intersection across tags — an item tagged {"vegetable", "graspable"}
        # must satisfy both.
        assets = None
        for tag in tags:
            tagged = {a.name for a in self.registry.get_assets_by_tag(tag)}
            assets = tagged if assets is None else assets & tagged
        return sorted(assets or [])

    def _resolve_name(self, name: str, required_tag: str | None) -> type | None:
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

    def _resolve_embodiment(self, name: str) -> str:
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
