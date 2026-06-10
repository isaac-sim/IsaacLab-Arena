# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_graph_spec import UnresolvedArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphSpatialRelationSpec,
    ArenaEnvGraphStateSpec,
    SpatialRelationSpec,
    TaskSpec,
)

from .environment_intent_spec import EnvironmentIntentSpec, Item

_INITIAL_STATE_SPEC_ID = "state_initial"


@dataclass
class TraceEvent:
    """One step in the resolution pipeline for debugging."""

    # Identifier for the resolution step, e.g. item.preferred_tags.substring.
    stage: str
    # The original query string that triggered this event.
    query: str
    # The registry key that was selected, or ``None`` when resolution failed.
    chosen: str | None
    # Human-readable annotation explaining why this choice was made.
    note: str = ""


# =============================================================================
# Asset query matching
#
# These module-level helpers resolve an agent's free-text query to a registered
# asset key via a two-stage strategy:
#   1. Exact key match.
#   2. Fuzzy matching: substring then difflib fuzzy, within the tag-narrowed pool.
# =============================================================================

_ASSET_ERROR_STAGES: frozenset[str] = frozenset({
    "item.required_tags.miss",
    "background.required_tags.miss",
    "embodiment.required_tags.miss",
})
"""Trace stage identifiers that indicate an asset-query failure."""


def match_asset(
    registry: AssetRegistry,
    query: str,
    trace_prefix: str,
    required_tags: list[str] | None = None,
    preferred_tags: list[str] | None = None,
) -> tuple[str | None, list[TraceEvent]]:
    """Match a free-text ``query`` to a registered asset key.

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
    events: list[TraceEvent] = []
    required_tags = required_tags or []
    candidates = sorted(registry.get_assets_with_all_tags(required_tags))

    if query in candidates:
        events.append(TraceEvent(f"{trace_prefix}.exact", query, query))
        return query, events

    preferred_tags = preferred_tags or []
    if preferred_tags:
        preferred_candidates = sorted(registry.get_assets_with_all_tags(required_tags + preferred_tags))
        chosen, sub_events = _best_match(
            query,
            preferred_candidates,
            trace_prefix=f"{trace_prefix}.preferred_tags",
            note=f"tags={required_tags + preferred_tags}, pool size={len(preferred_candidates)}",
        )
        events.extend(sub_events)
        if chosen is not None:
            return chosen, events

    chosen, sub_events = _best_match(
        query,
        candidates,
        trace_prefix=f"{trace_prefix}.required_tags",
        note=f"tags={required_tags}, pool size={len(candidates)}",
    )
    events.extend(sub_events)
    return chosen, events


def _best_match(
    query: str,
    pool: list[str],
    trace_prefix: str,
    note: str = "",
) -> tuple[str | None, list[TraceEvent]]:
    """Match ``query`` within ``pool``: substring then difflib fuzzy.

    Returns a ``(chosen, events)`` pair.
    """
    if not pool:
        return None, [TraceEvent(f"{trace_prefix}.empty_pool", query, None, note=note)]

    q = query.lower()
    substrs = [name for name in pool if q in name.lower()]
    if substrs:
        chosen = min(substrs, key=len)
        return chosen, [TraceEvent(f"{trace_prefix}.substring", query, chosen, note=note)]

    matches = get_close_matches(query, pool, n=3, cutoff=0.5)
    if matches:
        return matches[0], [TraceEvent(f"{trace_prefix}.fuzzy", query, matches[0], note=note)]

    return None, [TraceEvent(f"{trace_prefix}.miss", query, None, note=note)]


# =============================================================================
# IntentCompiler
# =============================================================================


class IntentCompiler:
    """Compiles an agent intent spec into a validated :class:`UnresolvedArenaEnvGraphSpec`."""

    _ERROR_TRACE_STAGES: frozenset[str] = frozenset({
        "relation.initial.unknown_subject",
        "relation.initial.unknown_reference",
        "task.unknown_param",
    })

    def __init__(self, registry: AssetRegistry | None = None) -> None:
        """Args:
        registry: Asset registry to use for catalog lookups.  Defaults to
            the global singleton :class:`AssetRegistry` when ``None``.
        """
        self.registry = registry or AssetRegistry()
        self.trace: list[TraceEvent] = []

    @property
    def resolution_errors(self) -> list[TraceEvent]:
        """Trace events flagged as failures of the last :meth:`compile` call."""
        error_stages = _ASSET_ERROR_STAGES | self._ERROR_TRACE_STAGES
        return [e for e in self.trace if e.stage in error_stages]

    @property
    def has_resolution_errors(self) -> bool:
        """``True`` if the last :meth:`compile` call produced any error-stage trace events."""
        return bool(self.resolution_errors)

    def compile(self, spec: EnvironmentIntentSpec, env_name: str | None = None) -> UnresolvedArenaEnvGraphSpec:
        """Compile an :class:`EnvironmentIntentSpec` into an :class:`UnresolvedArenaEnvGraphSpec`.

        Args:
            spec: Agent-produced intent spec describing the scene, initial relations,
                and task chain.
            env_name: Override for the graph's ``env_name`` field.  When ``None``
                the name is derived as ``llm_gen_{background}_{first_task_kind}``.

        Returns:
            An :class:`UnresolvedArenaEnvGraphSpec` ready for YAML round-tripping or
            further resolution via :meth:`~UnresolvedArenaEnvGraphSpec.resolve`.
        """
        self.trace = []

        nodes: list[ArenaEnvGraphNodeSpec] = []

        background_node = self._resolve_background_node(spec.background)
        if background_node is not None:
            nodes.append(background_node)

        embodiment_node = self._resolve_embodiment_node(spec.embodiment)
        if embodiment_node is not None:
            nodes.append(embodiment_node)

        # TODO(qianl): handle duplicate node IDs when two items share the same query and
        # neither has an instance_name.
        for item in spec.items:
            item_node = self._resolve_item_node(item)
            if item_node is not None:
                nodes.append(item_node)

        known_ids = {node.id for node in nodes}

        initial_state_spec = self._build_initial_state_spec(spec.initial_state_graph, known_ids)
        self._trace_tasks(spec.tasks, known_ids)

        return UnresolvedArenaEnvGraphSpec(
            env_name=env_name or self._derive_env_name(spec),
            nodes=nodes,
            tasks=spec.tasks,
            initial_state_spec=initial_state_spec,
        )

    @staticmethod
    def _derive_env_name(spec: EnvironmentIntentSpec) -> str:
        first_kind = spec.tasks[0].kind if spec.tasks else "task"
        return f"llm_gen_{spec.background}_{first_kind}"

    @staticmethod
    def _agent_node_id(query: str, *, instance_name: str | None = None) -> str:
        """Return the graph node id for an agent-emitted asset reference.

        The id stays as the agent's string so task params and spatial relations
        can reference it. ``instance_name`` overrides ``query`` for duplicate items.
        """
        return instance_name or query

    def _resolve_background_node(self, query: str) -> ArenaEnvGraphNodeSpec | None:
        asset_name, events = match_asset(self.registry, query, "background", ["background"])
        self.trace.extend(events)
        if asset_name is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=self._agent_node_id(query),
            name=asset_name,
            type=ArenaEnvGraphNodeType.BACKGROUND,
        )

    def _resolve_embodiment_node(self, query: str) -> ArenaEnvGraphNodeSpec | None:
        asset_name, events = match_asset(self.registry, query, "embodiment", ["embodiment"], ["ik"])
        self.trace.extend(events)
        if asset_name is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=self._agent_node_id(query),
            name=asset_name,
            type=ArenaEnvGraphNodeType.EMBODIMENT,
        )

    def _resolve_item_node(self, item: Item) -> ArenaEnvGraphNodeSpec | None:
        asset_name, events = match_asset(self.registry, item.query, "item", ["object"], item.category_tags)
        self.trace.extend(events)
        if asset_name is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=self._agent_node_id(item.query, instance_name=item.instance_name),
            name=asset_name,
            type=ArenaEnvGraphNodeType.OBJECT,
        )

    def _build_initial_state_spec(
        self, graph: list[SpatialRelationSpec], known_ids: set[str]
    ) -> ArenaEnvGraphStateSpec:
        constraints: list[ArenaEnvGraphSpatialRelationSpec] = []
        for index, rel in enumerate(graph):
            constraint = self._build_spatial_constraint(rel, index, known_ids)
            if constraint is not None:
                constraints.append(constraint)
        return ArenaEnvGraphStateSpec(
            id=_INITIAL_STATE_SPEC_ID,
            is_delta=False,
            spatial_constraints=constraints,
            task_constraints=[],
        )

    def _build_spatial_constraint(
        self, rel: SpatialRelationSpec, index: int, known_ids: set[str]
    ) -> ArenaEnvGraphSpatialRelationSpec | None:
        # rel.kind is guaranteed registered by SpatialRelationSpec._validate_kind_and_arity.
        stage_prefix = "relation.initial"
        if rel.subject not in known_ids:
            self.trace.append(TraceEvent(f"{stage_prefix}.unknown_subject", rel.subject, None, note=rel.kind))
            return None
        if rel.reference is not None and rel.reference not in known_ids:
            self.trace.append(TraceEvent(f"{stage_prefix}.unknown_reference", rel.reference, None, note=rel.kind))
            return None

        reference_part = f"_{rel.reference}" if rel.reference is not None else ""
        constraint_id = f"{_INITIAL_STATE_SPEC_ID}_{index}_{rel.kind}{reference_part}_{rel.subject}"
        self.trace.append(TraceEvent(f"{stage_prefix}.ok", rel.subject, rel.reference, note=rel.kind))
        return ArenaEnvGraphSpatialRelationSpec(
            id=constraint_id,
            kind=rel.kind,
            subject=rel.subject,
            reference=rel.reference,
            params=dict(rel.params),
        )

    def _trace_tasks(self, tasks: list[TaskSpec], known_ids: set[str]) -> None:
        """Emit trace events for each task: one ``task.resolve`` event and one
        ``task.unknown_param`` error event for every string param value that does
        not reference a resolved node ID."""
        for task in tasks:
            self.trace.append(
                TraceEvent(
                    "task.resolve",
                    task.kind,
                    task.kind,
                    note=f"params={task.params}",
                )
            )
            for param_name, param_value in task.params.items():
                if isinstance(param_value, str) and param_value not in known_ids:
                    self.trace.append(
                        TraceEvent(
                            "task.unknown_param",
                            param_value,
                            None,
                            note=f"param={param_name}, task kind={task.kind}",
                        )
                    )
