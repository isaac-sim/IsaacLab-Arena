# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.agentic_environment_generation.asset_matcher import (
    ASSET_ERROR_STAGES,
    IntentResolutionTraceEvent,
    match_asset,
)
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
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

_INITIAL_STATE_SPEC_ID = "state_initial"


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
        self.trace: list[IntentResolutionTraceEvent] = []

    @property
    def resolution_errors(self) -> list[IntentResolutionTraceEvent]:
        """Trace events flagged as failures of the last :meth:`compile` call."""
        error_stages = ASSET_ERROR_STAGES | self._ERROR_TRACE_STAGES
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
        asset_name = match_asset(self.registry, query, "background", self.trace, ["background"])
        if asset_name is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=self._agent_node_id(query),
            name=asset_name,
            type=ArenaEnvGraphNodeType.BACKGROUND,
        )

    def _resolve_embodiment_node(self, query: str) -> ArenaEnvGraphNodeSpec | None:
        asset_name = match_asset(self.registry, query, "embodiment", self.trace, ["embodiment"], ["ik"])
        if asset_name is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=self._agent_node_id(query),
            name=asset_name,
            type=ArenaEnvGraphNodeType.EMBODIMENT,
        )

    def _resolve_item_node(self, item: Item) -> ArenaEnvGraphNodeSpec | None:
        asset_name = match_asset(self.registry, item.query, "item", self.trace, ["object"], item.category_tags)
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
            self.trace.append(
                IntentResolutionTraceEvent(f"{stage_prefix}.unknown_subject", rel.subject, None, note=rel.kind)
            )
            return None
        if rel.reference is not None and rel.reference not in known_ids:
            self.trace.append(
                IntentResolutionTraceEvent(f"{stage_prefix}.unknown_reference", rel.reference, None, note=rel.kind)
            )
            return None

        reference_part = f"_{rel.reference}" if rel.reference is not None else ""
        constraint_id = f"{_INITIAL_STATE_SPEC_ID}_{index}_{rel.kind}{reference_part}_{rel.subject}"
        self.trace.append(IntentResolutionTraceEvent(f"{stage_prefix}.ok", rel.subject, rel.reference, note=rel.kind))
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
                IntentResolutionTraceEvent(
                    "task.resolve",
                    task.kind,
                    task.kind,
                    note=f"params={task.params}",
                )
            )
            for param_name, param_value in task.params.items():
                if isinstance(param_value, str) and param_value not in known_ids:
                    self.trace.append(
                        IntentResolutionTraceEvent(
                            "task.unknown_param",
                            param_value,
                            None,
                            note=f"param={param_name}, task kind={task.kind}",
                        )
                    )
