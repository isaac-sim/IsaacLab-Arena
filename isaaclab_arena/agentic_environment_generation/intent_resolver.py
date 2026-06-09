# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert :class:`EnvironmentIntentSpec` into :class:`ArenaEnvGraphSpec`."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphSpatialRelationSpec,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskSpec,
)
from isaaclab_arena.environments.arena_env_graph_types import SpatialRelationSpec, TaskSpec

from .asset_resolver import AssetResolver, TraceEvent
from .environment_intent_spec import EnvironmentIntentSpec, Item

_INITIAL_STATE_SPEC_ID = "state_initial"


def _success_state_spec_id(task_index: int) -> str:
    return f"state_success_{task_index}"


class IntentResolver:
    """Turns an agent intent spec into a validated environment graph spec.

    Uses :class:`AssetResolver` for catalog binding (background, embodiment, items).
    Relation and task wiring trace events share the same ``trace`` list.
    """

    _ERROR_TRACE_STAGES: frozenset[str] = frozenset({
        "relation.initial.unknown_subject",
        "relation.initial.unknown_parent",
        "task.unknown_param",
    })

    def __init__(self, registry: AssetRegistry | None = None) -> None:
        self.registry = registry or AssetRegistry()
        self.trace: list[TraceEvent] = []
        self._assets = AssetResolver(self.registry, self.trace)

    @property
    def asset_resolver(self) -> AssetResolver:
        return self._assets

    @property
    def resolution_errors(self) -> list[TraceEvent]:
        """Trace events flagged as failures of the last ``resolve()`` call."""
        error_stages = AssetResolver._ERROR_TRACE_STAGES | self._ERROR_TRACE_STAGES
        return [e for e in self.trace if e.stage in error_stages]

    @property
    def has_resolution_errors(self) -> bool:
        return bool(self.resolution_errors)

    def resolve(self, spec: EnvironmentIntentSpec, env_name: str | None = None) -> ArenaEnvGraphSpec:
        """Resolve an EnvironmentIntentSpec into a full :class:`ArenaEnvGraphSpec`."""
        self.trace = []
        self._assets = AssetResolver(self.registry, self.trace)

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

        initial_state_spec = self._build_initial_state_spec(spec.initial_state_graph, known_ids)
        success_state_specs = [ArenaEnvGraphStateSpec(id=_success_state_spec_id(i)) for i in range(len(spec.tasks))]
        state_specs = [initial_state_spec, *success_state_specs]
        tasks = self._build_task_specs(spec.tasks, known_ids)

        return ArenaEnvGraphSpec(
            env_name=env_name or self._derive_env_name(spec),
            nodes=nodes,
            tasks=tasks,
            state_specs=state_specs,
        )

    @staticmethod
    def _derive_env_name(spec: EnvironmentIntentSpec) -> str:
        first_kind = spec.tasks[0].kind if spec.tasks else "task"
        return f"llm_gen_{spec.background}_{first_kind}"

    def _resolve_background_node(self, query: str) -> ArenaEnvGraphNodeSpec | None:
        cls = self._assets.resolve_name(query, required_tag="background")
        if cls is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=query,
            name=cls.name,
            type=ArenaEnvGraphNodeType.BACKGROUND,
        )

    def _resolve_embodiment_node(self, query: str) -> ArenaEnvGraphNodeSpec:
        embodiment_name = self._assets.resolve_embodiment(query)
        return ArenaEnvGraphNodeSpec(
            id=embodiment_name,
            name=embodiment_name,
            type=ArenaEnvGraphNodeType.EMBODIMENT,
        )

    def _resolve_item_node(self, item: Item) -> ArenaEnvGraphNodeSpec | None:
        cls = self._assets.resolve_item(item)
        if cls is None:
            return None
        return ArenaEnvGraphNodeSpec(
            id=item.instance_name or item.query,
            name=cls.name,
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
            self.trace.append(TraceEvent(f"{stage_prefix}.unknown_parent", rel.reference, None, note=rel.kind))
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

    def _build_task_specs(self, tasks: list[TaskSpec], known_ids: set[str]) -> list[ArenaEnvGraphTaskSpec]:
        out: list[ArenaEnvGraphTaskSpec] = []
        for index, task in enumerate(tasks):
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
            out.append(
                ArenaEnvGraphTaskSpec(
                    id=f"task_{index}_{task.kind}",
                    kind=task.kind,
                    initial_state_spec_id=_INITIAL_STATE_SPEC_ID,
                    success_state_spec_id=_success_state_spec_id(index),
                    params=dict(task.params),
                    description=task.description,
                )
            )
        return out
