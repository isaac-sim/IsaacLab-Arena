# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for applying inferred background object references to intent specs."""

from __future__ import annotations

from isaaclab_arena.agentic_environment_generation.background_object_reference_spec import (
    BackgroundObjectReferenceInferenceSpec,
)
from isaaclab_arena.agentic_environment_generation.background_physics_catalog import (
    isaaclab_prim_path_for_background_reference,
)
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import (
    EnvironmentIntentSpec,
    ObjectReferenceItem,
)
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
)


def item_node_id(query: str, *, instance_name: str | None = None) -> str:
    """Return the graph node id for an intent item."""
    return instance_name or query


def apply_background_object_reference_inference(
    spec: EnvironmentIntentSpec,
    inference: BackgroundObjectReferenceInferenceSpec,
) -> EnvironmentIntentSpec:
    """Drop duplicate fixture items and rewrite task params to reference ids."""
    removed_ids = set(inference.remove_item_ids)
    items = [
        item for item in spec.items if item_node_id(item.query, instance_name=item.instance_name) not in removed_ids
    ]
    initial_state_graph = [
        rel
        for rel in spec.initial_state_graph
        if rel.subject not in removed_ids and (rel.reference is None or rel.reference not in removed_ids)
    ]
    tasks = list(spec.tasks)
    for binding in inference.task_param_bindings:
        assert binding.task_index < len(tasks), f"task_index {binding.task_index} is out of range"
        task = tasks[binding.task_index]
        tasks[binding.task_index] = task.model_copy(
            update={"params": {**task.params, binding.param_name: binding.reference_id}}
        )
    return spec.model_copy(
        update={
            "items": items,
            "object_references": list(inference.object_references),
            "initial_state_graph": initial_state_graph,
            "tasks": tasks,
        }
    )


def build_object_reference_nodes(
    references: list[ObjectReferenceItem],
    *,
    background_node_id: str,
    usd_path: str,
) -> list[ArenaEnvGraphNodeSpec]:
    """Materialize background object-reference graph nodes."""
    nodes: list[ArenaEnvGraphNodeSpec] = []
    for ref in references:
        object_type = ObjectType.RIGID if ref.object_type == "rigid" else ObjectType.ARTICULATION
        params: dict[str, object] = {}
        if ref.openable_joint_name:
            params["openable_joint_name"] = ref.openable_joint_name
        nodes.append(
            ArenaEnvGraphObjectReferenceNodeSpec(
                id=ref.id,
                name=ref.name,
                type=ArenaEnvGraphNodeType.OBJECT_REFERENCE,
                parent=background_node_id,
                prim_path=isaaclab_prim_path_for_background_reference(
                    background_node_id,
                    ref.usd_prim_path,
                    usd_path,
                ),
                object_type=object_type,
                params=params,
            )
        )
    return nodes
