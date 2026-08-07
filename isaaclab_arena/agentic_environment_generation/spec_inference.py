# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM inference for environment graph specs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    InferenceBackend,
    StructuredOutputRequest,
    build_strict_schema,
)
from isaaclab_arena.agentic_environment_generation.spec_validation import (
    collect_agent_ready_task_validation_traces,
    format_validation_error,
)
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import TaskCompositionType


class SpecInference:
    """Infers ArenaEnvGraphSpec from a natural-language prompt."""

    def __init__(self, inference_backend: InferenceBackend):
        self._inference_backend = inference_backend
        self._schema = build_strict_schema(ArenaEnvGraphSpec)

    def infer(
        self,
        prompt: str,
        traces: list[str],
        asset_catalog: Any,
        relation_catalog: Any,
        task_catalog: Any,
    ) -> tuple[ArenaEnvGraphSpec | None, dict[str, Any]]:
        """Generate an ArenaEnvGraphSpec from a natural-language prompt.

        Args:
            prompt: End-user environment description.
            traces: Accumulator for validation error lines, extended in place on failure.
            asset_catalog: Embodiment, background, and object vocabulary for the user message.
            relation_catalog: Relation vocabulary for the user message.
            task_catalog: Task vocabulary for the user message.

        Returns:
            A ``(spec, data)`` tuple. On success, ``spec`` is validated and ``data`` is the
            expanded model JSON. On failure, ``spec`` is ``None`` and ``data`` is the raw
            response object.
        """
        data = self._inference_backend.run_json(
            StructuredOutputRequest(
                schema_name="ArenaEnvGraphSpec",
                schema=self._schema,
                system=self._system_prompt(),
                user=self._user_message(
                    prompt,
                    asset_catalog,
                    relation_catalog,
                    task_catalog,
                ),
                retry_label="generate_spec",
            )
        )
        data = _expand_model_output(data)
        try:
            spec = ArenaEnvGraphSpec.model_validate(data)
        except ValidationError as exc:
            traces.extend(format_validation_error(exc))
            return None, data
        traces.extend(collect_agent_ready_task_validation_traces(spec))
        return spec, data

    @staticmethod
    def _user_message(
        prompt: str,
        asset_catalog: Any,
        relation_catalog: Any,
        task_catalog: Any,
    ) -> str:
        vocabulary = (
            f"{asset_catalog.to_catalog_string()}\n\n"
            f"{relation_catalog.to_catalog_string()}\n\n"
            f"{task_catalog.to_catalog_string()}"
        )
        return f"{vocabulary}\n\nUSER PROMPT:\n{prompt}"

    @staticmethod
    def _system_prompt() -> str:
        return """\
You are an environment-generator for robot manipulation tasks.
Convert a natural-language prompt into an ArenaEnvGraphSpec.

GUIDANCE:
- Follow the per-field ``description`` strings in the schema.
- Use only exact names from the catalog for ``registry_name``:
  EMBODIMENTS for ``embodiment``, BACKGROUNDS for ``background``, and OBJECTS for ``objects``.
- Do NOT hallucinate asset names — every ``registry_name`` must appear verbatim in the catalog.
  If the prompt includes the exact registry name, use it.
  If multiple reasonable matches are found, return the closest match or the one with the most specific name.
  An entry naming an object more generically than the prompt does is still that object, e.g. a
  ``blue_soda_can`` entry is the prompt's pepsi can.
  If no entry reasonably matches an object, leave that object out and build the rest of the scene.
- NEVER return an empty spec: ``env_name`` names the scene and ``task.subtasks`` holds at least one
  entry. ``objects`` holds one entry per asset the prompt asks to spawn, and stays empty when
  everything the prompt names is part of the background.
- For embodiment, if the prompt only mention the robot family (driod/franka) and there are multiple
  variance of that family in EMBODIMENTS, pick the one with the default tag.
- Add no object the prompt does not name; a scene holds what was asked for and nothing else.
- For multiple instances of the same registry asset, use semantic (left/right) or numerical (1/2/3)
  suffixes in ``id``.
- Use ``object_sets`` only when one object varies across environments; list its variants as ``members``.
  Every member must be an OBJECTS entry marked ``type=rigid``.
- A surface, appliance, or fixture the prompt names as part of a room-scale background — a counter
  top, a shelf, the floor — is its own ``object_reference``, and relations and task params name that
  node rather than the room. Add one only for what the prompt names, never for a surface it leaves
  out. Such a part is an ``object_reference`` only, never also an ``objects`` entry — ``objects``
  holds spawned assets from the OBJECTS catalog.
- When the background asset is itself the surface the prompt names (a table scene, whose BACKGROUNDS
  entry names that table), relations name the background node directly and it gets no
  ``object_reference``. Leave ``object_references`` unset whenever the prompt names no part within
  the background.
- For each ``object_reference``, leave ``prim_path`` empty.
- REQUIRED: anchor the scene's fixed ground with an ``is_anchor`` relation — every surface the
  prompt names inside the background, or the background itself when it names none. More than one
  is allowed, and neither the embodiment nor a spawned ``objects`` entry is ever one.
- Every object and the embodiment need an ``on`` relation naming the surface the prompt rests it
  on, and an anchor when the prompt names none.
- Emit the remaining relations the prompt states, the embodiment's own among them (e.g. 'next to
  the counter top', 'facing the fridge').
- ``task.composition`` is 'atomic' for a single subtask; over several it is 'sequential' only when
  the prompt names an order ('then', 'after', 'first'), and 'parallel' otherwise. One arm still
  works one object at a time, so repeating a subtask over several objects is 'parallel'.
- Write ``task.description`` as the task alone: a short imperative naming the objects its params
  point at (e.g. 'Pick up the banana, and place it into the plate'), rewritten from the user
  prompt. It is not the prompt itself, and it names no distractor or scene detail.
"""


def _expand_model_output(data: Any) -> Any:
    """Rewrite the parts of a model response that only name what the rest of it already says.

    Args:
        data: Parsed model JSON, returned unchanged when it carries no such part.

    Returns:
        The expanded response, or the argument itself when nothing needed rewriting.
    """
    if not isinstance(data, dict):
        return data
    for expansion in (
        _expand_empty_relation_references,
        _expand_nested_object_sets,
        _expand_registry_name_references,
        _expand_single_subtask_composition,
        _expand_placement_validators,
    ):
        data = expansion(data)
    return data


def _expand_placement_validators(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the placement validators; no prompt selects checks, so a model choice can only narrow them."""
    if data.get("placement_validators") is None:
        return data
    print("[generate_spec] expanded placement_validators away; every build-time check runs.", flush=True)
    return {**data, "placement_validators": None}


def _expand_empty_relation_references(data: dict[str, Any]) -> dict[str, Any]:
    """Read a relation's empty-string reference as no reference; strict output cannot omit the key."""
    relations = data.get("relations")
    if not isinstance(relations, list):
        return data
    return {
        **data,
        "relations": [
            (
                {**relation, "reference": None}
                if isinstance(relation, dict) and relation.get("reference") == ""
                else relation
            )
            for relation in relations
        ],
    }


def _expand_registry_name_references(data: dict[str, Any]) -> dict[str, Any]:
    """Point relation endpoints and task params that name an asset's registry_name at its node id.

    A catalog lists registry names and a graph is wired by node id, so the two are easy to confuse
    where only one asset carries the name.
    """
    assets = [data.get("embodiment"), data.get("background"), *(data.get("objects") or [])]
    assets = [asset for asset in assets if isinstance(asset, dict) and isinstance(asset.get("id"), str)]
    ids_by_registry_name: dict[str, set[str]] = {}
    for asset in assets:
        if isinstance(asset.get("registry_name"), str):
            ids_by_registry_name.setdefault(asset["registry_name"], set()).add(asset["id"])
    node_ids = _node_ids(data)
    renames = {
        registry_name: next(iter(ids))
        for registry_name, ids in ids_by_registry_name.items()
        if len(ids) == 1 and registry_name not in node_ids
    }
    if not renames:
        return data
    expanded = {**data, "relations": [_rename_relation_endpoints(rel, renames) for rel in data.get("relations") or []]}
    task = expanded.get("task")
    if isinstance(task, dict) and isinstance(task.get("subtasks"), list):
        expanded["task"] = {**task, "subtasks": [_rename_task_params(sub, renames) for sub in task["subtasks"]]}
    for registry_name, node_id in renames.items():
        if registry_name in _referenced_node_ids(data):
            print(f"[generate_spec] expanded references to {registry_name!r} into node {node_id!r}", flush=True)
    return expanded


def _rename_relation_endpoints(relation: Any, renames: dict[str, str]) -> Any:
    """Return ``relation`` with each endpoint naming a registry_name replaced by its node id."""
    if not isinstance(relation, dict):
        return relation
    endpoints = {key: renames[relation[key]] for key in ("subject", "reference") if relation.get(key) in renames}
    return {**relation, **endpoints} if endpoints else relation


def _rename_task_params(subtask: Any, renames: dict[str, str]) -> Any:
    """Return ``subtask`` with each param naming a registry_name replaced by its node id."""
    params = subtask.get("params") if isinstance(subtask, dict) else None
    if not isinstance(params, dict):
        return subtask
    return {
        **subtask,
        "params": {
            name: renames.get(value, value) if isinstance(value, str) else value for name, value in params.items()
        },
    }


def _expand_single_subtask_composition(data: dict[str, Any]) -> dict[str, Any]:
    """Label a composition over one subtask 'atomic'; only two or more subtasks can have an ordering."""
    task = data.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("subtasks"), list):
        return data
    composition = task.get("composition")
    if len(task["subtasks"]) != 1 or composition == TaskCompositionType.ATOMIC.value:
        return data
    print(f"[generate_spec] expanded composition {composition!r} over a single subtask to 'atomic'", flush=True)
    return {**data, "task": {**task, "composition": TaskCompositionType.ATOMIC.value}}


def _expand_nested_object_sets(data: dict[str, Any]) -> dict[str, Any]:
    """Inline object sets used as another set's members, and drop the ones nothing else refers to.

    A member names a registered rigid asset, so a set nested in another one contributes the assets
    it draws from. Nesting a set per variant is how a model groups variants it cannot nest.
    """
    object_sets = data.get("object_sets")
    if not isinstance(object_sets, list):
        return data
    members_by_id = {
        entry["id"]: entry["members"]
        for entry in object_sets
        if isinstance(entry, dict) and isinstance(entry.get("id"), str) and isinstance(entry.get("members"), list)
    }
    inlined: set[str] = set()
    expanded_sets: list[Any] = []
    for entry in object_sets:
        members = members_by_id.get(entry["id"]) if isinstance(entry, dict) else None
        nested = [member for member in members or [] if member in members_by_id and member != entry["id"]]
        if not nested:
            expanded_sets.append(entry)
            continue
        flattened = [name for member in members for name in (members_by_id.get(member) or [member])]
        inlined.update(nested)
        print(f"[generate_spec] expanded object set {entry['id']!r} members {members} into {flattened}", flush=True)
        expanded_sets.append({**entry, "members": flattened})
    # An inlined set still named by a relation or a task param is a node in its own right, so keep it.
    orphaned = inlined - _referenced_node_ids(data)
    if not orphaned:
        return data
    return {**data, "object_sets": [entry for entry in expanded_sets if entry.get("id") not in orphaned]}


def _node_ids(data: dict[str, Any]) -> set[str]:
    """Return the ids of every node a relation or task param may name."""
    nodes = [
        data.get("embodiment"),
        data.get("background"),
        *(data.get("objects") or []),
        *(data.get("object_sets") or []),
        *(data.get("object_references") or []),
    ]
    return {node["id"] for node in nodes if isinstance(node, dict) and isinstance(node.get("id"), str)}


def _referenced_node_ids(data: dict[str, Any]) -> set[str]:
    """Return the node ids the spec's relations and task params name."""
    referenced: set[str] = set()
    for relation in data.get("relations") or []:
        if isinstance(relation, dict):
            referenced.update(value for value in (relation.get("subject"), relation.get("reference")) if value)
    task = data.get("task")
    for subtask in (task.get("subtasks") or []) if isinstance(task, dict) else []:
        params = subtask.get("params") if isinstance(subtask, dict) else None
        referenced.update(value for value in (params or {}).values() if isinstance(value, str))
    return referenced
