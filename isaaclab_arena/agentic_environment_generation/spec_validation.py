# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for agent-generated environment graph specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

if TYPE_CHECKING:
    from isaaclab_arena.agentic_environment_generation.catalogues import (
        AssetCatalogue,
        RelationCatalogue,
        TaskCatalogue,
    )


_ASSERTION_FAILED_PREFIX = "Assertion failed, "


def _clean_validation_msg(msg: str) -> str:
    """Strip pydantic's assertion wrapper from validator error text."""
    if msg.startswith(_ASSERTION_FAILED_PREFIX):
        return msg[len(_ASSERTION_FAILED_PREFIX) :]
    return msg


def format_validation_error(exc: ValidationError) -> list[str]:
    """Flatten a Pydantic ``ValidationError`` into human-readable trace lines."""
    lines: list[str] = []
    for err in exc.errors():
        msg = _clean_validation_msg(err["msg"])
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"{loc}: {msg}" if loc else msg)
    return lines


def collect_agent_ready_validation_trace(
    spec: ArenaEnvGraphSpec,
    asset_catalog: AssetCatalogue,
    task_catalog: TaskCatalogue,
    relation_catalog: RelationCatalogue,
) -> list[str]:
    """Return spec violations against the exact catalogues exposed to the agent."""
    traces: list[str] = []

    # Check every registry_name against the asset catalogue the agent was given.
    asset_names = {
        "EMBODIMENTS": {entry["name"] for entry in asset_catalog.embodiments},
        "BACKGROUNDS": {entry["name"] for entry in asset_catalog.backgrounds},
        "OBJECTS": {entry["name"] for entry in asset_catalog.objects},
    }
    # Embodiment and background are required singles; flag any name outside the catalogue.
    if spec.embodiment.registry_name not in asset_names["EMBODIMENTS"]:
        traces.append(f"Embodiment registry_name {spec.embodiment.registry_name!r} is not in the EMBODIMENTS catalog")
    if spec.background.registry_name not in asset_names["BACKGROUNDS"]:
        traces.append(f"Background registry_name {spec.background.registry_name!r} is not in the BACKGROUNDS catalog")
    # Each movable object must resolve to a known OBJECTS entry.
    for obj in spec.objects:
        if obj.registry_name not in asset_names["OBJECTS"]:
            traces.append(f"Object {obj.id!r} registry_name {obj.registry_name!r} is not in the OBJECTS catalog")
    # Object-set members are registry names too; validate each one the same way.
    for object_set in spec.object_sets or []:
        for member in object_set.members:
            if member not in asset_names["OBJECTS"]:
                traces.append(
                    f"Object set {object_set.id!r} member registry_name {member!r} is not in the OBJECTS catalog"
                )

    # Check each subtask kind and its params against the task catalogue.
    task_entries = {entry.name: entry for entry in task_catalog.tasks}
    for task in spec.task.subtasks:
        entry = task_entries.get(task.kind)
        # Unknown task kinds cannot be repaired via params; skip further checks.
        if entry is None:
            traces.append(f"Task {task.kind!r} is not in the TASKS catalog")
            continue
        # Required params must be present; anything outside required|optional is rejected.
        for required_param in entry.required_params:
            if required_param not in task.params:
                traces.append(f"Task {task.kind!r} is missing required param {required_param!r}")
        supported_params = set(entry.required_params) | set(entry.optional_params)
        for unsupported_param in sorted(set(task.params) - supported_params):
            traces.append(
                f"Task {task.kind!r} has unsupported param {unsupported_param!r}; "
                f"supported params are {sorted(supported_params)!r}"
            )

    # Check each relation kind and its params against the relation catalogue.
    relation_entries = {entry.name: entry for entry in relation_catalog.relations}
    for relation in spec.relations:
        entry = relation_entries.get(relation.kind)
        # Same pattern as tasks: unknown kinds first, then required/unsupported params.
        if entry is None:
            traces.append(f"Relation {relation.kind!r} is not in the RELATIONS catalog")
            continue
        for required_param in entry.required_params:
            if required_param not in relation.params:
                traces.append(f"Relation {relation.kind!r} is missing required param {required_param!r}")
        supported_params = set(entry.required_params) | set(entry.optional_params)
        for unsupported_param in sorted(set(relation.params) - supported_params):
            traces.append(
                f"Relation {relation.kind!r} has unsupported param {unsupported_param!r}; "
                f"supported params are {sorted(supported_params)!r}"
            )
    return traces
