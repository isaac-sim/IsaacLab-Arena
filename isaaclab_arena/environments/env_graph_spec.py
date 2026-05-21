# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class EnvGraphNodeSpec:
    """Node in an environment graph.

    Could be an object, an object reference, an embodiment, a background, etc.
    """

    id: str
    name: str
    type: str
    parent: str | None = None
    prim_path: str | None = None
    object_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvGraphConstraintSpec:
    """Constraint edge in an environment graph state spec.

    It defines a spatial or task constraint between two nodes.
    """

    id: str
    type: str
    parent: str | None = None
    child: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvGraphEdgesSpec:
    """Grouped spatial and task constraints."""

    spatial_constraints: list[EnvGraphConstraintSpec] = field(default_factory=list)
    task_constraints: list[EnvGraphConstraintSpec] = field(default_factory=list)


@dataclass(frozen=True)
class EnvGraphStateSpec:
    """Snapshot of the environment state in the graph.

    Could be an initial, intermediate, or final state.
    """

    id: str
    name: str
    edges: EnvGraphEdgesSpec = field(default_factory=EnvGraphEdgesSpec)


@dataclass(frozen=True)
class EnvGraphTaskSpec:
    """Task entry in an environment graph."""

    id: str
    name: str
    type: str
    state_specs: dict[str, str] = field(default_factory=dict)
    task_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvGraphSpec:
    """Typed representation of an environment graph YAML file."""

    name: str
    nodes: list[EnvGraphNodeSpec] = field(default_factory=list)
    tasks: list[EnvGraphTaskSpec] = field(default_factory=list)
    state_specs: list[EnvGraphStateSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvGraphSpec":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvGraphSpec":
        data = _as_dict(data, "Env graph spec")
        nodes = _parse_list(data, "nodes", _parse_node)
        tasks = _parse_list(data, "tasks", _parse_task)
        state_specs = _parse_list(data, "state_specs", _parse_state_spec)

        _assert_unique_ids(nodes, "node")
        _assert_unique_ids(tasks, "task")
        _assert_unique_ids(state_specs, "state spec")
        _assert_references_exist(nodes, tasks, state_specs)

        return cls(
            name=_required_str(data, "name"),
            nodes=nodes,
            tasks=tasks,
            state_specs=state_specs,
        )

    @property
    def nodes_by_id(self) -> dict[str, EnvGraphNodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def tasks_by_id(self) -> dict[str, EnvGraphTaskSpec]:
        return {task.id: task for task in self.tasks}

    @property
    def state_specs_by_id(self) -> dict[str, EnvGraphStateSpec]:
        return {state_spec.id: state_spec for state_spec in self.state_specs}


def _parse_node(data: Any) -> EnvGraphNodeSpec:
    data = _as_dict(data, "Node spec")
    return EnvGraphNodeSpec(
        id=_required_str(data, "id"),
        name=_required_str(data, "name"),
        type=_required_str(data, "type"),
        parent=_optional_str(data, "parent"),
        prim_path=_optional_str(data, "prim_path"),
        object_type=_optional_str(data, "object_type"),
        params=_optional_dict(data, "params"),
    )


def _parse_constraint(data: Any) -> EnvGraphConstraintSpec:
    data = _as_dict(data, "Constraint spec")
    return EnvGraphConstraintSpec(
        id=_required_str(data, "id"),
        type=_required_str(data, "type"),
        parent=_optional_str(data, "parent"),
        child=_optional_str(data, "child"),
        params=_optional_dict(data, "params"),
    )


def _parse_edges(data: dict[str, Any] | None) -> EnvGraphEdgesSpec:
    if data is None:
        data = {}
    data = _as_dict(data, "Edges spec")
    return EnvGraphEdgesSpec(
        spatial_constraints=_parse_list(data, "spatial_constraints", _parse_constraint),
        task_constraints=_parse_list(data, "task_constraints", _parse_constraint),
    )


def _parse_state_spec(data: Any) -> EnvGraphStateSpec:
    data = _as_dict(data, "State spec")
    return EnvGraphStateSpec(
        id=_required_str(data, "id"),
        name=_required_str(data, "name"),
        edges=_parse_edges(data.get("edges")),
    )


def _parse_task(data: Any) -> EnvGraphTaskSpec:
    data = _as_dict(data, "Task spec")
    return EnvGraphTaskSpec(
        id=_required_str(data, "id"),
        name=_required_str(data, "name"),
        type=_required_str(data, "type"),
        state_specs=_optional_str_map(data, "state_specs"),
        task_args=_optional_dict(data, "task_args"),
    )


def _as_dict(data: Any, spec_name: str) -> dict[str, Any]:
    assert isinstance(data, dict), f"{spec_name} must be a dict, got {type(data).__name__}"
    return data


def _parse_list(data: dict[str, Any], key: str, parser: Callable[[Any], T]) -> list[T]:
    values = data.get(key, [])
    assert isinstance(values, list), f"Field '{key}' must be a list"
    return [parser(value) for value in values]


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    assert isinstance(value, str) and value, f"Missing required string field '{key}'"
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    assert value is None or isinstance(value, str), f"Optional field '{key}' must be a string when set"
    return value


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    assert value is None or isinstance(value, dict), f"Optional field '{key}' must be a dict when set"
    return dict(value or {})


def _optional_str_map(data: dict[str, Any], key: str) -> dict[str, str]:
    return {str(k): str(v) for k, v in _optional_dict(data, key).items()}


def _assert_unique_ids(specs: list[Any], spec_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            duplicates.add(spec.id)
        seen.add(spec.id)
    assert not duplicates, f"Duplicate {spec_name} ids found: {sorted(duplicates)}"


def _assert_references_exist(
    nodes: list[EnvGraphNodeSpec],
    tasks: list[EnvGraphTaskSpec],
    state_specs: list[EnvGraphStateSpec],
) -> None:
    node_ids = {node.id for node in nodes}
    state_spec_ids = {state_spec.id for state_spec in state_specs}

    for task in tasks:
        for label, state_spec_id in task.state_specs.items():
            assert state_spec_id in state_spec_ids, (
                f"Task '{task.id}' references unknown state spec '{state_spec_id}' for '{label}'"
            )

    for state_spec in state_specs:
        constraints = state_spec.edges.spatial_constraints + state_spec.edges.task_constraints
        for constraint in constraints:
            if constraint.parent is not None:
                assert constraint.parent in node_ids, (
                    f"Constraint '{constraint.id}' references unknown parent node '{constraint.parent}'"
                )
            if constraint.child is not None:
                assert constraint.child in node_ids, (
                    f"Constraint '{constraint.id}' references unknown child node '{constraint.child}'"
                )
