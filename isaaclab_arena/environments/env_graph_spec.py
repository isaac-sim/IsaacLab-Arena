# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvGraphNodeSpec:
    """Node in an environment graph. Could be an object, an object reference, an embodiment, a background, etc."""

    id: str
    name: str
    type: str
    parent: str | None = None
    prim_path: str | None = None
    object_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvGraphNodeSpec:
        assert isinstance(data, dict), f"Node spec must be a dict, got {type(data).__name__}"
        return cls(
            id=_required_str(data, "id"),
            name=_required_str(data, "name"),
            type=_required_str(data, "type"),
            parent=_optional_str(data, "parent"),
            prim_path=_optional_str(data, "prim_path"),
            object_type=_optional_str(data, "object_type"),
            params=_optional_dict(data, "params"),
        )


@dataclass(frozen=True)
class EnvGraphConstraintSpec:
    """Constraint edge in an environment graph state spec. It defines a spatial or task constraint between two nodes."""

    id: str
    type: str
    parent: str | None = None
    child: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvGraphConstraintSpec:
        assert isinstance(data, dict), f"Constraint spec must be a dict, got {type(data).__name__}"
        return cls(
            id=_required_str(data, "id"),
            type=_required_str(data, "type"),
            parent=_optional_str(data, "parent"),
            child=_optional_str(data, "child"),
            params=_optional_dict(data, "params"),
        )


@dataclass(frozen=True)
class EnvGraphEdgesSpec:
    """Grouped spatial and task constraints."""

    spatial_constraints: list[EnvGraphConstraintSpec] = field(default_factory=list)
    task_constraints: list[EnvGraphConstraintSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EnvGraphEdgesSpec:
        data = data or {}
        assert isinstance(data, dict), f"Edges spec must be a dict, got {type(data).__name__}"
        return cls(
            spatial_constraints=[
                EnvGraphConstraintSpec.from_dict(edge) for edge in data.get("spatial_constraints", [])
            ],
            task_constraints=[EnvGraphConstraintSpec.from_dict(edge) for edge in data.get("task_constraints", [])],
        )


@dataclass(frozen=True)
class EnvGraphStateSpec:
    """Snapshots of the environment state in the graph. Could be an initial, or intermediate, or final state."""

    id: str
    name: str
    edges: EnvGraphEdgesSpec = field(default_factory=EnvGraphEdgesSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvGraphStateSpec:
        assert isinstance(data, dict), f"State spec must be a dict, got {type(data).__name__}"
        return cls(
            id=_required_str(data, "id"),
            name=_required_str(data, "name"),
            edges=EnvGraphEdgesSpec.from_dict(data.get("edges")),
        )


@dataclass(frozen=True)
class EnvGraphTaskSpec:
    """Task entry in an environment graph. It defines the task to be completed in the environment."""

    id: str
    name: str
    type: str
    state_specs: dict[str, str] = field(default_factory=dict)
    task_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvGraphTaskSpec:
        assert isinstance(data, dict), f"Task spec must be a dict, got {type(data).__name__}"
        state_specs = data.get("state_specs", {})
        assert isinstance(state_specs, dict), "Task state_specs must be a dict"
        return cls(
            id=_required_str(data, "id"),
            name=_required_str(data, "name"),
            type=_required_str(data, "type"),
            state_specs={str(k): str(v) for k, v in state_specs.items()},
            task_args=_optional_dict(data, "task_args"),
        )


@dataclass(frozen=True)
class EnvGraphSpec:
    """Typed representation of an environment graph YAML file. It defines the nodes, tasks, and states of the environment graph."""

    name: str
    nodes: list[EnvGraphNodeSpec] = field(default_factory=list)
    tasks: list[EnvGraphTaskSpec] = field(default_factory=list)
    state_specs: list[EnvGraphStateSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvGraphSpec:
        """Load an environment graph spec from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvGraphSpec:
        """Build an environment graph spec from parsed YAML data."""
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return cls(
            name=_required_str(data, "name"),
            nodes=[EnvGraphNodeSpec.from_dict(node) for node in data.get("nodes", [])],
            tasks=[EnvGraphTaskSpec.from_dict(task) for task in data.get("tasks", [])],
            state_specs=[EnvGraphStateSpec.from_dict(state_spec) for state_spec in data.get("state_specs", [])],
        )

    @property
    def nodes_by_id(self) -> dict[str, EnvGraphNodeSpec]:
        """Return nodes keyed by id."""
        return {node.id: node for node in self.nodes}

    @property
    def tasks_by_id(self) -> dict[str, EnvGraphTaskSpec]:
        """Return tasks keyed by id."""
        return {task.id: task for task in self.tasks}

    @property
    def state_specs_by_id(self) -> dict[str, EnvGraphStateSpec]:
        """Return state specs keyed by id."""
        return {state_spec.id: state_spec for state_spec in self.state_specs}


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
