# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SerializeAsAny, field_validator, model_validator

from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphCliOverrideSpec,
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskConstraintSpec,
    ArenaEnvGraphTaskConstraintType,
    ArenaEnvGraphTaskSpec,
    parse_graph_node,
    UnresolvedArenaEnvGraphTaskSpec,
)
from isaaclab_arena.environments.graph_spec_utils import (
    assert_cli_override_specs_reference_nodes,
    assert_references_exist,
    assert_spatial_constraint_shapes,
    assert_unique_ids,
)

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Re-exported for callers that already import these names from this module.
__all__ = [
    "ArenaEnvGraphCliOverrideSpec",
    "ArenaEnvGraphNodeSpec",
    "ArenaEnvGraphNodeType",
    "ArenaEnvGraphObjectReferenceNodeSpec",
    "ArenaEnvGraphSpatialConstraintSpec",
    "ArenaEnvGraphSpec",
    "ArenaEnvGraphStateSpec",
    "ArenaEnvGraphTaskConstraintSpec",
    "ArenaEnvGraphTaskConstraintType",
    "ArenaEnvGraphTaskSpec",
    "UnresolvedArenaEnvGraphSpec",
    "UnresolvedArenaEnvGraphTaskSpec",
]


class ArenaEnvGraphSpec(BaseModel):
    """Typed representation of an environment graph YAML file."""

    env_name: str = Field(min_length=1)
    nodes: list[SerializeAsAny[ArenaEnvGraphNodeSpec]] = Field(default_factory=list)
    tasks: list[ArenaEnvGraphTaskSpec] = Field(default_factory=list)
    state_specs: list[ArenaEnvGraphStateSpec] = Field(default_factory=list)
    cli_override_specs: list[ArenaEnvGraphCliOverrideSpec] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _parse_nodes(cls, nodes: Any) -> list[Any]:
        if nodes is None:
            return []
        if not isinstance(nodes, list):
            raise ValueError("Field 'nodes' must be a list")
        return [parse_graph_node(node) for node in nodes]

    @model_validator(mode="after")
    def validate(self) -> ArenaEnvGraphSpec:
        """Check unique ids, cross-references, constraint shapes, and CLI overrides."""
        assert_unique_ids(self.nodes, self.tasks, self.state_specs)
        assert_references_exist(self.nodes, self.tasks, self.state_specs)
        assert_spatial_constraint_shapes(self.state_specs)
        assert_cli_override_specs_reference_nodes(self.nodes, self.cli_override_specs)
        return self

    @staticmethod
    def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
        """Load a graph YAML into a dict. Fail with a clear message if the file is missing."""
        path = Path(path)
        assert path.is_file(), f"Env graph spec YAML not found: {path}"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArenaEnvGraphSpec:
        return cls.from_dict(cls._load_yaml_dict(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArenaEnvGraphSpec:
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return cls.model_validate(data)

    @staticmethod
    def read_cli_override_specs(path: str | Path) -> list[ArenaEnvGraphCliOverrideSpec]:
        """Read just the ``cli_override_specs`` section of a graph YAML, skipping the rest.

        The CLI flags need to be registered before the simulator starts. Loading the full
        graph would import ``pxr`` too early, so this only reads the override entries.
        """
        raw_specs = ArenaEnvGraphSpec._load_yaml_dict(path).get("cli_override_specs") or []
        return [ArenaEnvGraphCliOverrideSpec.model_validate(entry) for entry in raw_specs]

    def apply_cli_override_args(self, args_cli: argparse.Namespace) -> None:
        """Apply the CLI override flags to this graph, in place.

        For each override, set the target node's asset ``name`` to the value passed on the
        command line. Flags left unset are skipped, so an untouched graph stays the same.
        """
        nodes_by_id = self.nodes_by_id
        for override in self.cli_override_specs:
            new_name = getattr(args_cli, override.dest, None)
            if new_name is not None:
                nodes_by_id[override.target_node_id].name = new_name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain YAML mapping — the inverse of ``from_dict``."""
        return self.model_dump(mode="json", exclude_none=True)

    def write_yaml(self, path: str | Path) -> None:
        """Validate this spec and write it to ``path`` as YAML."""
        self.validate()
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the plain YAML mapping — the inverse of ``from_dict``.

        Enums become their string values and tuples become lists, so the result round-trips
        through ``yaml.safe_dump`` and back through ``from_dict``.
        """
        return {
            "env_name": self.env_name,
            "nodes": [_node_to_dict(node) for node in self.nodes],
            "tasks": [_task_to_dict(task) for task in self.tasks],
            "state_specs": [_state_spec_to_dict(state) for state in self.state_specs],
        }

    def write_yaml(self, path: str | Path) -> None:
        """Validate this spec and write it to ``path`` as YAML."""
        self.validate()
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @property
    def nodes_by_id(self) -> dict[str, ArenaEnvGraphNodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def tasks_by_id(self) -> dict[str, ArenaEnvGraphTaskSpec]:
        return {task.id: task for task in self.tasks}

    @property
    def state_specs_by_id(self) -> dict[str, ArenaEnvGraphStateSpec]:
        return {state_spec.id: state_spec for state_spec in self.state_specs}

    def to_arena_env(self) -> IsaacLabArenaEnvironment:
        """Convert this graph spec into an `IsaacLabArenaEnvironment`.

        The first ``state_spec`` is used as the scene's initial state.
        """
        # Lazy import: build_arena_env_from_graph_spec pulls in Scene -> phyx_utils ->
        # pxr.PhysxSchema, which requires SimulationApp. Keeping the import here lets
        # data-only consumers of the spec (parsers, tests) import this module before
        # SimulationApp is started.
        # TODO(xinjieyao, 2026-05-26): once `build_arena_env_from_graph_spec` aggregates across all state_specs,
        # this wrapper stays single-arg — no caller-side selection is needed.
        from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

        return build_arena_env_from_graph_spec(self)


@dataclass
class UnresolvedArenaEnvGraphSpec:
    """A partially-populated env graph before the ``StateSpecResolver`` chains the per-task state specs.

    Carries the same nodes and task identities as an ``ArenaEnvGraphSpec``, but its tasks
    have no snapshots of the intermediate and final states and it holds only the initial
    state (``state_spec_0``); the ``StateSpecResolver`` derives the intermediate and final states.
    """

    env_name: str
    nodes: list[ArenaEnvGraphNodeSpec] = field(default_factory=list)
    tasks: list[UnresolvedArenaEnvGraphTaskSpec] = field(default_factory=list)
    state_specs: list[ArenaEnvGraphStateSpec] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UnresolvedArenaEnvGraphSpec":
        return cls.from_dict(ArenaEnvGraphSpec._load_yaml_dict(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnresolvedArenaEnvGraphSpec":
        data = as_dict(data, "Unresolved env graph spec")
        spec = cls(
            env_name=required_str(data, "env_name"),
            nodes=parse_list(data, "nodes", _parse_node),
            tasks=parse_list(data, "tasks", _parse_unresolved_task),
            state_specs=parse_list(data, "state_specs", _parse_state_spec),
            source=data,
        )
        # Since tasks have no state-id wiring yet, we skip the task wiring validation.
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate ids, node/constraint references, and shapes — minus the unset task wiring."""
        assert len(self.state_specs) == 1, (
            "unresolved env graph must define exactly the initial state (state_spec_0); "
            f"got {len(self.state_specs)} state specs"
        )
        assert_unique_ids(self.nodes, self.tasks, self.state_specs)
        assert_references_exist(self.nodes, self.tasks, self.state_specs, check_task_wiring=False)
        assert_spatial_constraint_shapes(self.state_specs)


def _parse_node(data: Any) -> ArenaEnvGraphNodeSpec:
    data = as_dict(data, "Node spec")
    node_type = required_enum(data, "type", ArenaEnvGraphNodeType)
    common = dict(
        id=required_str(data, "id"),
        name=required_str(data, "name"),
        type=node_type,
        params=optional_dict(data, "params"),
    )
    if node_type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
        return ArenaEnvGraphObjectReferenceNodeSpec(
            **common,
            parent=required_str(data, "parent"),
            prim_path=required_str(data, "prim_path"),
            object_type=required_enum(data, "object_type", ObjectType),
        )
    return ArenaEnvGraphNodeSpec(**common)


def _parse_spatial_constraint(data: Any) -> ArenaEnvGraphSpatialConstraintSpec:
    data = as_dict(data, "Spatial constraint spec")
    constraint_type = required_enum(data, "type", ArenaEnvGraphSpatialConstraintType)
    params = optional_dict(data, "params")
    # Parse optional position_xyz and rotation_xyzw fields and check their lengths.
    if params and "position_xyz" in params:
        params["position_xyz"] = required_number_sequence(params, "position_xyz", 3)
    if params and "rotation_xyzw" in params:
        params["rotation_xyzw"] = required_number_sequence(params, "rotation_xyzw", 4)

    return ArenaEnvGraphSpatialConstraintSpec(
        id=required_str(data, "id"),
        type=constraint_type,
        parent=required_str(data, "parent"),
        child=optional_str(data, "child"),
        params=params,
    )


def _parse_task_constraint(data: Any) -> ArenaEnvGraphTaskConstraintSpec:
    data = as_dict(data, "Task constraint spec")
    return ArenaEnvGraphTaskConstraintSpec(
        id=required_str(data, "id"),
        type=required_enum(data, "type", ArenaEnvGraphTaskConstraintType),
        parent=required_str(data, "parent"),
        child=optional_str(data, "child"),
        params=optional_dict(data, "params"),
    )


def _parse_state_spec(data: Any) -> ArenaEnvGraphStateSpec:
    data = as_dict(data, "State spec")
    assert "edges" not in data, "State spec must define spatial_constraints and task_constraints directly"
    return ArenaEnvGraphStateSpec(
        id=required_str(data, "id"),
        spatial_constraints=parse_list(data, "spatial_constraints", _parse_spatial_constraint),
        task_constraints=parse_list(data, "task_constraints", _parse_task_constraint),
    )


def _parse_task(data: Any) -> ArenaEnvGraphTaskSpec:
    data = as_dict(data, "Task spec")
    for old_key in ("state_specs", "initial_state_spec", "success_state_spec"):
        assert old_key not in data, "Task spec must use initial_state_spec_id and success_state_spec_id"
    return ArenaEnvGraphTaskSpec(
        id=required_str(data, "id"),
        type=required_str(data, "type"),
        initial_state_spec_id=required_str(data, "initial_state_spec_id"),
        success_state_spec_id=required_str(data, "success_state_spec_id"),
        task_args=optional_dict(data, "task_args"),
    )


def _parse_unresolved_task(data: Any) -> UnresolvedArenaEnvGraphTaskSpec:
    """Parse a task that has no state-id wiring yet; any ``*_state_spec_id`` keys are ignored."""
    data = as_dict(data, "Unresolved task spec")
    return UnresolvedArenaEnvGraphTaskSpec(
        id=required_str(data, "id"),
        type=required_str(data, "type"),
        task_args=optional_dict(data, "task_args"),
    )


def _yaml_safe(value: Any) -> Any:
    """Coerce parsed values into YAML-dumpable primitives (tuples -> lists, recursively)."""
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    return value


def _node_to_dict(node: ArenaEnvGraphNodeSpec) -> dict[str, Any]:
    data: dict[str, Any] = {"id": node.id, "name": node.name, "type": node.type.value}
    if isinstance(node, ArenaEnvGraphObjectReferenceNodeSpec):
        data["parent"] = node.parent
        data["prim_path"] = node.prim_path
        data["object_type"] = node.object_type.value
    if node.params:
        data["params"] = _yaml_safe(node.params)
    return data


def _task_to_dict(task: ArenaEnvGraphTaskSpec) -> dict[str, Any]:
    return {
        "id": task.id,
        "type": task.type,
        "initial_state_spec_id": task.initial_state_spec_id,
        "success_state_spec_id": task.success_state_spec_id,
        "task_args": _yaml_safe(task.task_args),
    }


def _state_spec_to_dict(state: ArenaEnvGraphStateSpec) -> dict[str, Any]:
    return {
        "id": state.id,
        "spatial_constraints": [_constraint_to_dict(c) for c in state.spatial_constraints],
        "task_constraints": [_constraint_to_dict(c) for c in state.task_constraints],
    }


def _constraint_to_dict(
    constraint: ArenaEnvGraphSpatialConstraintSpec | ArenaEnvGraphTaskConstraintSpec,
) -> dict[str, Any]:
    data: dict[str, Any] = {"id": constraint.id, "type": constraint.type.value, "parent": constraint.parent}
    if constraint.child is not None:
        data["child"] = constraint.child
    if constraint.params:
        data["params"] = _yaml_safe(constraint.params)
    return data
