# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskConstraintSpec,
    ArenaEnvGraphTaskConstraintType,
    ArenaEnvGraphTaskSpec,
)
from isaaclab_arena.environments.graph_spec_utils import (
    as_dict,
    assert_references_exist,
    assert_spatial_constraint_shapes,
    assert_unique_ids,
    optional_dict,
    optional_str,
    parse_list,
    required_enum,
    required_number_sequence,
    required_str,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Re-exported for callers that already import these names from this module.
__all__ = [
    "ArenaEnvGraphNodeSpec",
    "ArenaEnvGraphNodeType",
    "ArenaEnvGraphObjectReferenceNodeSpec",
    "ArenaEnvGraphSpatialConstraintSpec",
    "ArenaEnvGraphSpatialConstraintType",
    "ArenaEnvGraphSpec",
    "ArenaEnvGraphStateSpec",
    "ArenaEnvGraphTaskConstraintSpec",
    "ArenaEnvGraphTaskConstraintType",
    "ArenaEnvGraphTaskSpec",
]


@dataclass
class ArenaEnvGraphSpec:
    """Typed representation of an environment graph YAML file.
    It defines the nodes, tasks, and state specs of the environment graph.
    """

    env_name: str
    nodes: list[ArenaEnvGraphNodeSpec] = field(default_factory=list)
    tasks: list[ArenaEnvGraphTaskSpec] = field(default_factory=list)
    state_specs: list[ArenaEnvGraphStateSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArenaEnvGraphSpec":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArenaEnvGraphSpec":
        data = as_dict(data, "Env graph spec")
        nodes = parse_list(data, "nodes", _parse_node)
        tasks = parse_list(data, "tasks", _parse_task)
        state_specs = parse_list(data, "state_specs", _parse_state_spec)

        spec = cls(
            env_name=required_str(data, "env_name"),
            nodes=nodes,
            tasks=tasks,
            state_specs=state_specs,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate graph-level ids, references, and relationship shapes."""
        assert_unique_ids(self.nodes, self.tasks, self.state_specs)
        assert_references_exist(self.nodes, self.tasks, self.state_specs)
        assert_spatial_constraint_shapes(self.state_specs)

    @property
    def nodes_by_id(self) -> dict[str, ArenaEnvGraphNodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def tasks_by_id(self) -> dict[str, ArenaEnvGraphTaskSpec]:
        return {task.id: task for task in self.tasks}

    @property
    def state_specs_by_id(self) -> dict[str, ArenaEnvGraphStateSpec]:
        return {state_spec.id: state_spec for state_spec in self.state_specs}

    def to_arena_env(self) -> "IsaacLabArenaEnvironment":
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
