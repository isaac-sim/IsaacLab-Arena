# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.environments.utils import (
    as_dict,
    assert_env_graph_references_exist,
    assert_env_graph_universal_ids,
    optional_dict,
    optional_str,
    parse_list,
    required_enum,
    required_number_sequence,
    required_str,
)


class ArenaEnvGraphNodeType(Enum):
    EMBODIMENT = "embodiment"
    BACKGROUND = "background"
    OBJECT = "object"
    OBJECT_REFERENCE = "object_reference"
    LIGHTING = "lighting"


class ArenaEnvGraphSpatialConstraintType(Enum):
    IS_ANCHOR = "is_anchor"
    NEXT_TO = "next_to"
    ON = "on"
    AT_POSE = "at_pose"  # through set_initial_pose()
    AT_POSITION = "at_position"  # through object relation solver: AtPosition
    POSITION_LIMITS = "position_limits"
    RANDOM_AROUND_SOLUTION = "random_around_solution"
    ROTATE_AROUND_SOLUTION = "rotate_around_solution"
    # TODO(xinjieyao, 2026-05-21): Support "in" in solver
    IN = "in"


@dataclass
class ArenaEnvGraphNodeSpec:
    """Node in an environment graph.

    Could be an object, an embodiment, a background, etc. Object references — USD prims
    inside a parent background asset — are represented by the
    :class:`ArenaEnvGraphObjectReferenceNodeSpec` subclass, which adds the extra fields
    needed to locate and type the referenced prim.
    """

    id: str
    name: str  # Name registered in the asset registry
    type: ArenaEnvGraphNodeType
    # Asset-type specific optional kwargs (e.g. scale, spawn_cfg_addon) — distinct from
    # the typed graph metadata above. The Arena environment builder forwards these when
    # instantiating the asset class.
    params: dict[str, Any] = field(default_factory=dict)


# kw_only=True forces the three new fields to be keyword-only in __init__. Required because
# the base class ends with a defaulted field (`params`) and Python forbids non-default args
# from following default ones — placing the new required fields after `*` sidesteps that rule
# and lets us declare them as required (no default) instead of Optional with runtime checks.
@dataclass(kw_only=True)
class ArenaEnvGraphObjectReferenceNodeSpec(ArenaEnvGraphNodeSpec):
    """Object-reference node: a USD prim inside a parent background asset.

    All three extra fields are required for this node type — without them the
    builder cannot bind to the referenced prim or know how to wrap it.
    """

    parent: str  # id of the parent (typically background) node that owns the prim
    prim_path: str  # USD prim path of the referenced prim (may contain {ENV_REGEX_NS})
    object_type: ObjectType  # how to wrap the prim (rigid, articulation, etc.)


@dataclass
class ArenaEnvGraphSpatialConstraintSpec:
    """Spatial constraint edge in an environment graph state spec.

    It defines a relation between two nodes.
    """

    id: str
    type: ArenaEnvGraphSpatialConstraintType
    parent: str
    child: str | None = None  # Optional, e.g. is_anchor constraint does not have a child
    # Type-specific optional kwargs for the underlying RelationBase subclass selected by `type`
    # (e.g. {x_min, x_max, y_min, y_max} for position_limits; {side, distance} for next_to etc.).
    # The Arena environment builder forwards these when constructing the Relation instance.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArenaEnvGraphTaskConstraintSpec:
    """Task-dependent constraint edge in an environment graph state spec."""

    id: str
    type: str
    parent: str
    child: str | None = None  # Optional, could be a robot keeps gripper open or closed, or a single object
    # Type-specific optional kwargs for the underlying TaskConstraintBase subclass selected by `type`
    # (e.g. grasp pose offset the reach constraint.).
    # The Arena environment builder forwards these when constructing the TaskConstraint instance.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArenaEnvGraphStateSpec:
    """Snapshot of the environment state in the graph.

    Could be an initial, intermediate, or final state.
    """

    id: str
    name: str
    spatial_constraints: list[ArenaEnvGraphSpatialConstraintSpec] = field(default_factory=list)
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = field(default_factory=list)


@dataclass
class ArenaEnvGraphTaskSpec:
    """Task entry in an environment graph."""

    id: str
    name: str
    type: str  # Task class name, could be a custom task class or a built-in task class
    initial_state_spec_id: str
    success_state_spec_id: str
    task_args: dict[str, Any] = field(default_factory=dict)


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

        assert_env_graph_universal_ids(nodes, tasks, state_specs)
        assert_env_graph_references_exist(nodes, tasks, state_specs)

        return cls(
            env_name=required_str(data, "env_name"),
            nodes=nodes,
            tasks=tasks,
            state_specs=state_specs,
        )

    @property
    def nodes_by_id(self) -> dict[str, ArenaEnvGraphNodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def tasks_by_id(self) -> dict[str, ArenaEnvGraphTaskSpec]:
        return {task.id: task for task in self.tasks}

    @property
    def state_specs_by_id(self) -> dict[str, ArenaEnvGraphStateSpec]:
        return {state_spec.id: state_spec for state_spec in self.state_specs}


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
    if constraint_type == ArenaEnvGraphSpatialConstraintType.AT_POSE:
        params["position_xyz"] = required_number_sequence(params, "position_xyz", 3)
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
        type=required_str(data, "type"),
        parent=optional_str(data, "parent"),
        child=optional_str(data, "child"),
        params=optional_dict(data, "params"),
    )


def _parse_state_spec(data: Any) -> ArenaEnvGraphStateSpec:
    data = as_dict(data, "State spec")
    assert "edges" not in data, "State spec must define spatial_constraints and task_constraints directly"
    return ArenaEnvGraphStateSpec(
        id=required_str(data, "id"),
        name=required_str(data, "name"),
        spatial_constraints=parse_list(data, "spatial_constraints", _parse_spatial_constraint),
        task_constraints=parse_list(data, "task_constraints", _parse_task_constraint),
    )


def _parse_task(data: Any) -> ArenaEnvGraphTaskSpec:
    data = as_dict(data, "Task spec")
    for old_key in ("state_specs", "initial_state_spec", "success_state_spec"):
        assert old_key not in data, "Task spec must use initial_state_spec_id and success_state_spec_id"
    return ArenaEnvGraphTaskSpec(
        id=required_str(data, "id"),
        name=required_str(data, "name"),
        type=required_str(data, "type"),
        initial_state_spec_id=required_str(data, "initial_state_spec_id"),
        success_state_spec_id=required_str(data, "success_state_spec_id"),
        task_args=optional_dict(data, "task_args"),
    )
