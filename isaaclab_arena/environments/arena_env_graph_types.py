# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight schema for the env-graph spec.

Pure data: enums and dataclasses with no parsing or conversion behavior. Behavior (YAML
loading, validation, conversion entry point) lives in ``arena_env_graph_spec``.

Lives in its own module so that conversion-utilities can import these names at module
level without creating a circular import back into ``arena_env_graph_spec`` (which itself
depends on the conversion module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from isaaclab_arena.assets.object_type import ObjectType


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


class ArenaEnvGraphTaskConstraintType(Enum):
    REACH = "reach"


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
    type: ArenaEnvGraphTaskConstraintType
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
    spatial_constraints: list[ArenaEnvGraphSpatialConstraintSpec] = field(default_factory=list)
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = field(default_factory=list)


@dataclass
class ArenaEnvGraphTaskSpec:
    """Task entry in an environment graph."""

    id: str
    type: str  # Task class name, could be a custom task class or a built-in task class
    initial_state_spec_id: str
    success_state_spec_id: str
    task_args: dict[str, Any] = field(default_factory=dict)
