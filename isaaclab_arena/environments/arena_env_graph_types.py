# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight schema for the env-graph spec.

Pydantic models with no parsing or conversion behavior. YAML loading, validation, and
conversion entry points live in ``arena_env_graph_spec``.

Lives in its own module so that conversion-utilities can import these names at module
level without creating a circular import back into ``arena_env_graph_spec`` (which itself
depends on the conversion module).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    NOT_NEXT_TO = "not_next_to"
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


class ArenaEnvGraphNodeSpec(BaseModel):
    """Node in an environment graph.

    Could be an object, an embodiment, a background, etc. Object references — USD prims
    inside a parent background asset — are represented by the
    :class:`ArenaEnvGraphObjectReferenceNodeSpec` subclass, which adds the extra fields
    needed to locate and type the referenced prim.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str  # Name registered in the asset registry
    type: ArenaEnvGraphNodeType
    # Asset-type specific optional kwargs (e.g. scale, spawn_cfg_addon) — distinct from
    # the typed graph metadata above. The Arena environment builder forwards these when
    # instantiating the asset class.
    params: dict[str, Any] = Field(default_factory=dict)


class ArenaEnvGraphObjectReferenceNodeSpec(ArenaEnvGraphNodeSpec):
    """Object-reference node: a USD prim inside a parent background asset.

    All three extra fields are required for this node type — without them the
    builder cannot bind to the referenced prim or know how to wrap it.
    """

    type: Literal[ArenaEnvGraphNodeType.OBJECT_REFERENCE] = ArenaEnvGraphNodeType.OBJECT_REFERENCE
    parent: str  # id of the parent (typically background) node that owns the prim
    prim_path: str  # USD prim path of the referenced prim (may contain {ENV_REGEX_NS})
    object_type: ObjectType  # how to wrap the prim (rigid, articulation, etc.)


class ArenaEnvGraphSpatialConstraintSpec(BaseModel):
    """Spatial constraint edge in an environment graph state spec.

    It defines a relation between two nodes.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    type: ArenaEnvGraphSpatialConstraintType
    parent: str
    child: str | None = None  # Optional, e.g. is_anchor constraint does not have a child
    # Type-specific optional kwargs for the underlying RelationBase subclass selected by `type`
    # (e.g. {x_min, x_max, y_min, y_max} for position_limits; {side, distance} for next_to etc.).
    # The Arena environment builder forwards these when constructing the Relation instance.
    params: dict[str, Any] = Field(default_factory=dict)


class ArenaEnvGraphTaskConstraintSpec(BaseModel):
    """Task-dependent constraint edge in an environment graph state spec."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: ArenaEnvGraphTaskConstraintType
    parent: str
    child: str | None = None  # Optional, could be a robot keeps gripper open or closed, or a single object
    # Type-specific optional kwargs for the underlying TaskConstraintBase subclass selected by `type`
    # (e.g. grasp pose offset the reach constraint.).
    # The Arena environment builder forwards these when constructing the TaskConstraint instance.
    params: dict[str, Any] = Field(default_factory=dict)


class ArenaEnvGraphStateSpec(BaseModel):
    """Snapshot of the environment state in the graph.

    Could be an initial, intermediate, or final state.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    spatial_constraints: list[ArenaEnvGraphSpatialConstraintSpec] = Field(default_factory=list)
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = Field(default_factory=list)


class ArenaEnvGraphTaskSpec(BaseModel):
    """Task entry in an environment graph."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str  # Task class name, could be a custom task class or a built-in task class
    initial_state_spec_id: str
    success_state_spec_id: str
    task_args: dict[str, Any] = Field(default_factory=dict)
