# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for the env-graph spec.

Graph-level validation (unique ids, cross-references, relation arity) runs on
:class:`~isaaclab_arena.environments.arena_env_graph_spec.ArenaEnvGraphSpec` via
``model_validator``. Lives in its own module so conversion utilities can import
these names without a circular import through ``arena_env_graph_spec``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.graph_spec_utils import coerce_number_sequence


class ArenaEnvGraphNodeType(Enum):
    EMBODIMENT = "embodiment"
    BACKGROUND = "background"
    OBJECT = "object"
    OBJECT_REFERENCE = "object_reference"
    LIGHTING = "lighting"


_OBJECT_REFERENCE_VALUES = frozenset({
    ArenaEnvGraphNodeType.OBJECT_REFERENCE,
    ArenaEnvGraphNodeType.OBJECT_REFERENCE.value,
})


def _is_object_reference_type(node_type: Any) -> bool:
    return node_type in _OBJECT_REFERENCE_VALUES


def _coerce_graph_node(data: Any) -> Any:
    """Route ``object_reference`` dicts to :class:`ArenaEnvGraphObjectReferenceNodeSpec`."""
    if isinstance(data, ArenaEnvGraphNodeSpec):
        return data
    if isinstance(data, dict) and _is_object_reference_type(data.get("type")):
        return ArenaEnvGraphObjectReferenceNodeSpec.model_validate(data)
    return data


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
    """Node in an environment graph (all types except ``object_reference``)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: ArenaEnvGraphNodeType
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_object_reference_without_extra_fields(self) -> ArenaEnvGraphNodeSpec:
        if type(self) is ArenaEnvGraphNodeSpec and self.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            raise ValueError("object_reference nodes require parent, prim_path, and object_type fields")
        return self


class ArenaEnvGraphObjectReferenceNodeSpec(ArenaEnvGraphNodeSpec):
    """Object-reference node: a USD prim inside a parent background asset."""

    type: ArenaEnvGraphNodeType = ArenaEnvGraphNodeType.OBJECT_REFERENCE
    parent: str = Field(min_length=1)
    prim_path: str = Field(min_length=1)
    object_type: ObjectType

    @model_validator(mode="after")
    def _must_be_object_reference(self) -> ArenaEnvGraphObjectReferenceNodeSpec:
        assert self.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE, "internal invariant"
        return self


class ArenaEnvGraphSpatialConstraintSpec(BaseModel):
    """Spatial constraint edge in an environment graph state spec."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: ArenaEnvGraphSpatialConstraintType
    parent: str = Field(min_length=1)
    child: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="after")
    @classmethod
    def _normalize_pose_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(params)
        if "position_xyz" in normalized:
            normalized["position_xyz"] = coerce_number_sequence(normalized["position_xyz"], 3, "position_xyz")
        if "rotation_xyzw" in normalized:
            normalized["rotation_xyzw"] = coerce_number_sequence(normalized["rotation_xyzw"], 4, "rotation_xyzw")
        return normalized


class ArenaEnvGraphTaskConstraintSpec(BaseModel):
    """Task-dependent constraint edge in an environment graph state spec."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: ArenaEnvGraphTaskConstraintType
    parent: str = Field(min_length=1)
    child: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ArenaEnvGraphStateSpec(BaseModel):
    """Snapshot of the environment state in the graph."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    spatial_constraints: list[ArenaEnvGraphSpatialConstraintSpec] = Field(default_factory=list)
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_edges_wrapper(cls, data: Any) -> Any:
        if isinstance(data, dict) and "edges" in data:
            raise ValueError("State spec must define spatial_constraints and task_constraints directly")
        return data


class ArenaEnvGraphTaskSpec(BaseModel):
    """Task entry in an environment graph."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    initial_state_spec_id: str = Field(min_length=1)
    success_state_spec_id: str = Field(min_length=1)
    task_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_state_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for old_key in ("state_specs", "initial_state_spec", "success_state_spec"):
                if old_key in data:
                    raise ValueError("Task spec must use initial_state_spec_id and success_state_spec_id")
        return data
