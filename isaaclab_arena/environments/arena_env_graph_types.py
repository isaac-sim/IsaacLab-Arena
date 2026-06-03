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

from pydantic import BaseModel, Field, field_validator, model_validator

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.environments.graph_spec_utils import coerce_number_sequence


class ArenaEnvGraphNodeType(Enum):
    EMBODIMENT = "embodiment"
    BACKGROUND = "background"
    OBJECT = "object"
    OBJECT_REFERENCE = "object_reference"
    LIGHTING = "lighting"


# ``parse_graph_node`` runs before per-field coercion, so ``type`` may still be the
# YAML string ``"object_reference"`` or already an ``ArenaEnvGraphNodeType`` member.
_OBJECT_REFERENCE_VALUES = frozenset({
    ArenaEnvGraphNodeType.OBJECT_REFERENCE,
    ArenaEnvGraphNodeType.OBJECT_REFERENCE.value,
})


def _is_object_reference_type(node_type: Any) -> bool:
    return node_type in _OBJECT_REFERENCE_VALUES


def parse_graph_node(data: Any) -> Any:
    """Select the node spec class for ``data`` and validate ``object_reference`` nodes."""
    if isinstance(data, ArenaEnvGraphNodeSpec):
        return data
    if isinstance(data, dict) and _is_object_reference_type(data.get("type")):
        return ArenaEnvGraphObjectReferenceNodeSpec.model_validate(data)
    return data


# TODO(qianl): remove this enum and check against relation registry for task constraints
class ArenaEnvGraphTaskConstraintType(Enum):
    REACH = "reach"


class ArenaEnvGraphNodeSpec(BaseModel):
    """Node in an environment graph (all types except ``object_reference``)."""

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

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    parent: str = Field(min_length=1)
    child: str | None = None  # Optional, e.g. is_anchor constraint does not have a child
    # Type-specific optional kwargs for the underlying RelationBase subclass selected by `type`
    # (e.g. {x_min, x_max, y_min, y_max} for position_limits; {side, distance} for next_to etc.).
    # The Arena environment builder forwards these when constructing the Relation instance.
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_registered_relation_type(cls, value: str) -> str:
        registry = ObjectRelationLibraryRegistry()
        if not registry.is_registered(value):
            valid_values = sorted(registry.get_all_keys())
            raise ValueError(f"Unknown spatial constraint type '{value}'. Expected one of {valid_values}")
        return value

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

    id: str = Field(min_length=1)
    type: ArenaEnvGraphTaskConstraintType
    parent: str = Field(min_length=1)
    child: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ArenaEnvGraphStateSpec(BaseModel):
    """Snapshot of the environment state in the graph."""

    id: str = Field(min_length=1)
    spatial_constraints: list[ArenaEnvGraphSpatialConstraintSpec] = Field(default_factory=list)
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = Field(default_factory=list)


class ArenaEnvGraphTaskSpec(BaseModel):
    """Task entry in an environment graph."""

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    initial_state_spec_id: str = Field(min_length=1)
    success_state_spec_id: str = Field(min_length=1)
    task_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_registered_task_type(cls, value: str) -> str:
        registry = TaskRegistry()
        if not registry.is_registered(value):
            valid_values = sorted(registry.get_all_keys())
            raise ValueError(f"Unknown task type '{value}'. Expected one of {valid_values}")
        return value
