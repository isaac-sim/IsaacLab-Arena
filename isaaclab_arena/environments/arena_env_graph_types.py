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

# =============================================================================
# Nodes
# =============================================================================


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


class ArenaEnvGraphNodeSpec(BaseModel):
    """Node in an environment graph (all types except ``object_reference``)."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: ArenaEnvGraphNodeType
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_object_reference_without_extra_fields(self) -> ArenaEnvGraphNodeSpec:
        assert not (
            type(self) is ArenaEnvGraphNodeSpec and self.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE
        ), "object_reference nodes require parent, prim_path, and object_type fields"
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


# =============================================================================
# Tasks
# =============================================================================


class TaskSpec(BaseModel):
    """Task shared by agent intent specs and env-graph task entries."""

    kind: str = Field(
        min_length=1,
        description=(
            "Registered task class name from the TASKS block in the user message "
            "(e.g. 'PickAndPlaceTask', 'OpenDoorTask'). Must match TaskRegistry exactly."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Constructor kwargs for the task (listed in TASKS). "
            "Values are Item.query strings or the background name for scene parameters."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Natural-language summary of the task (e.g. 'pick up the avocado and place it in the bowl'). ",
    )

    @field_validator("kind")
    @classmethod
    def _validate_registered_task_type(cls, value: str) -> str:
        registry = TaskRegistry()
        assert registry.is_registered(value), f"Unknown task kind '{value}'"
        return value


class ArenaEnvGraphTaskSpec(TaskSpec):
    """Task entry in an environment graph (task payload plus state-spec wiring)."""

    id: str = Field(min_length=1)
    initial_state_spec_id: str = Field(min_length=1)
    success_state_spec_id: str = Field(min_length=1)


# =============================================================================
# Constraints and state specs
# =============================================================================


class SpatialRelationSpec(BaseModel):
    """Spatial relation shared by agent intent specs and env-graph spatial constraints."""

    kind: str = Field(
        min_length=1,
        description=(
            "Relation name from the RELATIONS block in the user message "
            "(e.g. 'on', 'next_to', 'is_anchor'). Must match a registered relation exactly."
        ),
    )
    subject: str = Field(
        min_length=1,
        description=(
            "Object (Item.query) this relation applies to. For binary relations "
            "(e.g. 'on'), this is the object placed relative to ``reference``. "
            "For unary relations (e.g. 'is_anchor', 'position_limits'), this is "
            "the anchored or constrained object."
        ),
    )
    reference: str | None = Field(
        default=None,
        description=(
            "Reference (Item.query or background name) for binary relations only "
            "— e.g. for 'on', the surface the subject rests on. Must be null "
            "for unary relations."
        ),
    )
    # TODO(qianl): free-form ``dict`` emits ``additionalProperties: true``,
    # which strict-mode structured-outputs endpoints reject with a 400.
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional kind-specific parameters; leave empty by default.",
    )

    @model_validator(mode="after")
    def _validate_kind_and_arity(self) -> SpatialRelationSpec:
        registry = ObjectRelationLibraryRegistry()
        assert registry.is_registered(self.kind), f"Unknown relation kind '{self.kind}'"
        relation_cls = registry.get_object_relation_by_name(self.kind)
        if relation_cls.is_unary():
            assert self.reference is None, f"Relation kind '{self.kind}' must not define relation.reference"
        else:
            assert self.reference is not None, f"Relation kind '{self.kind}' requires relation.reference"
        return self


def _normalize_relation_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if "position_xyz" in normalized:
        normalized["position_xyz"] = coerce_number_sequence(normalized["position_xyz"], 3, "position_xyz")
    if "rotation_xyzw" in normalized:
        normalized["rotation_xyzw"] = coerce_number_sequence(normalized["rotation_xyzw"], 4, "rotation_xyzw")
    return normalized


class ArenaEnvGraphSpatialRelationSpec(SpatialRelationSpec):
    """Spatial constraint edge in an environment graph state spec (relation plus constraint id)."""

    id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _normalize_relation_params(self) -> ArenaEnvGraphSpatialRelationSpec:
        self.params = _normalize_relation_params(self.params)
        return self


# TODO(qianl): remove this enum and check against relation registry for task constraints
class ArenaEnvGraphTaskConstraintType(Enum):
    REACH = "reach"


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
    spatial_constraints: list[ArenaEnvGraphSpatialRelationSpec] = Field(default_factory=list)
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = Field(default_factory=list)


# =============================================================================
# CLI overrides
# =============================================================================


class ArenaEnvGraphCliOverrideSpec(BaseModel):
    """One CLI flag that swaps a graph node's asset, declared in the graph YAML.

    Lets one graph YAML serve many variants without editing it.
    """

    arg: str = Field(min_length=1)  # flag name without leading dashes; "object" -> --object
    target_node_id: str = Field(min_length=1)  # whose `name` the flag overrides

    @property
    def dest(self) -> str:
        """The argparse attribute name for this flag (dashes become underscores)."""
        return self.arg.replace("-", "_")
