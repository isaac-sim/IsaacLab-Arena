# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for :class:`~isaaclab_arena.environments.arena_env_graph_spec.ArenaEnvGraphSpec`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.environments.graph_spec_utils import coerce_number_sequence


class AssetSpec(BaseModel):
    """One registered asset instance in an environment graph."""

    id: str = Field(
        min_length=1,
        description=(
            "Unique id for this asset instance. Use underscore-connected identifiers "
            "(e.g. 'banana_1', 'maple_table_robolab'). Referenced by relations and task params."
        ),
    )
    registry_name: str = Field(
        min_length=1,
        description="Exact registered asset name from EMBODIMENTS / BACKGROUNDS / OBJECTS.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional constructor kwargs forwarded to the asset class.",
    )

    @field_validator("registry_name")
    @classmethod
    def _validate_registry_name(cls, value: str) -> str:
        registry = AssetRegistry()
        assert registry.is_registered(value), f"Unknown asset registry_name '{value}'"
        return value


class ObjectReferenceSpec(BaseModel):
    """USD prim reference inside a parent background asset."""

    id: str = Field(min_length=1, description="Unique node id referenced by relations and task params.")
    parent_id: str = Field(min_length=1, description="Id of the parent background asset node.")
    prim_path: str | None = Field(
        default=None,
        description="USD prim path inside the parent background (required for conversion).",
    )
    object_type: ObjectType = Field(description="Physics type for the referenced prim.")
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_prim_path(self) -> ObjectReferenceSpec:
        assert self.prim_path is not None and self.prim_path.strip(), f"Object reference '{self.id}' requires prim_path"
        return self


class TaskSpec(BaseModel):
    """Task entry in an environment graph."""

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
            "Constructor kwargs for the task (listed in TASKS). Each object param must "
            "name exactly one asset or object-reference node id. Scene params use the "
            "background node id."
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


class SpatialRelationSpec(BaseModel):
    """Spatial relation in an environment graph."""

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
            "Node id this relation applies to. For binary relations (e.g. 'on'), it's the "
            "object placed relative to ``reference``. For unary relations (e.g. "
            "'is_anchor', 'position_limits'), it's the anchored or constrained object."
        ),
    )
    reference: str | None = Field(
        default=None,
        description=(
            "Reference node id for binary relations only — e.g. for 'on', the surface "
            "the subject rests on. Must be null for unary relations."
        ),
    )
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
        self.params = _normalize_relation_params(self.params)
        return self


def _normalize_relation_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if "position_xyz" in normalized:
        normalized["position_xyz"] = coerce_number_sequence(normalized["position_xyz"], 3, "position_xyz")
    if "rotation_xyzw" in normalized:
        normalized["rotation_xyzw"] = coerce_number_sequence(normalized["rotation_xyzw"], 4, "rotation_xyzw")
    return normalized


class CliOverrideSpec(BaseModel):
    """One CLI flag that swaps an asset's registry name, declared in the graph YAML."""

    arg: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)

    @property
    def dest(self) -> str:
        """The argparse attribute name for this flag (dashes become underscores)."""
        return self.arg.replace("-", "_")
