# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for :class:`~isaaclab_arena.environment_spec.arena_env_graph_spec.ArenaEnvGraphSpec`."""

from __future__ import annotations

from enum import Enum
from numbers import Real
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME


def _extract_asset_usd_path(asset_cls: type, **params: Any) -> str | None:
    """Return the asset's root USD path or URL, or ``None`` if not extractable."""
    class_usd = getattr(asset_cls, "usd_path", None)
    if isinstance(class_usd, str) and class_usd:
        return class_usd

    # Instantiate when usd_path is set lazily (e.g. Lightwheel backgrounds).
    # TODO(qianl): add support for embodiments, whose robot USD lives in scene_config.robot.spawn.
    try:
        instance = asset_cls(**params)
    except Exception:
        return None

    usd_path = getattr(instance, "usd_path", None)
    return str(usd_path) if usd_path else None


def _assert_registered_asset_name(registry_name: str, object_type: ObjectType | None = None) -> str:
    """Return ``registry_name`` after checking it names a registered asset of ``object_type``(if provided).
    Args:
        registry_name: Registered asset name to check.
        object_type: Object type the asset must declare. ``None`` accepts any asset.
    """
    assert AssetRegistry().is_registered(registry_name), f"Unknown asset registry_name '{registry_name}'"
    if object_type is not None:
        declared_type = getattr(AssetRegistry().get_asset_by_name(registry_name), "object_type", None)
        assert (
            declared_type is object_type
        ), f"Asset '{registry_name}' must be a {object_type.value} object, got {declared_type}"
    return registry_name


class AssetSpec(BaseModel):
    """One registered asset instance in an environment graph."""

    id: str = Field(
        min_length=1,
        description=(
            "Unique id for this asset instance. Use underscore-connected identifiers "
            "(e.g. 'banana', 'maple_table'). Referenced by relations and task params."
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
        return _assert_registered_asset_name(value)

    @field_validator("params")
    @classmethod
    def _drop_catalogue_tags(cls, value: dict[str, Any]) -> dict[str, Any]:
        # The asset catalogue prints 'tags=[...]', and a generated spec often copies them into params.
        # Asset classes pass their own tags to Asset.__init__, so a copy here is a duplicate keyword argument.
        if "tags" not in value:
            return value
        print("INFO: ignoring 'tags' in asset params; the asset class supplies its own tags.")
        return {key: item for key, item in value.items() if key != "tags"}

    def resolve_usd_path(self) -> str:
        """Return the USD path or URL for this registered asset instance."""
        asset_cls = AssetRegistry().get_asset_by_name(self.registry_name)
        usd_path = _extract_asset_usd_path(asset_cls, **self.params)
        assert usd_path, f"asset {self.registry_name!r} has no usd_path"
        return usd_path


class ObjectSetSpec(BaseModel):
    """A set of rigid objects distributed among parallel environments, one object per environment."""

    id: str = Field(
        min_length=1,
        description=(
            "Unique id for this object set (e.g. 'bottles'). Referenced by relations and task "
            "params exactly like an object id."
        ),
    )
    members: list[str] = Field(
        min_length=1,
        description=(
            "Exact registered object names from OBJECTS marked type=rigid that this set draws "
            "from; every environment spawns one of them."
        ),
    )
    random_choice: bool = Field(
        default=False,
        description=(
            "Sample each environment's member independently at random. When false, members are "
            "assigned by repeating their declared order across environments."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional constructor kwargs forwarded to RigidObjectSet; leave empty by default.",
    )

    # TODO(xinjieyao, 2026-08-03): Support searched SimReady assets as object set members.
    @field_validator("members")
    @classmethod
    def _validate_member_registry_names(cls, value: list[str]) -> list[str]:
        for registry_name in value:
            # ObjectSet members are built with no arguments, but Simready assets need usd_path. Only an object's params can carry.
            assert registry_name != SIMREADY_USD_OBJECT_REGISTRY_NAME, (
                f"'{registry_name}' cannot be an object set member, because a member has nowhere to"
                " carry the usd_path it needs. Use it as an object instead."
            )
        return [_assert_registered_asset_name(registry_name, ObjectType.RIGID) for registry_name in value]

    @field_validator("params")
    @classmethod
    def _reject_reserved_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        # These are forwarded from the fields above, so a duplicate here would be a TypeError at
        # build time, and an 'objects' override would skip the rigid-member check on members.
        reserved = sorted({"name", "objects", "random_choice"} & set(value))
        assert not reserved, f"params must not set {reserved}; use the id, members, random_choice fields instead"
        return value


class ObjectReferenceSpec(BaseModel):
    """USD prim reference inside a parent background asset."""

    id: str = Field(min_length=1, description="Unique node id referenced by relations and task params.")
    parent_id: str = Field(min_length=1, description="Id of the parent background asset node.")
    prim_path: str | None = Field(
        default=None,
        description="USD prim path inside the parent background; leave empty until resolved.",
    )
    object_type: ObjectType = Field(
        description=(
            "Physics type for the referenced prim. Use the first matching value:\n"
            "- articulation: door or other articulated prim in open/close door tasks\n"
            "- rigid: manipulable prim in pick-and-place tasks\n"
            "- base: static anchor prim (e.g. table surface) in is_anchor or placement relations"
        ),
    )
    params: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    """Atomic registered task leaf referenced by a composite root task."""

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
            "name exactly one asset or object-reference node id."
        ),
    )

    @field_validator("kind")
    @classmethod
    def _validate_registered_task_type(cls, value: str) -> str:
        assert TaskRegistry().is_registered(value), f"Unknown task kind '{value}'"
        return value


class TaskCompositionType(str, Enum):
    """How atomic subtasks combine in a composite root task."""

    ATOMIC = "atomic"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class CompositeTaskSpec(BaseModel):
    """Root task node for an environment graph."""

    composition: TaskCompositionType = Field(
        description="How the subtasks combine: " + ", ".join([f"'{e.value}'" for e in TaskCompositionType])
    )
    description: str = Field(
        min_length=1,
        description="Natural-language summary of the overall task (e.g. 'pick and place all bananas into the bin').",
    )
    subtasks: list[TaskSpec] = Field(
        default_factory=list,
        description="Atomic registered tasks that compose this root task.",
    )

    @model_validator(mode="after")
    def _validate_composition_task_count(self) -> CompositeTaskSpec:
        if self.composition is TaskCompositionType.ATOMIC:
            assert len(self.subtasks) == 1, "composition 'atomic' requires exactly one atomic task"
        else:
            assert (
                len(self.subtasks) >= 2
            ), f"composition '{self.composition.value}' requires at least two atomic tasks, got {len(self.subtasks)}"
        return self


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
            "'is_anchor', 'position_limits_box'), it's the anchored or constrained object."
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
        normalized["position_xyz"] = _convert_to_float_tuple(normalized["position_xyz"], 3, "position_xyz")
    if "rotation_xyzw" in normalized:
        normalized["rotation_xyzw"] = _convert_to_float_tuple(normalized["rotation_xyzw"], 4, "rotation_xyzw")
    return normalized


def _convert_to_float_tuple(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    """Coerce a fixed-length numeric list or tuple (e.g. position or quaternion)."""
    assert isinstance(value, (list, tuple)), f"Field '{field_name}' must be a list or tuple of {length} numbers"
    assert len(value) == length, f"Field '{field_name}' must contain exactly {length} numbers, got {len(value)}"
    assert all(
        isinstance(item, Real) and not isinstance(item, bool) for item in value
    ), f"Field '{field_name}' must contain only numbers"
    return tuple(float(item) for item in value)


class PlacementValidatorSpec(BaseModel):
    """Per-env placement validators.

    Selects which build-time geometric checks gate object placement for this env. Defaults to
    every build-time check.
    """

    enabled_checks: list[str] | None = Field(
        default=None,
        description=(
            "Build-time check names to evaluate during placement; none runs every registered build-time "
            "check. A check not listed here is never run. Built-in names: no_overlap, on_relation, "
            "next_to, not_next_to, face_to; externally-registered validators may add more."
        ),
    )
    required_checks: list[str] | None = Field(
        default=None,
        description=(
            "Enabled checks that must pass for a layout to be valid; none requires every enabled check. "
            "Must be a subset of enabled_checks."
        ),
    )

    @model_validator(mode="after")
    def _validate_required_subset(self) -> PlacementValidatorSpec:
        if self.enabled_checks is not None and self.required_checks is not None:
            extra = set(self.required_checks) - set(self.enabled_checks)
            assert not extra, f"required_checks must be a subset of enabled_checks; unexpected: {sorted(extra)}"
        return self


class CliOverrideSpec(BaseModel):
    """One CLI flag that swaps an asset's registry name, declared in the graph YAML."""

    arg: str = Field(min_length=1)  # flag name without leading dashes; "object" -> --object
    target_node_id: str = Field(min_length=1)  # graph asset id whose registry_name the flag swaps

    @property
    def dest(self) -> str:
        """The argparse attribute name for this flag (dashes become underscores)."""
        return self.arg.replace("-", "_")
