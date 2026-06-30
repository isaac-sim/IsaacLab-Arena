# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schemas and validation for inferred background object references."""

from __future__ import annotations

import inspect

from pydantic import BaseModel, Field, model_validator

from isaaclab_arena.agentic_environment_generation.background_physics_catalog import PhysicsPrimEntry
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import (
    EnvironmentIntentSpec,
    ObjectReferenceItem,
)
from isaaclab_arena.assets.registries import TaskRegistry


class TaskParamBinding(BaseModel):
    """Bind one task constructor param to a background object-reference node id."""

    task_index: int = Field(ge=0)
    param_name: str = Field(min_length=1)
    reference_id: str = Field(min_length=1)


class BackgroundObjectReferenceInferenceSpec(BaseModel):
    """Background subprims and task-param rewrites inferred for an intent spec."""

    reasoning: str = Field(description="Explanation of the chosen references and bindings.")
    object_references: list[ObjectReferenceItem] = Field(default_factory=list)
    task_param_bindings: list[TaskParamBinding] = Field(default_factory=list)
    remove_item_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bindings_reference_known_ids(self) -> BackgroundObjectReferenceInferenceSpec:
        ref_ids = {ref.id for ref in self.object_references}
        assert len(ref_ids) == len(self.object_references), "object_reference ids must be unique"
        for binding in self.task_param_bindings:
            assert (
                binding.reference_id in ref_ids
            ), f"task_param_bindings reference unknown id {binding.reference_id!r}; known ids: {sorted(ref_ids)}"
        return self


def validate_background_object_reference_inference(
    intent: EnvironmentIntentSpec,
    inference: BackgroundObjectReferenceInferenceSpec,
    physics_entries: list[PhysicsPrimEntry],
) -> None:
    """Validate inferred refs against intent tasks and the background physics catalog."""
    entries_by_path = {entry.usd_prim_path: entry for entry in physics_entries}
    refs_by_id = {ref.id: ref for ref in inference.object_references}

    for ref in inference.object_references:
        assert ref.usd_prim_path in entries_by_path, f"unknown physics prim path {ref.usd_prim_path!r}"
        entry = entries_by_path[ref.usd_prim_path]
        if ref.object_type == "rigid":
            assert "rigid_body" in entry.physics_kinds, f"{ref.id!r} must reference a rigid_body prim"
            assert ref.openable_joint_name is None, f"{ref.id!r} rigid references must not set openable_joint_name"
        else:
            assert "articulation" in entry.physics_kinds, f"{ref.id!r} must reference an articulation prim"
            if ref.openable_joint_name is not None:
                assert ref.openable_joint_name in entry.revolute_joint_names, (
                    f"{ref.id!r} openable_joint_name {ref.openable_joint_name!r} is not in "
                    f"{sorted(entry.revolute_joint_names)}"
                )

    task_registry = TaskRegistry()
    for binding in inference.task_param_bindings:
        assert binding.task_index < len(intent.tasks), f"task_index {binding.task_index} is out of range"
        task = intent.tasks[binding.task_index]
        task_cls = task_registry.get_task_by_name(task.kind)
        assert _task_accepts_param(
            task_cls, binding.param_name
        ), f"{task.kind}.{binding.param_name} is not a constructor parameter"
        ref = refs_by_id[binding.reference_id]
        _validate_binding_role(task.kind, binding.param_name, ref)


def _task_accepts_param(task_cls: type, param_name: str) -> bool:
    sig = inspect.signature(task_cls.__init__)
    return param_name in sig.parameters


def _validate_binding_role(task_kind: str, param_name: str, ref: ObjectReferenceItem) -> None:
    if task_kind in {"OpenDoorTask", "CloseDoorTask"} and param_name == "openable_object":
        assert ref.object_type == "articulation", f"{task_kind}.openable_object must bind to an articulation"
        assert ref.openable_joint_name, f"{task_kind}.openable_object requires openable_joint_name"
    if task_kind == "PickAndPlaceTask" and param_name in {"destination_object", "destination_location"}:
        assert ref.object_type == "rigid", f"PickAndPlaceTask.{param_name} must bind to a rigid reference"
