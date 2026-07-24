# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from isaaclab_arena.environment_spec.arena_env_graph_types import (
    AssetSpec,
    CliOverrideSpec,
    CompositeTaskSpec,
    ObjectReferenceSpec,
    PlacementValidatorSpec,
    SpatialRelationSpec,
    TaskConstraintSpec,
    TaskConstraintType,
    TaskSpec,
)
from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


class ArenaEnvGraphSpec(BaseModel):
    """Environment graph spec — the single source of truth for scene layout and tasks."""

    env_name: str = Field(min_length=1, description="Short snake_case label summarizing the scene and tasks.")
    embodiment: AssetSpec = Field(description="The robot that performs the tasks.")
    background: AssetSpec = Field(description="The static scene the robot and objects sit in.")
    objects: list[AssetSpec] = Field(default_factory=list, description="Movable scene objects, including distractors.")
    object_references: list[ObjectReferenceSpec] | None = Field(
        default=None, description="Optional prims inside the background exposed as assets (e.g. a table surface)."
    )
    relations: list[SpatialRelationSpec] = Field(
        default_factory=list, description="Spatial layout relations across all assets."
    )
    placement_validators: PlacementValidatorSpec | None = Field(
        default=None,
        description="Per-env placement validators; none runs all build-time checks.",
    )
    task: CompositeTaskSpec = Field(description="Root task the robot performs to manipulate the objects.")
    task_constraints: list[TaskConstraintSpec] | None = Field(
        default=None,
        description="Optional per-object task constraints.",
    )
    cli_override_specs: list[CliOverrideSpec] | None = Field(
        default=None, description="Optional authoring-time CLI flags that swap an asset's registry_name; usually empty."
    )

    @field_validator("object_references", "cli_override_specs", "task_constraints", mode="before")
    @classmethod
    def _none_if_empty_list(cls, value: Any) -> Any:
        if value == []:
            return None
        return value

    @model_validator(mode="after")
    def validate(self) -> Self:
        """Check unique asset ids, cross-references, task params, and CLI overrides."""
        known_ids = self._assert_asset_ids_unique()
        if self.object_references:
            valid_parent_ids = {self.background.id, *(obj.id for obj in self.objects)}
            self._assert_object_reference_parents(self.object_references, valid_parent_ids)
        self._assert_relation_references(self.relations, known_ids)
        self._assert_task_param_references(self.task.subtasks, known_ids)
        # Reachability targets must be placed movable objects; object references are static scene parts the
        # IK gate never checks, so they are not valid subjects.
        self._assert_task_constraint_subject_is_valid(self.task_constraints or [], self._placed_object_ids())
        self._validate_cli_override_specs()
        return self

    def _placed_object_ids(self) -> set[str]:
        """Ids of placed movable objects -- the only valid reachability subjects (excludes references and scene)."""
        return {obj.id for obj in self.objects}

    @staticmethod
    def _assert_task_constraint_subject_is_valid(constraints: list[TaskConstraintSpec], object_ids: set[str]) -> None:
        """Ensure each task constraint's subject names a placed object id."""
        for constraint in constraints:
            assert (
                constraint.subject in object_ids
            ), f"task_constraint '{constraint.type.value}' references unknown subject '{constraint.subject}'"

    def get_reachability_target_object_ids(self) -> tuple[str, ...]:
        """Object ids the robot shall be able to reach."""
        object_ids = self._placed_object_ids()
        # dict.fromkeys de-duplicates repeated object ids.
        # It also preserves order esp for composite tasks for consistency, or if the user adds explicit 'reachable' constraints.
        return tuple(
            dict.fromkeys((
                *(c.subject for c in (self.task_constraints or []) if c.type is TaskConstraintType.REACHABLE),
                *self._select_task_declared_reachable_subjects(object_ids),
            ))
        )

    def _select_task_declared_reachable_subjects(self, object_ids: set[str]) -> list[str]:
        """Each subtask declares candidates for reachability targets via its task class's reachability_target_objects."""
        # Lazy import: reading task-class metadata pulls the task library.
        from isaaclab_arena.assets.registries import TaskRegistry

        registry = TaskRegistry()
        subjects: list[str] = []
        for task in self.task.subtasks:
            task_cls = registry.get_task_by_name(task.kind)
            # If the task class declares reachability targets, add them to the list.
            if task_cls.reachability_target_objects:
                for param in task_cls.reachability_target_objects:
                    value = task.params.get(param)
                    if isinstance(value, str) and value in object_ids:
                        subjects.append(value)
        return subjects

    def _assert_asset_ids_unique(self) -> set[str]:
        """Ensure every asset and object-reference id in this spec is unique, returning the id set."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for asset_id in (
            self.embodiment.id,
            self.background.id,
            *(obj.id for obj in self.objects),
            *(ref.id for ref in (self.object_references or [])),
        ):
            if asset_id in seen:
                duplicates.add(asset_id)
            seen.add(asset_id)
        assert not duplicates, f"Duplicate graph asset ids found: {sorted(duplicates)}"
        return seen

    @staticmethod
    def _assert_object_reference_parents(
        object_references: list[ObjectReferenceSpec], valid_parent_ids: set[str]
    ) -> None:
        """Ensure each object reference's parent is a background or object asset."""
        for ref in object_references:
            assert ref.parent_id in valid_parent_ids, (
                f"Object reference '{ref.id}' references invalid parent '{ref.parent_id}'; "
                "parent must be the background or an object id"
            )

    @staticmethod
    def _assert_relation_references(relations: list[SpatialRelationSpec], known_ids: set[str]) -> None:
        """Ensure relation subject/reference endpoints name known asset ids."""
        for index, relation in enumerate(relations):
            assert (
                relation.subject in known_ids
            ), f"Relation[{index}] kind '{relation.kind}' references unknown subject '{relation.subject}'"
            if relation.reference is not None:
                assert (
                    relation.reference in known_ids
                ), f"Relation[{index}] kind '{relation.kind}' references unknown reference '{relation.reference}'"

    @staticmethod
    def _assert_task_param_references(subtasks: list[TaskSpec], known_ids: set[str]) -> None:
        """Ensure string-valued task params reference known asset ids."""
        for task in subtasks:
            for param_name, param_value in task.params.items():
                if isinstance(param_value, str):
                    assert (
                        param_value in known_ids
                    ), f"Task '{task.kind}' param '{param_name}' references unknown node '{param_value}'"

    def summary(self) -> str:
        """Return a one-line summary of object, task, and relation counts."""
        return (
            f"{len(self.objects)} objects · {len(self.task.subtasks)} atomic tasks "
            f"({self.task.composition}) · {len(self.relations)} relations"
        )

    def _validate_cli_override_specs(self) -> None:
        """Check each CLI override uses a unique flag and points to a swappable asset."""
        cli_override_specs = self.cli_override_specs or []
        swappable_ids = {
            self.embodiment.id,
            self.background.id,
            *(obj.id for obj in self.objects),
        }
        seen_args: set[str] = set()
        for override in cli_override_specs:
            assert override.arg not in seen_args, f"Duplicate cli_override arg '--{override.arg}'"
            seen_args.add(override.arg)
            assert (
                override.target_node_id in swappable_ids
            ), f"CLI override '--{override.arg}' targets unknown or non-swappable asset '{override.target_node_id}'"

    def _asset_by_id(self, asset_id: str) -> AssetSpec:
        if self.embodiment.id == asset_id:
            return self.embodiment
        if self.background.id == asset_id:
            return self.background
        for obj in self.objects:
            if obj.id == asset_id:
                return obj
        raise KeyError(asset_id)

    @staticmethod
    def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
        return load_env_graph_spec_dict(path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.from_dict(cls._load_yaml_dict(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain YAML mapping — the inverse of ``from_dict``."""
        return self.model_dump(mode="json", exclude_none=True)

    def write_yaml(self, path: str | Path) -> None:
        """Validate this spec and write it to ``path`` as YAML."""
        self.validate()
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def read_cli_override_specs(path: str | Path) -> list[CliOverrideSpec]:
        """Read just the ``cli_override_specs`` section of a graph YAML."""
        raw_specs = ArenaEnvGraphSpec._load_yaml_dict(path).get("cli_override_specs") or []
        return [CliOverrideSpec.model_validate(entry) for entry in raw_specs]

    def apply_cli_override_args(self, args_cli: argparse.Namespace) -> None:
        """Apply CLI override flags in place by swapping target assets' ``registry_name``."""
        # TODO(cvolk, 2026-07-06): [typed-config-migration] Replace Namespace-based graph overrides with a
        # typed graph-construction boundary in a dedicated environment-graph refactor.
        for override in self.cli_override_specs or []:
            new_name = getattr(args_cli, override.dest, None)
            if new_name is not None:
                self._asset_by_id(override.target_node_id).registry_name = new_name

    def to_arena_env(self, enable_cameras: bool = False) -> IsaacLabArenaEnvironment:
        """Convert this graph spec into an :class:`IsaacLabArenaEnvironment`.

        Args:
            enable_cameras: Forwarded to the embodiment so its cameras are spawned.
        """
        from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

        return build_arena_env_from_graph_spec(self, enable_cameras=enable_cameras)
