# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_types import (
    AssetSpec,
    CliOverrideSpec,
    ObjectReferenceSpec,
    SpatialRelationSpec,
    TaskSpec,
)

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def required_task_init_param_names(task_cls: type) -> list[str]:
    """Return required ``__init__`` parameter names for a task class."""
    sig = inspect.signature(task_cls.__init__)
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


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
    tasks: list[TaskSpec] = Field(
        default_factory=list, description="Tasks the robot performs to manipulate the objects."
    )
    cli_override_specs: list[CliOverrideSpec] | None = Field(
        default=None, description="Optional authoring-time CLI flags that swap an asset's registry_name; usually empty."
    )

    @field_validator("object_references", "cli_override_specs", mode="before")
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
            self._assert_object_reference_parents(self.object_references, known_ids)
        self._assert_relation_references(self.relations, known_ids)
        self._assert_task_param_references(self.tasks, known_ids)
        self._validate_cli_override_specs()
        self._validate_agent_ready_tasks()
        return self

    def _assert_asset_ids_unique(self) -> None:
        """Ensure every asset and object-reference id in this spec is unique."""
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
    def _assert_object_reference_parents(object_references: list[ObjectReferenceSpec], known_ids: set[str]) -> None:
        """Ensure each object reference parent exists."""
        for ref in object_references:
            assert (
                ref.parent_id in known_ids
            ), f"Object reference '{ref.id}' references unknown parent '{ref.parent_id}'"

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
    def _assert_task_param_references(tasks: list[TaskSpec], known_ids: set[str]) -> None:
        """Ensure string-valued task params reference known asset ids."""
        for task in tasks:
            for param_name, param_value in task.params.items():
                if isinstance(param_value, str):
                    assert (
                        param_value in known_ids
                    ), f"Task '{task.kind}' param '{param_name}' references unknown node '{param_value}'"

    def summary(self) -> str:
        """Return a one-line summary of object, task, and relation counts."""
        return f"{len(self.objects)} objects · {len(self.tasks)} tasks · {len(self.relations)} relations"

    def _validate_agent_ready_tasks(self) -> None:
        task_registry = TaskRegistry()
        for task in self.tasks:
            task_cls = task_registry.get_task_by_name(task.kind)
            assert getattr(task_cls, "agent_ready", False), f"Task {task.kind!r} is not agent-ready"
            assert task.description and task.description.strip(), f"Task {task.kind!r} requires a non-empty description"
            for required_param in required_task_init_param_names(task_cls):
                assert required_param in task.params, f"Task {task.kind!r} is missing required param {required_param}"
                value = task.params[required_param]
                assert (
                    isinstance(value, str) and value.strip()
                ), f"Task {task.kind!r} required param {required_param!r} must be a non-empty string"

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
        path = Path(path)
        assert path.is_file(), f"Env graph spec YAML not found: {path}"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return data

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
        for override in self.cli_override_specs or []:
            new_name = getattr(args_cli, override.dest, None)
            if new_name is not None:
                self._asset_by_id(override.target_node_id).registry_name = new_name

    def to_arena_env(self, enable_cameras: bool = False) -> IsaacLabArenaEnvironment:
        """Convert this graph spec into an :class:`IsaacLabArenaEnvironment`.

        Args:
            enable_cameras: Forwarded to the embodiment so its cameras are spawned.
        """
        from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

        return build_arena_env_from_graph_spec(self, enable_cameras=enable_cameras)
