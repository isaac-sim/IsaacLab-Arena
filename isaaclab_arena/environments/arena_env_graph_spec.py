# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator, model_validator

from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskConstraintSpec,
    ArenaEnvGraphTaskConstraintType,
    ArenaEnvGraphTaskSpec,
    _coerce_graph_node,
)
from isaaclab_arena.environments.graph_spec_utils import (
    validate_references_exist,
    validate_spatial_constraint_shapes,
    validate_unique_ids,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Re-exported for callers that already import these names from this module.
__all__ = [
    "ArenaEnvGraphNodeSpec",
    "ArenaEnvGraphNodeType",
    "ArenaEnvGraphObjectReferenceNodeSpec",
    "ArenaEnvGraphSpatialConstraintSpec",
    "ArenaEnvGraphSpatialConstraintType",
    "ArenaEnvGraphSpec",
    "ArenaEnvGraphStateSpec",
    "ArenaEnvGraphTaskConstraintSpec",
    "ArenaEnvGraphTaskConstraintType",
    "ArenaEnvGraphTaskSpec",
]


class ArenaEnvGraphSpec(BaseModel):
    """Typed representation of an environment graph YAML file."""

    model_config = ConfigDict(extra="ignore")

    env_name: str = Field(min_length=1)
    nodes: list[SerializeAsAny[ArenaEnvGraphNodeSpec]] = Field(default_factory=list)
    tasks: list[ArenaEnvGraphTaskSpec] = Field(default_factory=list)
    state_specs: list[ArenaEnvGraphStateSpec] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _parse_nodes(cls, nodes: Any) -> list[Any]:
        if nodes is None:
            return []
        if not isinstance(nodes, list):
            raise ValueError("Field 'nodes' must be a list")
        return [_coerce_graph_node(node) for node in nodes]

    @model_validator(mode="after")
    def _validate_graph_invariants(self) -> ArenaEnvGraphSpec:
        self._check_graph_invariants()
        return self

    def _check_graph_invariants(self) -> None:
        validate_unique_ids(self.nodes, self.tasks, self.state_specs)
        validate_references_exist(self.nodes, self.tasks, self.state_specs)
        validate_spatial_constraint_shapes(self.state_specs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArenaEnvGraphSpec:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Env graph spec YAML not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArenaEnvGraphSpec:
        if not isinstance(data, dict):
            raise ValueError(f"Env graph spec must be a dict, got {type(data).__name__}")
        return cls.model_validate(data)

    def validate(self) -> None:
        """Re-run graph-level validators (e.g. after mutating a loaded spec)."""
        self._check_graph_invariants()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain YAML mapping — the inverse of ``from_dict``.

        Uses Pydantic's JSON dump mode so enums become strings, ``None`` is omitted,
        tuples become lists, and ``object_reference`` nodes retain their extra fields
        (via :class:`~pydantic.SerializeAsAny`).
        """
        return self.model_dump(mode="json", exclude_none=True)

    def write_yaml(self, path: str | Path) -> None:
        """Validate this spec and write it to ``path`` as YAML."""
        self.validate()
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @property
    def nodes_by_id(self) -> dict[str, ArenaEnvGraphNodeSpec]:
        return {node.id: node for node in self.nodes}

    @property
    def tasks_by_id(self) -> dict[str, ArenaEnvGraphTaskSpec]:
        return {task.id: task for task in self.tasks}

    @property
    def state_specs_by_id(self) -> dict[str, ArenaEnvGraphStateSpec]:
        return {state_spec.id: state_spec for state_spec in self.state_specs}

    def to_arena_env(self) -> IsaacLabArenaEnvironment:
        """Convert this graph spec into an `IsaacLabArenaEnvironment`.

        The first ``state_spec`` is used as the scene's initial state.
        """
        from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

        return build_arena_env_from_graph_spec(self)
