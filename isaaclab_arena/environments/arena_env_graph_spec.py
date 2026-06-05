# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SerializeAsAny, field_validator, model_validator

from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphCliOverrideSpec,
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialRelationSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskConstraintSpec,
    ArenaEnvGraphTaskConstraintType,
    ArenaEnvGraphTaskSpec,
    parse_graph_node,
)
from isaaclab_arena.environments.graph_spec_utils import (
    assert_cli_override_specs_reference_nodes,
    assert_references_exist,
    assert_spatial_constraint_shapes,
    assert_unique_ids,
)

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Re-exported for callers that already import these names from this module.
__all__ = [
    "ArenaEnvGraphCliOverrideSpec",
    "ArenaEnvGraphNodeSpec",
    "ArenaEnvGraphNodeType",
    "ArenaEnvGraphObjectReferenceNodeSpec",
    "ArenaEnvGraphSpatialRelationSpec",
    "ArenaEnvGraphSpec",
    "ArenaEnvGraphStateSpec",
    "ArenaEnvGraphTaskConstraintSpec",
    "ArenaEnvGraphTaskConstraintType",
    "ArenaEnvGraphTaskSpec",
]


class ArenaEnvGraphSpec(BaseModel):
    """Typed representation of an environment graph YAML file."""

    env_name: str = Field(min_length=1)
    nodes: list[SerializeAsAny[ArenaEnvGraphNodeSpec]] = Field(default_factory=list)
    tasks: list[ArenaEnvGraphTaskSpec] = Field(default_factory=list)
    state_specs: list[ArenaEnvGraphStateSpec] = Field(default_factory=list)
    cli_override_specs: list[ArenaEnvGraphCliOverrideSpec] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _parse_nodes(cls, nodes: Any) -> list[Any]:
        if nodes is None:
            return []
        if not isinstance(nodes, list):
            raise ValueError("Field 'nodes' must be a list")
        return [parse_graph_node(node) for node in nodes]

    @model_validator(mode="after")
    def validate(self) -> ArenaEnvGraphSpec:
        """Check unique ids, cross-references, constraint shapes, and CLI overrides."""
        assert_unique_ids(self.nodes, self.tasks, self.state_specs)
        assert_references_exist(self.nodes, self.tasks, self.state_specs)
        assert_spatial_constraint_shapes(self.state_specs)
        assert_cli_override_specs_reference_nodes(self.nodes, self.cli_override_specs)
        return self

    @staticmethod
    def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
        """Load a graph YAML into a dict. Fail with a clear message if the file is missing."""
        path = Path(path)
        assert path.is_file(), f"Env graph spec YAML not found: {path}"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArenaEnvGraphSpec:
        return cls.from_dict(cls._load_yaml_dict(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArenaEnvGraphSpec:
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return cls.model_validate(data)

    @staticmethod
    def read_cli_override_specs(path: str | Path) -> list[ArenaEnvGraphCliOverrideSpec]:
        """Read just the ``cli_override_specs`` section of a graph YAML, skipping the rest.

        The CLI flags need to be registered before the simulator starts. Loading the full
        graph would import ``pxr`` too early, so this only reads the override entries.
        """
        raw_specs = ArenaEnvGraphSpec._load_yaml_dict(path).get("cli_override_specs") or []
        return [ArenaEnvGraphCliOverrideSpec.model_validate(entry) for entry in raw_specs]

    def apply_cli_override_args(self, args_cli: argparse.Namespace) -> None:
        """Apply the CLI override flags to this graph, in place.

        For each override, set the target node's asset ``name`` to the value passed on the
        command line. Flags left unset are skipped, so an untouched graph stays the same.
        """
        nodes_by_id = self.nodes_by_id
        for override in self.cli_override_specs:
            new_name = getattr(args_cli, override.dest, None)
            if new_name is not None:
                nodes_by_id[override.target_node_id].name = new_name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain YAML mapping — the inverse of ``from_dict``."""
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
        # Lazy import: build_arena_env_from_graph_spec pulls in Scene -> phyx_utils ->
        # pxr.PhysxSchema, which requires SimulationApp. Keeping the import here lets
        # data-only consumers of the spec (parsers, tests) import this module before
        # SimulationApp is started.
        # TODO(xinjieyao, 2026-05-26): once `build_arena_env_from_graph_spec` aggregates across all state_specs,
        # this wrapper stays single-arg — no caller-side selection is needed.
        from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

        return build_arena_env_from_graph_spec(self)
