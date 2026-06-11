# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field, SerializeAsAny, field_validator, model_validator

from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphCliOverrideSpec,
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphSpatialRelationSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskSpec,
    TaskSpec,
    parse_graph_node,
)
from isaaclab_arena.environments.graph_spec_utils import (
    assert_cli_override_specs_reference_nodes,
    assert_constraint_references,
    assert_spatial_constraint_shapes,
    assert_task_wiring,
    assert_unique_ids,
)
from isaaclab_arena.tasks.task_transition import Effect, Relocate, TaskTransition

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


class ArenaEnvGraphSpecBase(BaseModel):
    """Shared fields and serialization helpers for env-graph spec classes."""

    env_name: str = Field(min_length=1)
    nodes: list[SerializeAsAny[ArenaEnvGraphNodeSpec]] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _parse_nodes(cls, nodes: Any) -> list[Any]:
        if nodes is None:
            return []
        if not isinstance(nodes, list):
            raise ValueError("Field 'nodes' must be a list")
        return [parse_graph_node(node) for node in nodes]

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

    def validate(self) -> Self:
        """Validate this spec. Override in subclasses to enforce invariants."""
        return self

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


class ArenaEnvGraphSpec(ArenaEnvGraphSpecBase):
    """A full environment graph with all tasks wired to state specs."""

    tasks: list[ArenaEnvGraphTaskSpec] = Field(default_factory=list)
    state_specs: list[ArenaEnvGraphStateSpec] = Field(default_factory=list)
    cli_override_specs: list[ArenaEnvGraphCliOverrideSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate(self) -> ArenaEnvGraphSpec:
        """Check unique ids, cross-references, constraint shapes, task wiring, and CLI overrides."""
        assert_unique_ids(self.nodes, self.tasks, self.state_specs)
        assert_constraint_references(self.nodes, self.state_specs)
        assert_task_wiring(self.tasks, self.state_specs)
        assert_spatial_constraint_shapes(self.state_specs)
        assert_cli_override_specs_reference_nodes(self.nodes, self.cli_override_specs)
        return self

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


class ArenaEnvInitialGraphSpec(ArenaEnvGraphSpecBase):
    """Initial-state environment graph produced by the agent intent pipeline."""

    tasks: list[TaskSpec] = Field(default_factory=list)
    initial_state_spec: ArenaEnvGraphStateSpec

    @model_validator(mode="after")
    def validate(self) -> Self:
        """Check unique IDs, constraint references, and spatial constraint shapes."""
        assert_unique_ids(self.nodes, [], [self.initial_state_spec])
        assert_constraint_references(self.nodes, [self.initial_state_spec])
        assert_spatial_constraint_shapes([self.initial_state_spec])
        return self

    def link(self) -> ArenaEnvGraphSpec:
        """Link this initial graph into a fully-wired :class:`ArenaEnvGraphSpec`.

        Uses :attr:`initial_state_spec` as ``state_spec_0``, then chains each task's declared
        ``success_state_transition`` to produce a delta state.  The topology is implicit in the
        sequential task list: task ``i`` carries ``state_spec_i`` to ``state_spec_{i+1}``,
        yielding ``N+1`` states for ``N`` tasks.

        Returns:
            A fully-wired, strictly-validated :class:`ArenaEnvGraphSpec`.
        """
        states: list[ArenaEnvGraphStateSpec] = [self.initial_state_spec]
        out_tasks: list[ArenaEnvGraphTaskSpec] = []

        for i, task in enumerate(self.tasks):
            new_state_id = f"state_spec_{i + 1}"
            transition = _get_task_state_transition(task)
            states.append(_get_next_state_spec(new_state_id, transition))
            out_tasks.append(
                ArenaEnvGraphTaskSpec(
                    id=f"task_{i}_{task.kind}",
                    kind=task.kind,
                    params=task.params,
                    description=task.description,
                    initial_state_spec_id=states[i].id,
                    success_state_spec_id=new_state_id,
                )
            )

        return ArenaEnvGraphSpec(
            env_name=self.env_name,
            nodes=self.nodes,
            tasks=out_tasks,
            state_specs=states,
        )


# ---------------------------------------------------------------------------
# Link helpers (used by ArenaEnvInitialGraphSpec.link)
# ---------------------------------------------------------------------------


def _get_task_state_transition(task: TaskSpec) -> TaskTransition:
    """Look up the task class via ``TaskRegistry`` and return its declared success transition.

    All task params are forwarded; ``success_state_transition`` binds only the ones it acts
    on as named parameters and ignores the rest (scene, episode length, …) via ``**_``.
    """
    task_cls = TaskRegistry().get_task_by_name(task.kind)
    assert task_cls is not None, f"task '{task.kind}' not found in TaskRegistry"
    return task_cls.success_state_transition(**task.params)


def _get_task_effects(transition: TaskTransition) -> list[Effect]:
    """Return only the ``Relocate`` effects; other effect types (e.g. ``SetState``) are skipped."""
    return [effect for effect in transition.effects if isinstance(effect, Relocate)]


def _get_next_state_spec(new_state_id: str, transition: TaskTransition) -> ArenaEnvGraphStateSpec:
    """Build the delta state for one task's success transition.

    Each derived state records only the spatial constraints introduced by the task
    (``is_delta=True``), not a full snapshot.  E.g. for a PnP task that places the mug
    on the bowl::

        mug on bowl  # only the relocation effect

    Args:
        new_state_id: ID to assign to the new state spec.
        transition: The task's declared success transition.

    Returns:
        An ``ArenaEnvGraphStateSpec`` with ``is_delta=True`` containing only the
        relocation constraints produced by this task.
    """
    spatial_constraints = [
        ArenaEnvGraphSpatialRelationSpec(
            id=f"{new_state_id}_{r.subject}_{r.relation}_{r.target}",
            kind=r.relation,
            reference=r.target,
            subject=r.subject,
        )
        for r in _get_task_effects(transition)
    ]
    return ArenaEnvGraphStateSpec(id=new_state_id, is_delta=True, spatial_constraints=spatial_constraints)
