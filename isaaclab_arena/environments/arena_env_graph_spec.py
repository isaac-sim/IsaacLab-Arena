# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SerializeAsAny, ValidationInfo, field_validator, model_validator

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphCliOverrideSpec,
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintSpec,
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
    assert_task_wiring,
    assert_unique_ids,
)
from isaaclab_arena.tasks.task_transition import Effect, Relocate, TaskTransition

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Re-exported for callers that already import these names from this module.
__all__ = [
    "ArenaEnvGraphCliOverrideSpec",
    "ArenaEnvGraphNodeSpec",
    "ArenaEnvGraphNodeType",
    "ArenaEnvGraphObjectReferenceNodeSpec",
    "ArenaEnvGraphSpatialConstraintSpec",
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
    def _validate_on_construction(self, info: ValidationInfo) -> ArenaEnvGraphSpec:
        """Run graph validation at construction; an unresolved load skips the task wiring check.

        ``from_yaml`` / ``from_dict`` pass ``is_task_wiring_enabled`` through the validation context;
        it is ``False`` for an unresolved graph whose tasks are not yet wired to states.
        """
        is_task_wiring_enabled = True if info.context is None else info.context.get("is_task_wiring_enabled", True)
        self._run_validation(is_task_wiring_enabled=is_task_wiring_enabled)
        return self

    def validate(self) -> ArenaEnvGraphSpec:
        """Re-check the full (strict) graph invariants, e.g. after mutating the spec in place."""
        self._run_validation(is_task_wiring_enabled=True)
        return self

    def _run_validation(self, is_task_wiring_enabled: bool) -> None:
        """Check unique ids, cross-references, constraint shapes, and CLI override specs."""
        assert_unique_ids(self.nodes, self.tasks, self.state_specs)
        assert_references_exist(self.nodes, self.tasks, self.state_specs)
        if is_task_wiring_enabled:
            assert_task_wiring(self.tasks, self.state_specs)
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
    def from_yaml(cls, path: str | Path, is_task_wiring_enabled: bool = True) -> ArenaEnvGraphSpec:
        return cls.from_dict(cls._load_yaml_dict(path), is_task_wiring_enabled=is_task_wiring_enabled)

    @classmethod
    def from_dict(cls, data: dict[str, Any], is_task_wiring_enabled: bool = True) -> ArenaEnvGraphSpec:
        """Validate and build a spec from a mapping.

        Pass ``is_task_wiring_enabled=False`` to load an *unresolved* graph.
        """
        assert isinstance(data, dict), f"Env graph spec must be a dict, got {type(data).__name__}"
        return cls.model_validate(data, context={"is_task_wiring_enabled": is_task_wiring_enabled})

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

    def resolve_constraints(self, env_name: str | None = None) -> ArenaEnvGraphSpec:
        """Chain this unresolved graph's task success conditions into its full state specs.

        The upstream emits a partially-wired graph: nodes, the ordered task list, and only the initial
        state ``state_spec_0`` -- tasks are not yet wired to states. The topology is implicit in the
        sequential task list (task ``i`` carries ``state_spec_i`` to ``state_spec_{i+1}``), so resolving
        only fills the per-state constraints. The chain captures the diffs: ``state_spec_0`` is the
        initial snapshot, and each derived state records only the delta its task introduced (e.g. where the
        moved objects land), not a full snapshot. ``N`` tasks therefore yield ``N+1`` state specs.

        Returns a fully-wired, strict-validated ``ArenaEnvGraphSpec``.
        """
        assert len(self.state_specs) == 1, (
            f"unresolved graph must define exactly the initial state (state_spec_0); got {len(self.state_specs)} state"
            " specs"
        )
        # state_spec_0 is the given initial state -- a full snapshot, not a delta -- and the transitions
        # chain the rest off the task list.
        states: list[ArenaEnvGraphStateSpec] = [self.state_specs[0].model_copy(update={"is_delta": False})]
        out_tasks: list[ArenaEnvGraphTaskSpec] = []

        for i, curr_task in enumerate(self.tasks):
            new_state_id = f"state_spec_{i + 1}"
            curr_task_transition = _get_task_state_transition(curr_task)
            assert curr_task_transition is not None, f"task {curr_task.id} has no transition"
            states.append(
                _get_next_state_spec(
                    new_state_id=new_state_id,
                    transition=curr_task_transition,
                )
            )
            out_tasks.append(
                curr_task.model_copy(
                    update={"initial_state_spec_id": f"state_spec_{i}", "success_state_spec_id": new_state_id}
                )
            )

        return ArenaEnvGraphSpec(
            env_name=env_name or self.env_name,
            nodes=self.nodes,
            tasks=out_tasks,
            state_specs=states,
            cli_override_specs=self.cli_override_specs,
        )

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


# --- Constraint resolution helpers (used by ArenaEnvGraphSpec.resolve_constraints) ---


def _get_task_state_transition(task: ArenaEnvGraphTaskSpec) -> TaskTransition:
    """Look up the task class via ``TaskRegistry`` and return its declared success transition."""
    task_cls = TaskRegistry().get_task_by_name(task.type)
    assert task_cls is not None, f"task {task.type} not found in TaskRegistry"
    asset_args = {}
    for name in task_cls.success_transition_asset_args:
        assert name in task.task_args, f"task {task.id} ({task.type}) is missing required task_arg '{name}'"
        asset_args[name] = Asset(name=task.task_args[name])
    return task_cls.success_state_transition(**asset_args)


def _get_task_effects(transition: TaskTransition) -> list[Effect]:
    """Return the transition's ``Effect``s. Right now we only support Relocate effects."""
    return [effect for effect in transition.effects if isinstance(effect, Relocate)]


def _get_next_state_spec(new_state_id: str, transition: TaskTransition) -> ArenaEnvGraphStateSpec:
    """Build the success state as the *delta* of the task's transition -- only what it changed.

    The chain focuses on the delta: ``state_spec_0`` holds the initial state, and each derived state
    records only the effects the task established, not a fresh snapshot of the whole scene (followed by is_delta=True).

    E.g. task "place mug on bowl" (mug is the moved object) yields a state holding only::

        mug on bowl  # the effect from the PnP task's relocation

    Args:
        new_state_id: The id of the new state.
        transition: The current task's declared success transition.

    Returns:
        The next state spec, carrying only the task's relocation constraints.
    """
    task_effects = _get_task_effects(transition)
    spatial_constraints = [
        ArenaEnvGraphSpatialConstraintSpec(
            id=f"{new_state_id}_{relocation.subject}_{relocation.relation}_{relocation.target}",
            type=relocation.relation,
            parent=relocation.target,
            child=relocation.subject,
        )
        for relocation in task_effects
    ]
    return ArenaEnvGraphStateSpec(id=new_state_id, is_delta=True, spatial_constraints=spatial_constraints)
