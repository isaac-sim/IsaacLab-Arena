# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, SerializeAsAny, ValidationInfo, field_validator, model_validator

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
    spatial_constraint_is_spawn_pose,
)
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition

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
        only fills the per-state constraints: each task's success condition is applied to the previous
        state to derive the next one. ``N`` tasks therefore yield ``N+1`` state specs.

        Returns a fully-wired, strict-validated ``ArenaEnvGraphSpec``.
        """
        assert len(self.state_specs) == 1, (
            f"unresolved graph must define exactly the initial state (state_spec_0); got {len(self.state_specs)} state"
            " specs"
        )
        embodiment_id = _get_embodiment_id_from_nodes(self.nodes)

        transitions = [_get_task_state_transition(task) for task in self.tasks]

        # state_spec_0 is the given initial state; chain the rest off the task list.
        states: list[ArenaEnvGraphStateSpec] = [self.state_specs[0]]
        out_tasks: list[ArenaEnvGraphTaskSpec] = []
        num_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            new_state_id = f"state_spec_{i + 1}"
            is_final_state = i == num_tasks - 1
            # A success state is both a postcondition of the task that just ran and a precondition of the
            # next one, so it asserts reachability of both: the completed task's target (e.g. a place
            # destination), and -- when a next task exists -- that task's subject (the next thing to act on).
            reach_targets_postcondition = [transitions[i].reach_target_on_success]
            reach_targets_precondition = []
            # final state has no precondition.
            if not is_final_state:
                reach_targets_precondition = [transitions[i + 1].subject]
            reach_targets = reach_targets_postcondition + reach_targets_precondition
            states.append(
                _get_next_state_spec(
                    prev_state_spec=states[-1],
                    new_state_id=new_state_id,
                    transition=transitions[i],
                    embodiment_id=embodiment_id,
                    reach_targets=reach_targets,
                    is_final_state=is_final_state,
                )
            )
            out_tasks.append(
                task.model_copy(
                    update={"initial_state_spec_id": f"state_spec_{i}", "success_state_spec_id": new_state_id}
                )
            )

        return ArenaEnvGraphSpec(
            env_name=env_name or self.env_name,
            nodes=self.nodes,
            tasks=out_tasks,
            state_specs=states,
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


def _get_embodiment_id_from_nodes(nodes: list[ArenaEnvGraphNodeSpec]) -> str:
    """Return the embodiment node's id (the parent of every reach constraint); the first one wins."""
    ids = [node.id for node in nodes if node.type == ArenaEnvGraphNodeType.EMBODIMENT]
    assert ids, "graph has no embodiment node"
    return ids[0]


def _get_task_state_transition(task: ArenaEnvGraphTaskSpec) -> TaskTransition:
    """Look up the task class via ``TaskRegistry`` and return its declared success transition."""
    task_cls = TaskRegistry().get_task_by_name(task.type)
    assert task_cls is not None, f"task {task.type} not found in TaskRegistry"
    return task_cls.success_state_transition(task.task_args)


def _get_task_relocations(transition: TaskTransition) -> list[Relocate]:
    """Return the transition's ``Relocate`` effects."""
    for effect in transition.effects:
        if not isinstance(effect, Relocate):
            raise NotImplementedError(f"Effect {effect} is not yet supported.")
    return list(transition.effects)


def _get_next_state_spec(
    prev_state_spec: ArenaEnvGraphStateSpec,
    new_state_id: str,
    transition: TaskTransition,
    embodiment_id: str,
    reach_targets: list[str | None],
    is_final_state: bool,
) -> ArenaEnvGraphStateSpec:
    """Apply the task's success condition to the previous state to build the next state.

    Carry every constraint forward, except a moved object's old placement (replaced by where it
    lands). The final state also drops spawn poses, keeping only structural relations -- it is only
    ever checked as a goal, never reset into.

    Worked example -- task "place mug on bowl" (mug is the moved object), prev state holds::

        mug at_position {x, y, z}          # mug's spawn pose
        bowl on table                      # structural
        bowl position_limits {x_min, ...}  # bowl's spawn pose (bowl is never moved)

    yields, at an interior success state::

        bowl on table                      # carried (unaffected)
        bowl position_limits {...}         # carried (still a reset-from state)
        mug on bowl                        # added (the relocation)
        # mug at_position dropped -- replaced by the relocation above

    and at the final state, additionally::

        # bowl position_limits dropped -- final state keeps only structural relations

    Args:
        prev_state_spec: The previous state spec.
        new_state_id: The id of the new state.
        transition: The current task's declared success transition.
        embodiment_id: The id of the embodiment.
        reach_targets: The list of targets the embodiment must reach in this state.
        is_final_state: Whether the state is the last one when reaching the final state.

    Returns:
        The next state spec.
    """
    relocations = _get_task_relocations(transition)
    moved_objects_ids = {relocation.subject for relocation in relocations}
    spatial_constraints: list[ArenaEnvGraphSpatialConstraintSpec] = []

    for prev_spatial_constraint in prev_state_spec.spatial_constraints:
        # Spawn pose constraints are those that are set at reset time. e.g."at position", "position_limits", etc.
        constraint_is_spawn_pose = spatial_constraint_is_spawn_pose(prev_spatial_constraint.type)
        # A moved object's old placement is replaced by the new placement
        # Case 1: when the object is the constraint's child. (e.g. "cube on table" -> "cube on shelf")
        # Case 2: when the object is the owner of its own (unary) spawn-pose constraint. (e.g. "cube at_position A" -> "cube at_position B")
        child_is_relocated = prev_spatial_constraint.child in moved_objects_ids
        parent_is_spawn_pose_owner = (prev_spatial_constraint.parent in moved_objects_ids) and constraint_is_spawn_pose
        constraint_is_replaced = child_is_relocated or parent_is_spawn_pose_owner

        # Because the spawn pose is at reset time, it is not affected by the success condition.
        # e.g. a never-moved bowl's "position_limits".
        constraint_is_dropped = is_final_state and constraint_is_spawn_pose

        # Keep the old ones (spawn pose or structural) when they are not affected by the success condition.
        if not (constraint_is_replaced or constraint_is_dropped):
            spatial_constraints.append(
                prev_spatial_constraint.model_copy(
                    update={"id": _reprefix_id(prev_spatial_constraint.id, prev_state_spec.id, new_state_id)}
                )
            )
    # Add the new ones.
    for relocation in relocations:
        spatial_constraints.append(
            ArenaEnvGraphSpatialConstraintSpec(
                id=f"{new_state_id}_{relocation.subject}_{relocation.relation}_{relocation.target}",
                type=relocation.relation,
                parent=relocation.target,
                child=relocation.subject,
            )
        )
    # One reach constraint per distinct target the embodiment must reach in this state.
    # e.g. PnP A to B -> close door: B & door shall be reachable in this state.
    seen_targets: set[str] = set()
    task_constraints: list[ArenaEnvGraphTaskConstraintSpec] = []
    for reach_target in reach_targets:
        if reach_target is not None and reach_target not in seen_targets:
            seen_targets.add(reach_target)
            task_constraints.append(
                ArenaEnvGraphTaskConstraintSpec(
                    id=f"{new_state_id}_{embodiment_id.split('_')[0]}_reach_{reach_target}",
                    type=ArenaEnvGraphTaskConstraintType.REACH,
                    parent=embodiment_id,
                    child=reach_target,
                )
            )
    return ArenaEnvGraphStateSpec(
        id=new_state_id, spatial_constraints=spatial_constraints, task_constraints=task_constraints
    )


def _reprefix_id(old_id: str, old_prefix: str, new_prefix: str) -> str:
    """Swap a constraint id's ``{state_id}`` prefix so carried constraints stay uniquely named."""
    assert old_id.startswith(old_prefix), f"constraint id {old_id!r} is not prefixed by its state id {old_prefix!r}"
    return new_prefix + old_id[len(old_prefix) :]
