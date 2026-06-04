# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministically resolve a partially-wired ``ArenaEnvGraphSpec`` into a fully-populated one.

The upstream produces a partially-wired env graph: the nodes, the task list, an initial state (``state_spec_0``),
but no task-to-state wiring. This module chains the per-task success conditions into the missing intermediate
and final states, then wires each task to its initial and success states.
"""

from __future__ import annotations

from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphSpatialConstraintSpec,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskConstraintSpec,
    ArenaEnvGraphTaskConstraintType,
    ArenaEnvGraphTaskSpec,
)
from isaaclab_arena.environments.graph_spec_utils import spatial_constraint_is_spawn_pose
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition


def resolve_constraints(
    partially_wired_arena_env_graph_spec: ArenaEnvGraphSpec, env_name: str | None = None
) -> ArenaEnvGraphSpec:
    """Chain the spec's task success conditions into its full state specs and return a validated spec.

    The topology is implicit in the sequential task list (task ``i`` carries ``state_spec_i`` to
    ``state_spec_{i+1}``), so resolving only fills the per-state constraints.

    Args:
        partially_wired_arena_env_graph_spec: An unresolved graph — tasks not yet wired, only the initial
            state ``state_spec_0`` present.
        env_name: Name for the resolved env.

    Returns:
        A fully-wired, validated ``ArenaEnvGraphSpec``.
    """
    spec = partially_wired_arena_env_graph_spec
    assert (
        len(spec.state_specs) == 1
    ), f"unresolved graph must define exactly the initial state (state_spec_0); got {len(spec.state_specs)} state specs"
    embodiment_id = _get_embodiment_id_from_nodes(spec.nodes)

    transitions = [_get_task_state_transition(task) for task in spec.tasks]

    # state_spec_0 is the given initial state; chain the rest off the task list.
    states: list[ArenaEnvGraphStateSpec] = [spec.state_specs[0]]
    out_tasks: list[ArenaEnvGraphTaskSpec] = []
    num_tasks = len(spec.tasks)
    for i, task in enumerate(spec.tasks):
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
                prev_state=states[-1],
                new_state_id=new_state_id,
                transition=transitions[i],
                embodiment_id=embodiment_id,
                reach_targets=reach_targets,
                is_final_state=is_final_state,
            )
        )
        out_tasks.append(
            task.model_copy(update={"initial_state_spec_id": f"state_spec_{i}", "success_state_spec_id": new_state_id})
        )

    return ArenaEnvGraphSpec(
        # Defaults to the input spec's ``env_name`` if not provided.
        env_name=env_name or spec.env_name,
        nodes=spec.nodes,
        tasks=out_tasks,
        state_specs=states,
    )


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
    """Apply current task's success condition to the previous state spec to produce the next state spec.

    Carries every constraint forward (re-prefixing its id to the new state id) except each
    relocated object's old placement, which is replaced by what the success condition implies.
    The final state spec additionally drops spawn-pose constraints (e.g. "at_position"), keeping only structural relations.
    The drop is a semantic-cleanliness rule: spawn poses belong to reset-from states (they are
    reset-time placement hints), structural relations belong to success states; the final state is
    never reset-from, only checked as a success condition, so it is purely the latter.

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
