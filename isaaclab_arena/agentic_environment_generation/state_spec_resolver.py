# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministically resolve an unresolved ``ArenaEnvGraphSpec`` into a fully-populated one.

The upstream produces an *unresolved* env graph: the nodes, the task list, and a
single intended initial state (``state_spec_0``); the tasks carry no
``initial_state_spec_id`` / ``success_state_spec_id`` yet. ``resolve_constraints`` chains
the per-task success conditions into the missing intermediate and final states, then wires
each task to its states.

The topology is implicit in the sequential task list, so resolving never changes it — it
only fills the per-state constraints. Each task's success-to-state-change mapping is a total
lookup against the task class itself (``TaskBase.success_state_transition`` via
``TaskRegistry``). The chaining and id assignment must be exact (a groundtruth fixture pins the
output). Reached via ``ArenaEnvGraphSpec.resolve_constraints``.
"""

from __future__ import annotations

from typing import Any

from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec, ArenaEnvGraphTaskSpec
from isaaclab_arena.environments.graph_spec_utils import spatial_constraint_is_spawn_pose
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition


def resolve_constraints(spec: ArenaEnvGraphSpec, env_name: str | None = None) -> ArenaEnvGraphSpec:
    """Chain ``spec``'s task success conditions into the full state graph and return it validated.

    Args:
        spec: An unresolved graph (loaded via ``ArenaEnvGraphSpec.from_dict(..., is_task_wiring_enabled=False)``):
            tasks not yet wired, only the initial state ``state_spec_0`` present.
        env_name: Name for the resolved env. Defaults to ``spec.env_name``.

    Returns:
        A fully-wired ``ArenaEnvGraphSpec`` (``from_dict`` validates references and constraint
        shapes before it is returned).
    """
    return ArenaEnvGraphSpec.from_dict(_resolve_to_dict(spec, env_name=env_name))


def _resolve_to_dict(spec: ArenaEnvGraphSpec, env_name: str | None = None) -> dict[str, Any]:
    """Build the full env-graph mapping (pre-validation) from the unresolved graph.

    Chaining runs in dict space: nodes and the initial state come from ``spec.to_dict()``; tasks
    are read from its typed ``tasks``.
    """
    raw = spec.to_dict()
    nodes = raw["nodes"]
    tasks_in = spec.tasks
    states_in = raw.get("state_specs", [])
    assert (
        len(states_in) == 1
    ), f"unresolved graph must define exactly the initial state (state_spec_0); got {len(states_in)} state specs"
    embodiment_id = _embodiment_id(nodes)

    # state_spec_0 is the given initial state; chain the rest off the task list.
    states: list[dict[str, Any]] = [states_in[0]]
    out_tasks: list[dict[str, Any]] = []
    num_tasks = len(tasks_in)
    for i, task in enumerate(tasks_in):
        new_state_id = f"state_spec_{i + 1}"
        is_terminal = i == num_tasks - 1
        if is_terminal:
            # The terminal state of the chain reaches the last task's reach_target_on_success.
            reach_target = _transition_for(task).reach_target_on_success
        else:
            # While task i+1 exists, the next state of the chain reaches its subject.
            reach_target = _transition_for(tasks_in[i + 1]).subject
        states.append(_successor_state(states[-1], new_state_id, task, embodiment_id, reach_target, is_terminal))
        out_tasks.append({
            "id": task.id,
            "type": task.type,
            "initial_state_spec_id": f"state_spec_{i}",
            "success_state_spec_id": new_state_id,
            "task_args": task.task_args,
        })

    return {
        "env_name": env_name or spec.env_name,
        "nodes": nodes,
        "tasks": out_tasks,
        "state_specs": states,
    }


def _embodiment_id(nodes: list[dict[str, Any]]) -> str:
    ids = [node["id"] for node in nodes if node["type"] == "embodiment"]
    assert ids, "unresolved graph has no embodiment node"
    return ids[0]


def _transition_for(task: ArenaEnvGraphTaskSpec) -> TaskTransition:
    """Resolve a task entry to its ``TaskTransition`` via the ``Task`` class."""
    task_cls = TaskRegistry().get_task_by_name(task.type)
    return task_cls.success_state_transition(task.task_args)


def _relocations(task: ArenaEnvGraphTaskSpec) -> list[Relocate]:
    """Return the task's ``Relocate`` effects."""
    transition = _transition_for(task)
    for effect in transition.effects:
        if not isinstance(effect, Relocate):
            raise NotImplementedError(f"Effect {effect} is not yet supported.")
    return list(transition.effects)


def _successor_state(
    prev_state: dict[str, Any],
    new_state_id: str,
    task: ArenaEnvGraphTaskSpec,
    embodiment_id: str,
    reach_target: str | None,
    is_terminal: bool,
) -> dict[str, Any]:
    """Apply ``task``'s success condition to ``prev_state`` to produce the next state.

    Carries every constraint forward (re-prefixing its id to ``new_state_id``) except each
    relocated object's old placement, which is replaced by what success condition implies.
    The terminal state additionally drops spawn-pose constraints, keeping only
    structural relations.
    """
    relocations = _relocations(task)
    subjects = {relocation.subject for relocation in relocations}
    spatial: list[dict[str, Any]] = []
    for constraint in prev_state.get("spatial_constraints", []):
        is_spawn_pose = spatial_constraint_is_spawn_pose(constraint["type"])
        # Drop a relocated subject's old placement (replaced by the success effects below).
        replaces_old_placement = constraint.get("child") in subjects or (
            constraint["parent"] in subjects and is_spawn_pose
        )
        # Drop spawn-pose constraints in the terminal state.
        drops_spawn_pose = is_terminal and is_spawn_pose
        if not (replaces_old_placement or drops_spawn_pose):
            # Carry forward the constraint that is not replaced by the success effects.
            carried = dict(constraint)
            carried["id"] = _reprefix_id(constraint["id"], prev_state["id"], new_state_id)
            spatial.append(carried)
    # Spatial constraints implied by the task's success effects.
    for relocation in relocations:
        spatial.append({
            "id": f"{new_state_id}_{relocation.subject}_{relocation.relation}_{relocation.target}",
            "type": relocation.relation,
            "parent": relocation.target,
            "child": relocation.subject,
        })
    # Reachability of subject node for the embodiment.
    task_constraints = (
        []
        if reach_target is None
        else [{
            "id": f"{new_state_id}_{embodiment_id.split('_')[0]}_reach_{reach_target}",
            "type": "reach",
            "parent": embodiment_id,
            "child": reach_target,
        }]
    )
    return {"id": new_state_id, "spatial_constraints": spatial, "task_constraints": task_constraints}


def _reprefix_id(old_id: str, old_prefix: str, new_prefix: str) -> str:
    """Swap a constraint id's ``{state_id}`` prefix so carried constraints stay uniquely named."""
    assert old_id.startswith(old_prefix), f"constraint id {old_id!r} is not prefixed by its state id {old_prefix!r}"
    return new_prefix + old_id[len(old_prefix) :]
