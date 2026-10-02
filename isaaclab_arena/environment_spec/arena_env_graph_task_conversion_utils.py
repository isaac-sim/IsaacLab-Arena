# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import TaskRegistry

if TYPE_CHECKING:
    from isaaclab_arena.environment_spec.arena_env_graph_types import CompositeTaskSpec, TaskSpec


def build_task_from_spec(task_spec: CompositeTaskSpec, assets_by_node_id: dict[str, Any]) -> Any:
    """Build the root graph task into a live env-level task instance."""
    from isaaclab_arena.environment_spec.arena_env_graph_types import TaskCompositionType

    if task_spec.composition is TaskCompositionType.ATOMIC and task_spec.desired_subtask_success_state is None:
        return _build_atomic_task_from_spec(
            task_spec.subtasks[0],
            assets_by_node_id,
            task_description=task_spec.description,
            episode_length_s=task_spec.episode_length_s,
        )

    subtasks = [_build_atomic_task_from_spec(spec, assets_by_node_id) for spec in task_spec.subtasks]
    # Lazy import: CompositeTaskBase pulls in pxr (USD), which requires a
    # launched SimulationApp. Deferring it keeps this module importable by data-only consumers
    # (spec parsers, unit tests, pytest collection) without dragging in sim deps at import time.
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

    return CompositeTaskBase(
        subtasks=subtasks,
        task_description=task_spec.description,
        subtasks_are_sequential=task_spec.composition is TaskCompositionType.SEQUENTIAL,
        episode_length_s=task_spec.episode_length_s,
        desired_subtask_success_state=task_spec.desired_subtask_success_state,
    )


def _build_atomic_task_from_spec(
    task_spec: TaskSpec,
    assets_by_node_id: dict[str, Any],
    *,
    task_description: str | None = None,
    episode_length_s: float | None = None,
) -> Any:
    """Look up the task class by name, resolve any Asset-typed kwargs, instantiate."""
    task_class = TaskRegistry().get_task_by_name(task_spec.kind)
    task_init_kwargs = _resolve_node_refs_in_task_args(task_class, task_spec.params, assets_by_node_id)
    if task_description and "task_description" not in task_init_kwargs:
        task_init_kwargs["task_description"] = task_description
    if episode_length_s is not None:
        task_init_kwargs["episode_length_s"] = episode_length_s
    return task_class(**task_init_kwargs)


def _resolve_node_refs_in_task_args(
    task_class: type, raw_task_args: dict[str, Any], assets_by_node_id: dict[str, Any]
) -> dict[str, Any]:
    """Swap node-id strings for live assets on Asset / list[Asset] params; pass others through.

    Example — for ``PickAndPlaceTask(pick_up_object: Asset, ..., episode_length_s: float)``::

        raw_task_args     = {"pick_up_object": "cube", ..., "episode_length_s": 5.0}
        assets_by_node_id = {"cube": <Object>, ...}
        # -> {"pick_up_object": <Object: cube>, ..., "episode_length_s": 5.0}

    Misspelled / non-string node ids raise AssertionError.
    """
    # The task class is the single source of truth for which params come from graph nodes.
    # Params absent from this map aren't node refs.
    is_collection_by_param_name = find_node_ref_params_in_signature(task_class)

    # Non-node-ref params (floats, strings, tuples) pass through unchanged; start with the exact raw copy.
    #   e.g. "minimum_height_to_lift": 0.1  ->  "minimum_height_to_lift": 0.1
    resolved_task_kwargs: dict[str, Any] = dict(raw_task_args)
    for param_name, is_collection in is_collection_by_param_name.items():
        if param_name in raw_task_args:
            raw_param_value = raw_task_args[param_name]
            if raw_param_value is None:
                continue
            if is_collection:
                # list[Asset]-typed param: resolve each element to its live asset.
                #   e.g. "targets": ["cube", "ball"]  ->  "targets": [<Object: cube>, <Object: ball>]
                resolved_task_kwargs[param_name] = [
                    _lookup_asset_by_node_id(raw_node_id, assets_by_node_id, task_class, param_name)
                    for raw_node_id in raw_param_value
                ]
            else:
                # Asset-typed param: resolve the single node id to its live asset.
                #   e.g. "pick_up_object": "cube"  ->  "pick_up_object": <Object: cube>
                resolved_task_kwargs[param_name] = _lookup_asset_by_node_id(
                    raw_param_value, assets_by_node_id, task_class, param_name
                )
    return resolved_task_kwargs


def _lookup_asset_by_node_id(node_id: Any, assets_by_node_id: dict[str, Any], task_class: type, param_name: str) -> Any:
    """Return the live asset for ``node_id``; raise AssertionError naming the task/param on miss."""
    assert (
        isinstance(node_id, str) and node_id in assets_by_node_id
    ), f"{task_class.__name__}.{param_name}: unknown node id {node_id!r}"
    return assets_by_node_id[node_id]


def find_node_ref_params_in_signature(task_class: type) -> dict[str, bool]:
    """Map each node-ref ``__init__`` param to is_collection, where True means list[Asset], False means Asset, and None means non-refs.

    e.g. ``(obj: Asset, group: list[Asset], height: float)`` -> ``{"obj": False, "group": True}``.
    """
    from isaaclab_arena.agentic_environment_generation.authoring_metadata import constructor_parameters

    node_ref_params: dict[str, bool] = {}
    for param_name, schema in constructor_parameters(task_class).items():
        for branch in schema.get("anyOf", [schema]):
            if branch.get("x-arena-reference-collection"):
                node_ref_params[param_name] = True
            elif schema.get("x-arena-reference") or branch.get("x-arena-reference"):
                node_ref_params[param_name] = False
    return node_ref_params
