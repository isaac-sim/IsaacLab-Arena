# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
import typing
from typing import TYPE_CHECKING, Any

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.registries import TaskRegistry

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphTaskSpec


# Annotation bases that mark a task __init__ kwarg as a graph-node reference.
#   * Asset           — direct ("background_scene: Asset")
#   * AffordanceBase  — task interface enforces an affordance contract on the kwarg
#                       ("placeable_object: Placeable").
NODE_REF_BASES: tuple[type, ...] = (Asset, AffordanceBase)


def build_task_from_specs(task_specs: list[ArenaEnvGraphTaskSpec], assets_by_node_id: dict[str, Any]) -> Any | None:
    """Build each task spec into a live task and combine them into one env-level task.

    None for no specs, the sole task for one, or a SequentialTaskBase (all required to succeed) for many.
    """
    task_instances = [_build_task_from_spec(spec, assets_by_node_id) for spec in task_specs]
    if not task_instances:
        return None
    if len(task_instances) == 1:
        return task_instances[0]
    # Lazy import: SequentialTaskBase -> composite_task_base pulls in pxr (USD), which requires a
    # launched SimulationApp. Deferring it keeps this module importable by data-only consumers
    # (spec parsers, unit tests, pytest collection) without dragging in sim deps at import time.
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    return SequentialTaskBase(
        subtasks=task_instances,
        desired_subtask_success_state=[True] * len(task_instances),
    )


def _build_task_from_spec(task_spec: ArenaEnvGraphTaskSpec, assets_by_node_id: dict[str, Any]) -> Any:
    """Look up the task class by name, resolve any Asset-typed kwargs, instantiate."""
    task_class = TaskRegistry().get_task_by_name(task_spec.type)
    task_init_kwargs = _resolve_node_refs_in_task_args(task_class, task_spec.task_args, assets_by_node_id)
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
    node_ref_params: dict[str, bool] = {}
    for param_name, annotation in typing.get_type_hints(task_class.__init__).items():
        is_collection = _classify_node_ref(annotation)
        # Keep only node refs.
        if is_collection is not None:
            node_ref_params[param_name] = is_collection
    return node_ref_params


def _classify_node_ref(annotation: Any) -> bool | None:
    """Match node-ref type to a bool: False=scalar, True=list, None=not a ref. e.g. ``list[Asset] | None`` -> True."""
    # Look at non-None members of a union, like list[Asset] | None.
    for branch in _strip_none(annotation):
        # For a scalar ref: the branch is itself an Asset / AffordanceBase subclass. e.g. Asset.
        if _is_node_ref_type(branch):
            return False
        # For a list ref: list[X] with X a ref. e.g. list[Asset].
        if typing.get_origin(branch) is list and _is_node_ref_type(next(iter(typing.get_args(branch)), None)):
            return True
    return None


def _is_node_ref_type(annotation: Any) -> bool:
    """True if annotation is an Asset / AffordanceBase subclass. e.g. Asset -> True, list[Asset] / None -> False."""
    # The isinstance(..., type) guard rejects non-classes (None, generics like list[Asset]) so issubclass won't raise.
    return isinstance(annotation, type) and issubclass(annotation, NODE_REF_BASES)


def _strip_none(annotation: Any) -> tuple[Any, ...]:
    """Non-None members of a union, else the annotation alone. e.g. ``Asset | None`` -> ``(Asset,)``; ``float`` -> ``(float,)``."""
    # Only unions branch; everything else is wrapped in a 1-tuple so callers iterate uniformly.
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return tuple(member for member in typing.get_args(annotation) if member is not type(None))
    return (annotation,)
