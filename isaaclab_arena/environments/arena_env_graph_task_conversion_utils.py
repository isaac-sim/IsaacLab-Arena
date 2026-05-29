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
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphTaskSpec


# Annotation bases that mark a task __init__ kwarg as a graph-node reference.
#   * Asset           — direct ("background_scene: Asset")
#   * AffordanceBase  — task interface enforces an affordance contract on the kwarg
#                       ("placeable_object: Placeable").
NODE_REF_BASES: tuple[type, ...] = (Asset, AffordanceBase)


def build_task_from_specs(task_specs: list[ArenaEnvGraphTaskSpec], assets_by_node_id: dict[str, Any]) -> Any | None:
    """Return None for no specs, the single task for one spec, or a SequentialTaskBase for many."""
    if not task_specs:
        return None
    task_instances = [_build_task_from_spec(task_spec, assets_by_node_id) for task_spec in task_specs]
    if len(task_instances) == 1:
        return task_instances[0]
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

    Misspelled / non-string node ids raise AssertionError instead of silently passing through.
    """
    # Introspect __init__ once — the task class is the single source of truth for which
    # params come from graph nodes. `None` from .get() below means "not a node-ref param".
    is_collection_by_param_name = find_node_ref_params_in_signature(task_class)

    resolved_task_kwargs: dict[str, Any] = {}
    for param_name, raw_param_value in raw_task_args.items():
        is_collection = is_collection_by_param_name.get(param_name)
        if is_collection is None:
            # Not annotated as a node ref (e.g. floats, strings, tuples) — forward unchanged.
            resolved_task_kwargs[param_name] = raw_param_value
        elif is_collection:
            # list[Asset]-typed param: resolve each element to its live asset.
            resolved_task_kwargs[param_name] = [
                _lookup_asset_by_node_id(raw_node_id, assets_by_node_id, task_class, param_name)
                for raw_node_id in raw_param_value
            ]
        else:
            # Asset-typed param: resolve the single node id to its live asset.
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
    """Return ``{param_name: is_collection}`` for ``__init__`` params annotated as a NODE_REF_BASES subclass.

    Optional / ``X | None`` counts as a node ref. ``is_collection=True`` for ``list[X]``
    (``tuple[X, ...]`` is intentionally unsupported — no task uses it). Single source of
    truth for what's a graph-node ref; also consumable by validators / YAML generators.
    """
    is_collection_by_param_name: dict[str, bool] = {}
    # get_type_hints resolves stringified / forward-ref annotations into real classes so issubclass works.
    # e.g. `pick_up_object: "Asset"` (a str under `from __future__ import annotations`) becomes the Asset class.
    for param_name, param_annotation in typing.get_type_hints(task_class.__init__).items():
        # Skip the implicit `self` slot and any `return` annotation — neither is a kwarg.
        if param_name not in ("self", "return"):
            # Walk Union members so `Asset | None` is recognized via its Asset branch.
            # First matching branch wins; later branches in the same param are ignored.
            for annotation_branch in _strip_none(param_annotation):
                # Scalar node ref: annotation is itself an Asset / AffordanceBase subclass.
                if isinstance(annotation_branch, type) and issubclass(annotation_branch, NODE_REF_BASES):
                    is_collection_by_param_name[param_name] = False
                    break
                # Collection node ref: list[X] where X is an Asset / AffordanceBase subclass.
                # The isinstance(args[0], type) guard rejects parametrized generics like list[list[Asset]].
                if typing.get_origin(annotation_branch) is list:
                    list_element_args = typing.get_args(annotation_branch)
                    if (
                        list_element_args
                        and isinstance(list_element_args[0], type)
                        and issubclass(list_element_args[0], NODE_REF_BASES)
                    ):
                        is_collection_by_param_name[param_name] = True
                        break
    return is_collection_by_param_name


def _strip_none(annotation: Any) -> tuple[Any, ...]:
    """Non-None members of a ``X | None`` / ``Optional[X]``; ``(annotation,)`` otherwise."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return tuple(member for member in typing.get_args(annotation) if member is not type(None))
    return (annotation,)
