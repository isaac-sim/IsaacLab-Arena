# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING, Any

import isaaclab_arena.tasks as tasks_package
from isaaclab_arena.environments.graph_spec_utils import map_nested_leaf_values
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.task_base import TaskBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphTaskSpec


def build_task_or_sequence(task_specs: list[ArenaEnvGraphTaskSpec], assets_by_id: dict[str, Any]) -> Any | None:
    """Build the env's task from the graph's task specs.

    Returns None if there are no specs, a single task instance if there's one,
    or a SequentialTaskBase wrapping all of them in order.
    """
    if not task_specs:
        return None
    subtasks = [_build_task(task_spec, assets_by_id) for task_spec in task_specs]
    if len(subtasks) == 1:
        return subtasks[0]
    return SequentialTaskBase(subtasks=subtasks, desired_subtask_success_state=[True for _ in subtasks])


def _build_task(task_spec: ArenaEnvGraphTaskSpec, assets_by_id: dict[str, Any]) -> Any:
    """Build one task instance, swapping node-id strings in its args for the live asset objects.

    Upstream contract: `task_args` keys are emitted in the task constructor's exact parameter
    names — no loose / case-insensitive matching is performed here.
    """
    task_cls = _resolve_task_class(task_spec.type)
    task_args = _resolve_task_args(task_spec.task_args, assets_by_id)
    return task_cls(**task_args)


def _resolve_task_class(task_type: str) -> Any:
    """Resolve a YAML ``type:`` string to a concrete task class.

    Two forms are supported:
      * Qualified import path — ``pkg.module:Class`` or ``pkg.module.Class``.
      * Bare class name — searched for in the tasks package by exact ``__name__``.
    """
    task_cls = _import_symbol(task_type)
    if task_cls is not None:
        return task_cls
    matches = _discover_task_classes(task_type)
    assert matches, f"Unknown task type '{task_type}'. Use the exact TaskBase subclass name or an import path."
    assert len(matches) == 1, (
        f"Task type '{task_type}' matched multiple task classes: "
        f"{[match.__module__ + ':' + match.__name__ for match in matches]}"
    )
    return matches[0]


def _import_symbol(import_path: str) -> Any | None:
    """Import a qualified ``pkg.module:Class`` (or dotted) path.

    Returns None for bare class names; those go through `_discover_task_classes` instead.
    """
    if ":" in import_path:
        module_name, class_name = import_path.split(":", 1)
    elif "." in import_path:
        module_name, class_name = import_path.rsplit(".", 1)
    else:
        return None
    return getattr(importlib.import_module(module_name), class_name)


def _discover_task_classes(task_type: str) -> list[type]:
    """Walk the tasks package looking for a class named exactly ``task_type``.

    Upstream contract: ``task_type`` is the exact class name (e.g. ``PickAndPlaceTask``).
    """
    matches: list[type] = []
    for module_info in pkgutil.walk_packages(tasks_package.__path__, tasks_package.__name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
        except ModuleNotFoundError as exc:
            if exc.name == module_info.name:
                continue
            raise
        for cls in _get_module_task_classes(module, TaskBase):
            if cls.__name__ == task_type:
                matches.append(cls)
    return matches


def _get_module_task_classes(module: Any, task_base_cls: type) -> list[type]:
    """TaskBase subclasses defined in this module (re-exported imports are skipped)."""
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls is not task_base_cls and issubclass(cls, task_base_cls) and cls.__module__ == module.__name__
    ]


def _resolve_task_args(value: Any, assets_by_id: dict[str, Any]) -> Any:
    """Swap node-id strings in task args for the live asset objects they refer to."""
    return map_nested_leaf_values(
        value,
        lambda item: assets_by_id[item] if isinstance(item, str) and item in assets_by_id else item,
    )
