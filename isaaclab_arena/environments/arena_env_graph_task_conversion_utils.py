# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.graph_spec_utils import (
    camel_to_snake,
    map_nested_leaf_values,
    normalize_identifier,
    strip_suffix,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphTaskSpec


def build_task_or_sequence(task_specs: list[ArenaEnvGraphTaskSpec], assets_by_id: dict[str, Any]) -> Any | None:
    """Return no task, one task, or a sequential task using every graph task entry."""
    if not task_specs:
        return None
    subtasks = [_build_task(task_spec, assets_by_id) for task_spec in task_specs]
    if len(subtasks) == 1:
        return subtasks[0]

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    return SequentialTaskBase(subtasks=subtasks, desired_subtask_success_state=[True for _ in subtasks])


def _build_task(task_spec: ArenaEnvGraphTaskSpec, assets_by_id: dict[str, Any]) -> Any:
    """Instantiate one task after resolving graph node ids to asset objects."""
    task_cls = _resolve_task_class(task_spec.type)
    task_args = _resolve_task_args(task_spec.task_args, assets_by_id)
    return task_cls(**_align_task_kwargs(task_cls, task_args))


def _resolve_task_class(task_type: str) -> Any:
    """Find the task class named by a graph task type."""
    task_cls = _import_symbol(task_type)
    if task_cls is not None:
        return task_cls
    matches = _discover_task_classes(task_type)
    assert matches, f"Unknown task type '{task_type}'. Use a TaskBase subclass name, tasks module stem, or import path."
    assert len(matches) == 1, (
        f"Task type '{task_type}' matched multiple task classes: "
        f"{[match.__module__ + ':' + match.__name__ for match in matches]}"
    )
    return matches[0]


def _import_symbol(import_path: str) -> Any | None:
    """Load an explicit module path such as package.module:Class."""
    if ":" in import_path:
        module_name, class_name = import_path.split(":", 1)
    elif "." in import_path:
        module_name, class_name = import_path.rsplit(".", 1)
    else:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _discover_task_classes(task_type: str) -> list[type]:
    """Search first-party task modules for a class matching the YAML task type."""
    import isaaclab_arena.tasks as tasks_package
    from isaaclab_arena.tasks.task_base import TaskBase

    requested = _normalize_task_name(task_type)
    candidates = _task_module_candidates(task_type, tasks_package.__name__)
    matches = _discover_task_classes_from_modules(candidates, requested, TaskBase)
    if matches:
        return matches

    module_names = [
        module_info.name for module_info in pkgutil.walk_packages(tasks_package.__path__, tasks_package.__name__ + ".")
    ]
    return _discover_task_classes_from_modules(module_names, requested, TaskBase)


def _discover_task_classes_from_modules(module_names: list[str], requested: str, task_base_cls: type) -> list[type]:
    """Return task classes whose class name or module name matches the request."""
    class_matches: list[type] = []
    module_matches: list[type] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise

        task_classes = _get_module_task_classes(module, task_base_cls)
        class_matches.extend(cls for cls in task_classes if _normalize_task_name(cls.__name__) == requested)
        if _normalize_task_name(module_name.rsplit(".", 1)[-1]) == requested:
            module_matches.extend(task_classes)
    return class_matches or module_matches


def _task_module_candidates(task_type: str, package_name: str) -> list[str]:
    """Try the most likely module names before walking the whole task package."""
    module_stem = camel_to_snake(task_type)
    module_stems = [module_stem]
    if not module_stem.endswith("_task"):
        module_stems.append(f"{module_stem}_task")
    return [f"{package_name}.{stem}" for stem in module_stems]


def _get_module_task_classes(module: Any, task_base_cls: type) -> list[type]:
    """Collect task classes owned by one module, excluding imported classes."""
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls is not task_base_cls and issubclass(cls, task_base_cls) and cls.__module__ == module.__name__
    ]


def _resolve_task_args(value: Any, assets_by_id: dict[str, Any]) -> Any:
    """Replace node-id strings in task args with the instantiated assets they name."""
    return map_nested_leaf_values(
        value,
        lambda item: assets_by_id[item] if isinstance(item, str) and item in assets_by_id else item,
    )


def _align_task_kwargs(task_cls: type, task_args: dict[str, Any]) -> dict[str, Any]:
    """Match YAML task-arg keys to the task constructor's parameter names."""
    parameters = {
        name: parameter
        for name, parameter in inspect.signature(task_cls).parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required = {name for name, parameter in parameters.items() if parameter.default is inspect.Parameter.empty}
    kwargs: dict[str, Any] = {}
    for key, value in task_args.items():
        parameter_name = _match_task_arg_to_parameter(key, parameters, required, set(kwargs))
        kwargs[parameter_name] = value
    return kwargs


def _match_task_arg_to_parameter(
    key: str,
    parameters: dict[str, inspect.Parameter],
    required_params: set[str],
    assigned_params: set[str],
) -> str:
    """Choose the single constructor parameter that best matches one YAML key."""
    if key in parameters and key not in assigned_params:
        return key
    normalized_key = normalize_identifier(key)
    candidates = [
        name
        for name in parameters
        if name not in assigned_params
        and (
            normalize_identifier(name) == normalized_key
            or normalized_key in normalize_identifier(name)
            or normalize_identifier(name) in normalized_key
        )
    ]
    required_candidates = [name for name in candidates if name in required_params]
    candidates = required_candidates or candidates
    assert (
        len(candidates) == 1
    ), f"Task arg '{key}' does not map cleanly to constructor parameters. Candidates: {candidates}."
    return candidates[0]


def _normalize_task_name(name: str) -> str:
    """Put task class names, module names, and YAML names in the same form."""
    return strip_suffix(normalize_identifier(name), "task")
