# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any

from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    ArenaEnvGraphTaskSpec,
)


def build_arena_env_from_graph_spec(
    spec: ArenaEnvGraphSpec,
    state_spec_id: str | None = None,
    task_factories: dict[str, Callable[[ArenaEnvGraphTaskSpec, dict[str, Any]], Any]] | None = None,
) -> Any:
    """Build an IsaacLabArenaEnvironment from the parsed graph spec."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    state_spec = _select_state_spec(spec, spec.tasks, state_spec_id)

    assets_by_id = _instantiate_node_assets(spec.nodes, AssetRegistry())
    if state_spec is not None:
        _apply_state_spatial_constraints(state_spec, assets_by_id)

    embodiment = _select_embodiment(spec.nodes, assets_by_id)
    scene_assets = [assets_by_id[node.id] for node in spec.nodes if node.type != ArenaEnvGraphNodeType.EMBODIMENT]
    task = _build_task_or_sequence(spec.tasks, assets_by_id, task_factories or {})

    return IsaacLabArenaEnvironment(
        name=spec.env_name,
        scene=Scene(assets=scene_assets),
        embodiment=embodiment,
        task=task,
    )


def _select_state_spec(
    spec: ArenaEnvGraphSpec,
    task_specs: list[ArenaEnvGraphTaskSpec],
    state_spec_id: str | None,
) -> ArenaEnvGraphStateSpec | None:
    """Select the scene state to materialize before task construction."""
    if state_spec_id is None and task_specs:
        state_spec_id = task_specs[0].initial_state_spec_id
    if state_spec_id is None:
        if not spec.state_specs:
            return None
        assert len(spec.state_specs) == 1, (
            "state_spec_id is required when converting a graph spec without a selected task "
            "and with multiple state specs"
        )
        return spec.state_specs[0]
    state_spec = spec.state_specs_by_id.get(state_spec_id)
    assert state_spec is not None, f"Unknown state spec id '{state_spec_id}'"
    return state_spec


def _instantiate_node_assets(nodes: list[ArenaEnvGraphNodeSpec], asset_registry: Any) -> dict[str, Any]:
    """Instantiate registry assets and then bind object-reference nodes."""
    from isaaclab_arena.assets.object_reference import ObjectReference

    assets_by_id: dict[str, Any] = {}
    for node in nodes:
        if node.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            continue
        asset_cls = asset_registry.get_asset_by_name(node.name)
        kwargs = dict(node.params)
        if (
            node.id != node.name
            and "instance_name" not in kwargs
            and _explicitly_accepts_kwarg(asset_cls, "instance_name")
        ):
            kwargs["instance_name"] = node.id
        assets_by_id[node.id] = asset_cls(**kwargs)

    for node in nodes:
        if node.type != ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            continue
        assert isinstance(node, ArenaEnvGraphObjectReferenceNodeSpec)
        params = dict(node.params)
        reserved_keys = {"name", "prim_path", "parent_asset", "object_type"}
        overlapping_keys = sorted(reserved_keys.intersection(params))
        assert (
            not overlapping_keys
        ), f"Object reference node '{node.id}' params override reserved fields: {overlapping_keys}"
        parent_asset = assets_by_id[node.parent]
        assets_by_id[node.id] = ObjectReference(
            name=node.name,
            prim_path=node.prim_path,
            parent_asset=parent_asset,
            object_type=node.object_type,
            **params,
        )

    return assets_by_id


def _explicitly_accepts_kwarg(callable_obj: Any, name: str) -> bool:
    """Return whether a callable signature explicitly declares a keyword."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _select_embodiment(nodes: list[ArenaEnvGraphNodeSpec], assets_by_id: dict[str, Any]) -> Any | None:
    """Return the single embodiment asset, if the graph defines one."""
    embodiment_nodes = [node for node in nodes if node.type == ArenaEnvGraphNodeType.EMBODIMENT]
    assert len(embodiment_nodes) <= 1, "Only one embodiment node can be converted to an IsaacLabArenaEnvironment"
    if not embodiment_nodes:
        return None
    return assets_by_id[embodiment_nodes[0].id]


def _apply_state_spatial_constraints(state_spec: ArenaEnvGraphStateSpec, assets_by_id: dict[str, Any]) -> None:
    """Apply supported spatial constraints to instantiated assets."""
    from isaaclab_arena.relations.relations import (
        AtPosition,
        IsAnchor,
        NextTo,
        On,
        PositionLimits,
        RandomAroundSolution,
        RotateAroundSolution,
        Side,
    )
    from isaaclab_arena.utils.pose import Pose

    for constraint in state_spec.spatial_constraints:
        params = dict(constraint.params)
        parent_asset = assets_by_id[constraint.parent]

        if constraint.type == ArenaEnvGraphSpatialConstraintType.IS_ANCHOR:
            _add_relation(parent_asset, IsAnchor(), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.NEXT_TO:
            if isinstance(params.get("side"), str):
                params["side"] = Side(params["side"])
            _add_relation(_child_asset(constraint, assets_by_id), NextTo(parent_asset, **params), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.ON:
            _add_relation(_child_asset(constraint, assets_by_id), On(parent_asset, **params), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSE:
            position_xyz = params.pop("position_xyz", None)
            rotation_xyzw = params.pop("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))
            assert position_xyz is not None, f"at_pose constraint '{constraint.id}' requires params.position_xyz"
            assert not params, f"Unsupported at_pose params for constraint '{constraint.id}': {sorted(params)}"
            _set_initial_pose(
                parent_asset,
                Pose(position_xyz=tuple(position_xyz), rotation_xyzw=tuple(rotation_xyzw)),
                constraint,
            )
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSITION:
            if "position_xyz" in params:
                position_xyz = params.pop("position_xyz")
                params.setdefault("x", position_xyz[0])
                params.setdefault("y", position_xyz[1])
                params.setdefault("z", position_xyz[2])
            _add_relation(parent_asset, AtPosition(**params), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.POSITION_LIMITS:
            _add_relation(parent_asset, PositionLimits(**params), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.RANDOM_AROUND_SOLUTION:
            _add_relation(parent_asset, RandomAroundSolution(**params), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.ROTATE_AROUND_SOLUTION:
            _add_relation(parent_asset, RotateAroundSolution(**params), constraint)
        elif constraint.type == ArenaEnvGraphSpatialConstraintType.IN:
            raise NotImplementedError(
                f"Spatial constraint type '{constraint.type.value}' is not supported for initial state conversion"
            )
        else:
            raise NotImplementedError(f"Unsupported spatial constraint type '{constraint.type.value}'")


def _child_asset(constraint: ArenaEnvGraphSpatialConstraintSpec, assets_by_id: dict[str, Any]) -> Any:
    """Return a constraint child asset or fail with the constraint id."""
    assert constraint.child is not None, f"Constraint '{constraint.id}' requires a child node"
    return assets_by_id[constraint.child]


def _add_relation(asset: Any, relation: Any, constraint: ArenaEnvGraphSpatialConstraintSpec) -> None:
    """Attach a relation to an asset with graph-context validation."""
    assert hasattr(
        asset, "add_relation"
    ), f"Constraint '{constraint.id}' targets node '{constraint.parent}', which cannot accept relations"
    asset.add_relation(relation)


def _set_initial_pose(asset: Any, pose: Any, constraint: ArenaEnvGraphSpatialConstraintSpec) -> None:
    """Apply an at_pose constraint as an asset initial pose."""
    assert constraint.child is None, f"at_pose constraint '{constraint.id}' must not define a child node"
    assert hasattr(
        asset, "set_initial_pose"
    ), f"Constraint '{constraint.id}' targets node '{constraint.parent}', which cannot set an initial pose"
    asset.set_initial_pose(pose)


def _build_task_or_sequence(
    task_specs: list[ArenaEnvGraphTaskSpec],
    assets_by_id: dict[str, Any],
    task_factories: dict[str, Callable[[ArenaEnvGraphTaskSpec, dict[str, Any]], Any]],
) -> Any | None:
    """Build no task, one task, or a sequential composite task."""
    if not task_specs:
        return None

    subtasks = [_build_task(task_spec, assets_by_id, task_factories) for task_spec in task_specs]
    if len(subtasks) == 1:
        return subtasks[0]

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    return SequentialTaskBase(
        subtasks=subtasks,
        desired_subtask_success_state=[True for _ in subtasks],
    )


def _build_task(
    task_spec: ArenaEnvGraphTaskSpec,
    assets_by_id: dict[str, Any],
    task_factories: dict[str, Callable[[ArenaEnvGraphTaskSpec, dict[str, Any]], Any]],
) -> Any:
    """Build one concrete task from a graph task spec."""
    if task_spec.type in task_factories:
        return task_factories[task_spec.type](task_spec, assets_by_id)

    task_cls = _resolve_task_class(task_spec.type)
    task_args = _resolve_task_args(task_spec.task_args, assets_by_id)
    return task_cls(**_align_task_kwargs(task_cls, task_args))


def _resolve_task_class(task_type: str) -> Any:
    """Resolve a task type by import path, task class name, or task module stem."""
    task_cls = _import_symbol(task_type)
    if task_cls is not None:
        return task_cls

    matches = _discover_task_classes(task_type)
    assert matches, (
        f"Unknown task type '{task_type}'. Use a TaskBase subclass name, a tasks module stem, "
        "an import path, or task_factories."
    )
    assert len(matches) == 1, (
        f"Task type '{task_type}' matched multiple task classes: "
        f"{[match.__module__ + ':' + match.__name__ for match in matches]}"
    )
    return matches[0]


def _import_symbol(import_path: str) -> Any | None:
    """Import a symbol when the task type is an import path."""
    if ":" in import_path:
        module_name, class_name = import_path.split(":", 1)
    elif "." in import_path:
        module_name, class_name = import_path.rsplit(".", 1)
    else:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _discover_task_classes(task_type: str) -> list[type]:
    """Find matching first-party TaskBase subclasses at runtime."""
    from isaaclab_arena.tasks.task_base import TaskBase

    import isaaclab_arena.tasks as tasks_package

    requested = _normalize_task_class_name(task_type)
    direct_matches = _discover_task_classes_from_modules(
        _task_module_candidates(task_type, tasks_package.__name__),
        requested,
        TaskBase,
    )
    if direct_matches:
        return direct_matches

    module_names = [
        module_info.name for module_info in pkgutil.walk_packages(tasks_package.__path__, tasks_package.__name__ + ".")
    ]
    return _discover_task_classes_from_modules(
        module_names,
        requested,
        TaskBase,
    )


def _discover_task_classes_from_modules(module_names: list[str], requested: str, task_base_cls: type) -> list[type]:
    """Find task class/module matches in the supplied module names."""
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
        for task_cls in task_classes:
            if _normalize_task_class_name(task_cls.__name__) == requested:
                class_matches.append(task_cls)

        module_stem = module_name.rsplit(".", 1)[-1]
        if _normalize_task_module_name(module_stem) == requested:
            module_matches.extend(task_classes)

    if class_matches:
        return class_matches
    return module_matches


def _task_module_candidates(task_type: str, package_name: str) -> list[str]:
    """Return likely task module import paths before package-wide scanning."""
    snake_task_type = _camel_to_snake(task_type)
    module_stems = [snake_task_type]
    if not snake_task_type.endswith("_task"):
        module_stems.append(f"{snake_task_type}_task")
    return [f"{package_name}.{module_stem}" for module_stem in module_stems]


def _get_module_task_classes(module: Any, task_base_cls: type) -> list[type]:
    """Return TaskBase subclasses defined directly in a module."""
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls is not task_base_cls and issubclass(cls, task_base_cls) and cls.__module__ == module.__name__
    ]


def _resolve_task_args(value: Any, assets_by_id: dict[str, Any]) -> Any:
    """Recursively replace node id strings in task args with asset objects."""
    if isinstance(value, str) and value in assets_by_id:
        return assets_by_id[value]
    if isinstance(value, list):
        return [_resolve_task_args(item, assets_by_id) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_task_args(item, assets_by_id) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_task_args(item, assets_by_id) for key, item in value.items()}
    return value


def _align_task_kwargs(task_cls: type, task_args: dict[str, Any]) -> dict[str, Any]:
    """Align YAML task arg names to constructor parameter names."""
    parameters = inspect.signature(task_cls).parameters
    constructor_params = {
        name: parameter
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required_params = {
        name for name, parameter in constructor_params.items() if parameter.default is inspect.Parameter.empty
    }

    kwargs: dict[str, Any] = {}
    for key, value in task_args.items():
        parameter_name = _match_task_arg_to_parameter(key, constructor_params, required_params, set(kwargs))
        kwargs[parameter_name] = value
    return kwargs


def _match_task_arg_to_parameter(
    key: str,
    parameters: dict[str, inspect.Parameter],
    required_params: set[str],
    assigned_params: set[str],
) -> str:
    """Match one YAML task arg to one constructor parameter."""
    if key in parameters and key not in assigned_params:
        return key

    normalized_key = _normalize_identifier(key)
    candidates = [
        name
        for name in parameters
        if name not in assigned_params
        and (
            _normalize_identifier(name) == normalized_key
            or normalized_key in _normalize_identifier(name)
            or _normalize_identifier(name) in normalized_key
        )
    ]
    required_candidates = [name for name in candidates if name in required_params]
    if required_candidates:
        candidates = required_candidates
    assert len(candidates) == 1, (
        f"Task arg '{key}' does not map cleanly to constructor parameters. "
        f"Candidates: {candidates}. Use the constructor parameter name or task_factories."
    )
    return candidates[0]


def _normalize_task_class_name(class_name: str) -> str:
    """Normalize a task class name for graph task-type matching."""
    return _strip_suffix(_normalize_identifier(class_name), "task")


def _normalize_task_module_name(module_name: str) -> str:
    """Normalize a task module name for graph task-type matching."""
    return _strip_suffix(_normalize_identifier(module_name), "task")


def _normalize_identifier(identifier: str) -> str:
    """Drop separators and case for loose identifier matching."""
    return "".join(char for char in identifier.lower() if char.isalnum())


def _camel_to_snake(identifier: str) -> str:
    """Convert CamelCase-ish identifiers into snake_case-ish module stems."""
    chars: list[str] = []
    for index, char in enumerate(identifier):
        if char.isupper() and index > 0 and (identifier[index - 1].islower() or identifier[index - 1].isdigit()):
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _strip_suffix(value: str, suffix: str) -> str:
    """Remove a suffix when present."""
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value
