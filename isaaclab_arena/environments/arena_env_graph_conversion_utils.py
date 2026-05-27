# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_spec import (
        ArenaEnvGraphNodeSpec,
        ArenaEnvGraphSpatialConstraintSpec,
        ArenaEnvGraphSpec,
        ArenaEnvGraphStateSpec,
        ArenaEnvGraphTaskSpec,
    )


def build_arena_env_from_graph_spec(
    spec: ArenaEnvGraphSpec,
    state_spec_id: str | None = None,
) -> Any:
    """Build an IsaacLabArenaEnvironment from the parsed graph spec."""
    _validate_graph_spec_for_conversion(spec, state_spec_id)

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphNodeType
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    state_spec = _select_state_spec(spec, spec.tasks, state_spec_id)
    assets_by_id = _instantiate_node_assets(spec.nodes, AssetRegistry())
    if state_spec is not None:
        _apply_state_spatial_constraints(state_spec, assets_by_id)

    return IsaacLabArenaEnvironment(
        name=spec.env_name,
        scene=Scene(
            assets=[assets_by_id[node.id] for node in spec.nodes if node.type != ArenaEnvGraphNodeType.EMBODIMENT]
        ),
        embodiment=_select_embodiment(spec.nodes, assets_by_id),
        task=_build_task_or_sequence(spec.tasks, assets_by_id),
    )


def _validate_graph_spec_for_conversion(spec: ArenaEnvGraphSpec, state_spec_id: str | None) -> None:
    """Validate graph references and relationship shapes before runtime conversion."""
    from isaaclab_arena.environments.graph_spec_utils import assert_references_exist, assert_unique_ids

    assert_unique_ids(spec.nodes, spec.tasks, spec.state_specs)
    assert_references_exist(spec.nodes, spec.tasks, spec.state_specs)
    _validate_spatial_constraint_shapes(spec)
    _validate_task_arg_node_references(spec)

    state_spec = _select_state_spec(spec, spec.tasks, state_spec_id)
    if state_spec is not None:
        _validate_initial_state_supported_for_conversion(state_spec)


def _validate_spatial_constraint_shapes(spec: ArenaEnvGraphSpec) -> None:
    """Validate relationship endpoints and arity."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType

    child_required_types = {
        ArenaEnvGraphSpatialConstraintType.NEXT_TO,
        ArenaEnvGraphSpatialConstraintType.ON,
        ArenaEnvGraphSpatialConstraintType.IN,
    }
    child_forbidden_types = {
        ArenaEnvGraphSpatialConstraintType.IS_ANCHOR,
        ArenaEnvGraphSpatialConstraintType.AT_POSE,
        ArenaEnvGraphSpatialConstraintType.AT_POSITION,
        ArenaEnvGraphSpatialConstraintType.POSITION_LIMITS,
        ArenaEnvGraphSpatialConstraintType.RANDOM_AROUND_SOLUTION,
        ArenaEnvGraphSpatialConstraintType.ROTATE_AROUND_SOLUTION,
    }

    for state_spec in spec.state_specs:
        for constraint in state_spec.spatial_constraints:
            if constraint.type in child_required_types:
                assert constraint.child is not None, (
                    f"Spatial constraint '{constraint.id}' of type '{constraint.type.value}' requires a child node"
                )
            if constraint.type in child_forbidden_types:
                assert constraint.child is None, (
                    f"Spatial constraint '{constraint.id}' of type '{constraint.type.value}' "
                    "must not define a child node"
                )
            if constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSE:
                assert "position_xyz" in constraint.params, (
                    f"Spatial constraint '{constraint.id}' of type 'at_pose' requires params.position_xyz"
                )


def _validate_task_arg_node_references(spec: ArenaEnvGraphSpec) -> None:
    """Validate common task argument fields that reference graph nodes."""
    node_ids = {node.id for node in spec.nodes}
    node_ref_keys = {
        "background",
        "backgroundscene",
        "destination",
        "destinationlocation",
        "destinationobject",
        "object",
        "pickupobject",
    }

    for task in spec.tasks:
        for key, value in task.task_args.items():
            if _normalize_identifier(key) in node_ref_keys:
                _assert_task_arg_node_reference(task.id, key, value, node_ids)


def _assert_task_arg_node_reference(task_id: str, key: str, value: Any, node_ids: set[str]) -> None:
    """Assert a node-reference task arg points to existing node ids."""
    if isinstance(value, str):
        assert value in node_ids, f"Task '{task_id}' arg '{key}' references unknown node '{value}'"
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_task_arg_node_reference(task_id, key, item, node_ids)
    elif isinstance(value, dict):
        for nested_key, item in value.items():
            _assert_task_arg_node_reference(task_id, f"{key}.{nested_key}", item, node_ids)


def _validate_initial_state_supported_for_conversion(state_spec: ArenaEnvGraphStateSpec) -> None:
    """Validate selected initial-state relationships are supported for materialization."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType

    unsupported_types = {ArenaEnvGraphSpatialConstraintType.IN}
    for constraint in state_spec.spatial_constraints:
        assert constraint.type not in unsupported_types, (
            f"Spatial constraint type '{constraint.type.value}' is not supported for initial state conversion"
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
        assert len(spec.state_specs) == 1, "state_spec_id is required when multiple state specs exist"
        return spec.state_specs[0]
    state_spec = spec.state_specs_by_id.get(state_spec_id)
    assert state_spec is not None, f"Unknown state spec id '{state_spec_id}'"
    return state_spec


def _instantiate_node_assets(nodes: list[ArenaEnvGraphNodeSpec], asset_registry: Any) -> dict[str, Any]:
    """Instantiate registry assets and then bind object-reference nodes."""
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.environments.arena_env_graph_spec import (
        ArenaEnvGraphNodeType,
        ArenaEnvGraphObjectReferenceNodeSpec,
    )

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
        assets_by_id[node.id] = ObjectReference(
            name=node.name,
            prim_path=node.prim_path,
            parent_asset=assets_by_id[node.parent],
            object_type=node.object_type,
            **params,
        )
    return assets_by_id


def _explicitly_accepts_kwarg(callable_obj: Any, name: str) -> bool:
    """Return whether a callable signature explicitly declares a keyword."""
    try:
        parameter = inspect.signature(callable_obj).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _select_embodiment(nodes: list[ArenaEnvGraphNodeSpec], assets_by_id: dict[str, Any]) -> Any | None:
    """Return the single embodiment asset, if the graph defines one."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphNodeType

    embodiment_nodes = [node for node in nodes if node.type == ArenaEnvGraphNodeType.EMBODIMENT]
    assert len(embodiment_nodes) <= 1, "Only one embodiment node can be converted to an IsaacLabArenaEnvironment"
    return assets_by_id[embodiment_nodes[0].id] if embodiment_nodes else None


def _apply_state_spatial_constraints(state_spec: ArenaEnvGraphStateSpec, assets_by_id: dict[str, Any]) -> None:
    """Apply supported spatial constraints to instantiated assets."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType
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
                f"Spatial constraint type '{constraint.type.value}' is not supported for initial state"
            )
        else:
            raise NotImplementedError(f"Unsupported spatial constraint type '{constraint.type.value}'")


def _child_asset(constraint: ArenaEnvGraphSpatialConstraintSpec, assets_by_id: dict[str, Any]) -> Any:
    """Return a constraint child asset or fail with the constraint id."""
    assert constraint.child is not None, f"Constraint '{constraint.id}' requires a child node"
    return assets_by_id[constraint.child]


def _add_relation(asset: Any, relation: Any, constraint: ArenaEnvGraphSpatialConstraintSpec) -> None:
    """Attach a relation to an asset with graph-context validation."""
    assert hasattr(asset, "add_relation"), f"Constraint '{constraint.id}' target cannot accept relations"
    asset.add_relation(relation)


def _set_initial_pose(asset: Any, pose: Any, constraint: ArenaEnvGraphSpatialConstraintSpec) -> None:
    """Apply an at_pose constraint as an asset initial pose."""
    assert constraint.child is None, f"at_pose constraint '{constraint.id}' must not define a child node"
    assert hasattr(asset, "set_initial_pose"), f"Constraint '{constraint.id}' target cannot set an initial pose"
    asset.set_initial_pose(pose)


def _build_task_or_sequence(task_specs: list[ArenaEnvGraphTaskSpec], assets_by_id: dict[str, Any]) -> Any | None:
    """Build no task, one task, or a sequential composite task."""
    if not task_specs:
        return None
    subtasks = [_build_task(task_spec, assets_by_id) for task_spec in task_specs]
    if len(subtasks) == 1:
        return subtasks[0]

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    return SequentialTaskBase(subtasks=subtasks, desired_subtask_success_state=[True for _ in subtasks])


def _build_task(task_spec: ArenaEnvGraphTaskSpec, assets_by_id: dict[str, Any]) -> Any:
    """Build one concrete task from a graph task spec."""
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
        f"Unknown task type '{task_type}'. Use a TaskBase subclass name, tasks module stem, or import path."
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
        class_matches.extend(cls for cls in task_classes if _normalize_task_name(cls.__name__) == requested)
        if _normalize_task_name(module_name.rsplit(".", 1)[-1]) == requested:
            module_matches.extend(task_classes)
    return class_matches or module_matches


def _task_module_candidates(task_type: str, package_name: str) -> list[str]:
    """Return likely task module import paths before package-wide scanning."""
    module_stem = _camel_to_snake(task_type)
    module_stems = [module_stem]
    if not module_stem.endswith("_task"):
        module_stems.append(f"{module_stem}_task")
    return [f"{package_name}.{stem}" for stem in module_stems]


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
    candidates = required_candidates or candidates
    assert len(candidates) == 1, (
        f"Task arg '{key}' does not map cleanly to constructor parameters. Candidates: {candidates}."
    )
    return candidates[0]


def _normalize_task_name(name: str) -> str:
    """Normalize a task class or module name for graph task-type matching."""
    return _strip_suffix(_normalize_identifier(name), "task")


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
    return value[: -len(suffix)] if value.endswith(suffix) else value
