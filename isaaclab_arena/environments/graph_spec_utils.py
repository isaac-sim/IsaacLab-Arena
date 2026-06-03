# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from numbers import Real
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphSpatialConstraintType
    from isaaclab_arena.relations.relations import RelationBase


def coerce_number_sequence(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    """Coerce a fixed-length numeric list or tuple (e.g. position or quaternion)."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Field '{field_name}' must contain {length} numbers")
    if len(value) != length:
        raise ValueError(f"Field '{field_name}' must contain {length} numbers")
    if not all(isinstance(item, Real) and not isinstance(item, bool) for item in value):
        raise ValueError(f"Field '{field_name}' must contain only numbers")
    return tuple(float(item) for item in value)


def validate_unique_ids(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
    """Ensure every graph id is unique, including constraint ids inside states."""
    id_locations: dict[str, list[str]] = {}
    for node in nodes:
        _add_id_location(id_locations, node.id, f"node '{node.id}'")
    for task in tasks:
        _add_id_location(id_locations, task.id, f"task '{task.id}'")
    for state_spec in state_specs:
        _add_id_location(id_locations, state_spec.id, f"state spec '{state_spec.id}'")
        for constraint in state_spec.spatial_constraints:
            _add_id_location(id_locations, constraint.id, f"spatial constraint '{constraint.id}'")
        for constraint in state_spec.task_constraints:
            _add_id_location(id_locations, constraint.id, f"task constraint '{constraint.id}'")

    duplicates = {spec_id: locations for spec_id, locations in id_locations.items() if len(locations) > 1}
    if duplicates:
        raise ValueError(f"Duplicate env graph ids found: {duplicates}")


def validate_references_exist(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
    """Ensure every graph reference points to a node or state spec that exists."""
    node_ids = {node.id for node in nodes}
    state_spec_ids = {state_spec.id for state_spec in state_specs}

    seen_node_ids: set[str] = set()
    for node in nodes:
        parent = getattr(node, "parent", None)
        if parent is not None:
            if parent not in node_ids:
                raise ValueError(f"Node '{node.id}' references unknown parent '{parent}'")
            if parent not in seen_node_ids:
                raise ValueError(
                    f"Node '{node.id}' references parent '{parent}' defined later in the node list; "
                    "a parent must appear before any node that references it"
                )
        seen_node_ids.add(node.id)

    for task in tasks:
        for label, state_spec_id in (
            ("initial_state_spec_id", task.initial_state_spec_id),
            ("success_state_spec_id", task.success_state_spec_id),
        ):
            if state_spec_id not in state_spec_ids:
                raise ValueError(f"Task '{task.id}' references unknown state spec '{state_spec_id}' for '{label}'")

    for state_spec in state_specs:
        for constraint in state_spec.spatial_constraints:
            if constraint.parent not in node_ids:
                raise ValueError(f"Constraint '{constraint.id}' references unknown parent node '{constraint.parent}'")
            if constraint.child is not None and constraint.child not in node_ids:
                raise ValueError(f"Constraint '{constraint.id}' references unknown child node '{constraint.child}'")

        for constraint in state_spec.task_constraints:
            if constraint.parent is not None and constraint.parent not in node_ids:
                raise ValueError(f"Constraint '{constraint.id}' references unknown parent node '{constraint.parent}'")
            if constraint.child is not None and constraint.child not in node_ids:
                raise ValueError(f"Constraint '{constraint.id}' references unknown child node '{constraint.child}'")


def validate_spatial_constraint_shapes(state_specs: list[Any]) -> None:
    """Check each spatial constraint has the parent/child shape its relation expects."""
    for state_spec in state_specs:
        for constraint in state_spec.spatial_constraints:
            constraint_type = _enum_value(constraint.type)
            if constraint_type == "at_pose":
                if "position_xyz" not in constraint.params:
                    raise ValueError(
                        f"Spatial constraint '{constraint.id}' of type 'at_pose' requires params.position_xyz"
                    )
                is_unary = True
            elif constraint_type == "in":
                is_unary = False
            else:
                relation_cls = relation_class_for_spatial_constraint_type(constraint.type)
                if relation_cls is None:
                    raise ValueError(f"Spatial constraint type '{constraint_type}' is not mapped to a relation class")
                is_unary = relation_cls.is_unary()

            if is_unary:
                if constraint.child is not None:
                    raise ValueError(
                        f"Spatial constraint '{constraint.id}' of type '{constraint_type}' must not define a child node"
                    )
            elif constraint.child is None:
                raise ValueError(
                    f"Spatial constraint '{constraint.id}' of type '{constraint_type}' requires a child node"
                )


def _add_id_location(id_locations: dict[str, list[str]], spec_id: str, location: str) -> None:
    id_locations.setdefault(spec_id, []).append(location)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def relation_class_for_spatial_constraint_type(
    constraint_type: "ArenaEnvGraphSpatialConstraintType",
) -> "type[RelationBase] | None":
    """Resolve a spatial-constraint enum member to its RelationBase subclass.

    Returns None for enum members that have no registered class yet (e.g. AT_POSE,
    handled via set_initial_pose; IN, not yet supported by the solver).
    # TODO(xinjieyao, 2026-05-28): add support for AT_POSE and IN.
    """
    registry = ObjectRelationLibraryRegistry()
    if registry.is_registered(constraint_type.value):
        return registry.get_object_relation_by_name(constraint_type.value)
    return None


def iter_nested_leaf_values(value: Any, key_path: str = "") -> Iterator[tuple[str, Any]]:
    """Walk nested task-arg values while keeping a readable path for errors.

    Example:
        >>> list(iter_nested_leaf_values({"object": "mug", "destination": ["table", "shelf"]}))
        [('object', 'mug'), ('destination[0]', 'table'), ('destination[1]', 'shelf')]
    """
    if isinstance(value, dict):
        for key, item in value.items():
            nested_key_path = f"{key_path}.{key}" if key_path else str(key)
            yield from iter_nested_leaf_values(item, nested_key_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested_key_path = f"{key_path}[{index}]" if key_path else f"[{index}]"
            yield from iter_nested_leaf_values(item, nested_key_path)
    else:
        yield key_path, value


def map_nested_leaf_values(value: Any, transform: Callable[[Any], Any]) -> Any:
    """Apply a transform to nested task-arg leaves while preserving container shape.

    Example:
        >>> map_nested_leaf_values({"a": [1, 2], "b": (3, 4)}, lambda x: x * 10)
        {'a': [10, 20], 'b': (30, 40)}
    """
    if isinstance(value, dict):
        return {key: map_nested_leaf_values(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [map_nested_leaf_values(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(map_nested_leaf_values(item, transform) for item in value)
    return transform(value)


def normalize_identifier(identifier: str) -> str:
    """Normalize names so YAML keys can be matched across casing and separators.

    Example:
        >>> normalize_identifier("Pickup_Object")
        'pickupobject'
    """
    return "".join(char for char in identifier.lower() if char.isalnum())


def camel_to_snake(identifier: str) -> str:
    """Turn a class-like name into the module-style name we try during discovery.

    Example:
        >>> camel_to_snake("AtPosition")
        'at_position'
    """
    chars: list[str] = []
    for index, char in enumerate(identifier):
        if char.isupper() and index > 0 and (identifier[index - 1].islower() or identifier[index - 1].isdigit()):
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def strip_suffix(value: str, suffix: str) -> str:
    """Remove a suffix only when the value actually has it.

    Example:
        >>> strip_suffix("AtPositionSpec", "Spec")
        'AtPosition'
        >>> strip_suffix("AtPosition", "Spec")
        'AtPosition'
    """
    return value[: -len(suffix)] if value.endswith(suffix) else value
