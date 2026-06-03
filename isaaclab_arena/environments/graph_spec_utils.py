# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphCliOverrideSpec

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType
    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec
    from isaaclab_arena.relations.relations import RelationBase


def assert_dict(data: Any, spec_name: str) -> dict[str, Any]:
    """Require a YAML section to be a mapping before parsing it."""
    assert isinstance(data, dict), f"{spec_name} must be a dict, got {type(data).__name__}"
    return data


def parse_list(data: dict[str, Any], key: str, parser: Callable[[Any], Any]) -> list[Any]:
    """Parse a list field, treating a missing field as an empty list."""
    values = data.get(key, [])
    assert isinstance(values, list), f"Field '{key}' must be a list"
    return [parser(value) for value in values]


def required_str(data: dict[str, Any], key: str) -> str:
    """Read a required non-empty string field."""
    value = data.get(key)
    assert isinstance(value, str) and value, f"Missing required string field '{key}'"
    return value


def optional_str(data: dict[str, Any], key: str) -> str | None:
    """Read an optional string field without inventing a default value."""
    value = data.get(key)
    assert value is None or isinstance(value, str), f"Optional field '{key}' must be a string when set"
    return value


def optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Read an optional mapping field and return a mutable copy."""
    value = data.get(key, {})
    assert value is None or isinstance(value, dict), f"Optional field '{key}' must be a dict when set"
    return dict(value or {})


def required_number_sequence(data: dict[str, Any], key: str, length: int) -> tuple[float, ...]:
    """Read a fixed-length numeric list such as a position or quaternion."""
    value = data.get(key)
    assert isinstance(value, (list, tuple)), f"Missing required numeric sequence field '{key}'"
    assert len(value) == length, f"Field '{key}' must contain {length} numbers"
    assert all(
        isinstance(item, Real) and not isinstance(item, bool) for item in value
    ), f"Field '{key}' must contain only numbers"
    return tuple(float(item) for item in value)


def required_enum(data: dict[str, Any], key: str, enum_type: type[Enum]) -> Enum:
    """Read a required enum field from its YAML string value."""
    value = data.get(key)
    assert value is not None, f"Missing required field '{key}'"
    parsed = parse_enum(value, key, enum_type)
    assert parsed is not None
    return parsed


def optional_enum(data: dict[str, Any], key: str, enum_type: type[Enum]) -> Enum | None:
    """Read an optional enum field from its YAML string value."""
    return parse_enum(data.get(key), key, enum_type)


def parse_enum(value: Any, key: str, enum_type: type[Enum]) -> Enum | None:
    """Convert a YAML string to an enum value and show valid options on failure."""
    if value is None or isinstance(value, enum_type):
        return value
    assert isinstance(value, str), f"Field '{key}' must be a string when set"
    try:
        return enum_type(value)
    except ValueError:
        valid_values = [enum_value.value for enum_value in enum_type]
        raise AssertionError(f"Unknown {key} '{value}'. Expected one of {valid_values}") from None


def assert_unique_ids(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
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
    assert not duplicates, f"Duplicate env graph ids found: {duplicates}"


def assert_references_exist(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
    """Ensure every graph reference points to a node or state spec that exists."""
    node_ids = {node.id for node in nodes}
    state_spec_ids = {state_spec.id for state_spec in state_specs}

    # Track ids seen so far so a node's parent must be defined *earlier* in the list. The
    # conversion process (_instantiate_assets_from_nodes) materializes nodes in order and looks
    # up the parent, so a parent listed after its reference would otherwise only fail
    # there with a raw KeyError.
    seen_node_ids: set[str] = set()
    for node in nodes:
        parent = getattr(node, "parent", None)
        if parent is not None:
            assert parent in node_ids, f"Node '{node.id}' references unknown parent '{parent}'"
            assert parent in seen_node_ids, (
                f"Node '{node.id}' references parent '{parent}' defined later in the node list; "
                "a parent must appear before any node that references it"
            )
        seen_node_ids.add(node.id)

    for task in tasks:
        for label, state_spec_id in (
            ("initial_state_spec_id", task.initial_state_spec_id),
            ("success_state_spec_id", task.success_state_spec_id),
        ):
            assert (
                state_spec_id in state_spec_ids
            ), f"Task '{task.id}' references unknown state spec '{state_spec_id}' for '{label}'"

    for state_spec in state_specs:
        for constraint in state_spec.spatial_constraints:
            assert (
                constraint.parent in node_ids
            ), f"Constraint '{constraint.id}' references unknown parent node '{constraint.parent}'"
            if constraint.child is not None:
                assert (
                    constraint.child in node_ids
                ), f"Constraint '{constraint.id}' references unknown child node '{constraint.child}'"

        for constraint in state_spec.task_constraints:
            if constraint.parent is not None:
                assert (
                    constraint.parent in node_ids
                ), f"Constraint '{constraint.id}' references unknown parent node '{constraint.parent}'"
            if constraint.child is not None:
                assert (
                    constraint.child in node_ids
                ), f"Constraint '{constraint.id}' references unknown child node '{constraint.child}'"


def assert_spatial_constraint_shapes(state_specs: list[Any]) -> None:
    """Check each spatial constraint has the parent/child shape its relation expects."""
    for state_spec in state_specs:
        for constraint in state_spec.spatial_constraints:
            constraint_type = _enum_value(constraint.type)
            if constraint_type == "at_pose":
                # Special: no relation class; pose is supplied directly via params.
                assert (
                    "position_xyz" in constraint.params
                ), f"Spatial constraint '{constraint.id}' of type 'at_pose' requires params.position_xyz"
                is_unary = True
            elif constraint_type == "in":
                # Special: no relation class; semantically a binary parent/child constraint.
                # TODO(xinjieyao, 2026-05-27): add an `In` relation class so this can resolve through the registry.
                is_unary = False
            else:
                relation_cls = relation_class_for_spatial_constraint_type(constraint.type)
                assert (
                    relation_cls is not None
                ), f"Spatial constraint type '{constraint_type}' is not mapped to a relation class"
                is_unary = relation_cls.is_unary()

            if is_unary:
                assert (
                    constraint.child is None
                ), f"Spatial constraint '{constraint.id}' of type '{constraint_type}' must not define a child node"
            else:
                assert (
                    constraint.child is not None
                ), f"Spatial constraint '{constraint.id}' of type '{constraint_type}' requires a child node"


def assert_cli_override_specs_reference_nodes(
    nodes: list["ArenaEnvGraphNodeSpec"], cli_override_specs: list[ArenaEnvGraphCliOverrideSpec]
) -> None:
    """Check each CLI override uses a unique flag and points to a real node."""
    node_ids = {node.id for node in nodes}
    seen_args: set[str] = set()
    for override in cli_override_specs:
        assert override.arg not in seen_args, f"Duplicate cli_override arg '--{override.arg}'"
        seen_args.add(override.arg)
        assert (
            override.target_node_id in node_ids
        ), f"CLI override '--{override.arg}' targets unknown node '{override.target_node_id}'"


def add_cli_override_args(
    parser: "argparse.ArgumentParser", override_specs: list[ArenaEnvGraphCliOverrideSpec]
) -> None:
    """Add each declared override to the CLI ``parser`` as a ``--flag``.

    Each flag defaults to `None`, so an omitted flag falls back to the node's YAML-specified asset.

    A declared flag that collides with one already on the parser (a built-in like ``--num_envs``
    or ``--seed``, or any flag added by ``AppLauncher.add_app_launcher_args``) is rejected.
    """
    for override in override_specs:
        flag = f"--{override.arg}"
        # _option_string_actions maps every registered option string ('--num_envs') to its action
        assert flag not in parser._option_string_actions, (  # noqa: SLF001 (introspect registered flags)
            f"CLI override flag '{flag}' (node '{override.target_node_id}') is already a parser flag "
            "(e.g. --num_envs/--seed or an AppLauncher flag); rename its 'arg' in the YAML."
        )
        parser.add_argument(
            flag,
            type=str,
            default=None,
            help=f"Override the asset behind graph node '{override.target_node_id}'.",
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
