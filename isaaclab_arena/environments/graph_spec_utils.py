# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator
from numbers import Real
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry

if TYPE_CHECKING:
    import argparse

    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphCliOverrideSpec, ArenaEnvGraphNodeSpec
    from isaaclab_arena.relations.relations import RelationBase


def coerce_number_sequence(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    """Coerce a fixed-length numeric list or tuple (e.g. position or quaternion)."""
    assert isinstance(value, (list, tuple)), f"Field '{field_name}' must be a list or tuple of {length} numbers"
    assert len(value) == length, f"Field '{field_name}' must contain exactly {length} numbers, got {len(value)}"
    assert all(
        isinstance(item, Real) and not isinstance(item, bool) for item in value
    ), f"Field '{field_name}' must contain only numbers"
    return tuple(float(item) for item in value)


def assert_unique_ids(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
    """Ensure every graph id is unique, including spatial constraint ids inside states."""
    id_locations: dict[str, list[str]] = {}
    for node in nodes:
        _add_id_location(id_locations, node.id, f"node '{node.id}'")
    for task in tasks:
        _add_id_location(id_locations, task.id, f"task '{task.id}'")
    for state_spec in state_specs:
        _add_id_location(id_locations, state_spec.id, f"state spec '{state_spec.id}'")
        for constraint in state_spec.spatial_constraints:
            _add_id_location(id_locations, constraint.id, f"spatial constraint '{constraint.id}'")

    duplicates = {spec_id: locations for spec_id, locations in id_locations.items() if len(locations) > 1}
    assert not duplicates, f"Duplicate env graph ids found: {duplicates}"


def assert_constraint_references(nodes: list[Any], state_specs: list[Any]) -> None:
    """Ensure every node parent and constraint reference points to a node that exists."""
    node_ids = {node.id for node in nodes}

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

    for state_spec in state_specs:
        for constraint in state_spec.spatial_constraints:
            assert (
                constraint.subject in node_ids
            ), f"Constraint '{constraint.id}' references unknown subject node '{constraint.subject}'"
            if constraint.reference is not None:
                assert (
                    constraint.reference in node_ids
                ), f"Constraint '{constraint.id}' references unknown reference node '{constraint.reference}'"

        for task_constraint in state_spec.task_constraints:
            assert (
                task_constraint.parent in node_ids
            ), f"Task constraint '{task_constraint.id}' references unknown parent node '{task_constraint.parent}'"
            if task_constraint.child is not None:
                assert (
                    task_constraint.child in node_ids
                ), f"Task constraint '{task_constraint.id}' references unknown child node '{task_constraint.child}'"


def assert_task_wiring(tasks: list[Any], state_specs: list[Any]) -> None:
    """Ensure each task's ``initial_state_spec_id`` / ``success_state_spec_id`` references a state."""
    state_spec_ids = {state_spec.id for state_spec in state_specs}
    for task in tasks:
        for label, state_spec_id in (
            ("initial_state_spec_id", task.initial_state_spec_id),
            ("success_state_spec_id", task.success_state_spec_id),
        ):
            assert (
                state_spec_id in state_spec_ids
            ), f"Task '{task.id}' references unknown state spec '{state_spec_id}' for '{label}'"


def assert_spatial_constraint_shapes(state_specs: list[Any]) -> None:
    """Check each spatial constraint has the subject/reference shape its relation expects."""
    for state_spec in state_specs:
        for constraint in state_spec.spatial_constraints:
            relation_cls = relation_class_for_spatial_constraint_type(constraint.kind)
            is_unary = relation_cls.is_unary()
            constraint_kind = constraint.kind

            if is_unary:
                assert constraint.reference is None, (
                    f"Spatial constraint '{constraint.id}' of kind '{constraint_kind}' must not define"
                    " relation.reference"
                )
            else:
                assert (
                    constraint.reference is not None
                ), f"Spatial constraint '{constraint.id}' of kind '{constraint_kind}' requires relation.reference"


def assert_cli_override_specs_reference_nodes(
    nodes: list[ArenaEnvGraphNodeSpec], cli_override_specs: list[ArenaEnvGraphCliOverrideSpec]
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


def add_cli_override_args(parser: argparse.ArgumentParser, override_specs: list[ArenaEnvGraphCliOverrideSpec]) -> None:
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


def relation_class_for_spatial_constraint_type(constraint_type: str) -> type[RelationBase]:
    """Look up the ``RelationBase`` class registered for a constraint-type name; raises if unknown."""
    return ObjectRelationLibraryRegistry().get_object_relation_by_name(constraint_type)


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
