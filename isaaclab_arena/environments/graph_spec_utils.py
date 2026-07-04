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

    from isaaclab_arena.environments.arena_env_graph_types import CliOverrideSpec
    from isaaclab_arena.relations.relations import RelationBase


def coerce_number_sequence(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    """Coerce a fixed-length numeric list or tuple (e.g. position or quaternion)."""
    assert isinstance(value, (list, tuple)), f"Field '{field_name}' must be a list or tuple of {length} numbers"
    assert len(value) == length, f"Field '{field_name}' must contain exactly {length} numbers, got {len(value)}"
    assert all(
        isinstance(item, Real) and not isinstance(item, bool) for item in value
    ), f"Field '{field_name}' must contain only numbers"
    return tuple(float(item) for item in value)


def unique_node_id(existing_ids: set[str], base: str) -> str:
    """Return the first non-colliding id from ``base``, ``base_1``, ``base_2``, ... given ``existing_ids``."""
    if base not in existing_ids:
        return base
    suffix = 1
    while f"{base}_{suffix}" in existing_ids:
        suffix += 1
    return f"{base}_{suffix}"


def assert_graph_spec_asset_ids_unique(
    embodiment: Any,
    background: Any,
    objects: list[Any],
    object_references: list[Any],
) -> None:
    """Ensure every asset and object-reference id in a graph spec is unique."""
    id_locations: dict[str, list[str]] = {}
    for asset, label in (
        (embodiment, "embodiment"),
        (background, "background"),
    ):
        _add_id_location(id_locations, asset.id, label)
    for obj in objects:
        _add_id_location(id_locations, obj.id, f"object '{obj.id}'")
    for ref in object_references:
        _add_id_location(id_locations, ref.id, f"object_reference '{ref.id}'")

    duplicates = {spec_id: locations for spec_id, locations in id_locations.items() if len(locations) > 1}
    assert not duplicates, f"Duplicate graph asset ids found: {duplicates}"


def assert_graph_spec_object_reference_parents(
    object_references: list[Any],
    known_ids: set[str],
) -> None:
    """Ensure each object reference parent exists."""
    for ref in object_references:
        assert ref.parent_id in known_ids, f"Object reference '{ref.id}' references unknown parent '{ref.parent_id}'"


def assert_graph_spec_relation_references(relations: list[Any], known_ids: set[str]) -> None:
    """Ensure relation subject/reference endpoints name known asset ids."""
    for index, relation in enumerate(relations):
        assert (
            relation.subject in known_ids
        ), f"Relation[{index}] kind '{relation.kind}' references unknown subject '{relation.subject}'"
        if relation.reference is not None:
            assert (
                relation.reference in known_ids
            ), f"Relation[{index}] kind '{relation.kind}' references unknown reference '{relation.reference}'"


def assert_graph_spec_task_param_references(tasks: list[Any], known_ids: set[str]) -> None:
    """Ensure string-valued task params reference known asset ids."""
    for task in tasks:
        for param_name, param_value in task.params.items():
            if isinstance(param_value, str):
                assert (
                    param_value in known_ids
                ), f"Task '{task.kind}' param '{param_name}' references unknown node '{param_value}'"


def assert_spatial_relation_shapes(relations: list[Any]) -> None:
    """Check each relation has the subject/reference shape its kind expects."""
    for index, relation in enumerate(relations):
        relation_cls = relation_class_for_spatial_constraint_type(relation.kind)
        if relation_cls.is_unary():
            assert (
                relation.reference is None
            ), f"Relation[{index}] kind '{relation.kind}' must not define relation.reference"
        else:
            assert (
                relation.reference is not None
            ), f"Relation[{index}] kind '{relation.kind}' requires relation.reference"


def add_cli_override_args(parser: argparse.ArgumentParser, override_specs: list[CliOverrideSpec]) -> None:
    """Add each declared override to the CLI ``parser`` as a ``--flag``."""
    for override in override_specs:
        flag = f"--{override.arg}"
        assert flag not in parser._option_string_actions, (  # noqa: SLF001
            f"CLI override flag '{flag}' (asset '{override.target_node_id}') is already a parser flag "
            "(e.g. --num_envs/--seed or an AppLauncher flag); rename its 'arg' in the YAML."
        )
        parser.add_argument(
            flag,
            type=str,
            default=None,
            help=f"Override the registry name behind graph asset '{override.target_node_id}'.",
        )


def _add_id_location(id_locations: dict[str, list[str]], spec_id: str, location: str) -> None:
    id_locations.setdefault(spec_id, []).append(location)


def relation_class_for_spatial_constraint_type(constraint_type: str) -> type[RelationBase]:
    """Look up the ``RelationBase`` class registered for a constraint-type name; raises if unknown."""
    return ObjectRelationLibraryRegistry().get_object_relation_by_name(constraint_type)


def iter_nested_leaf_values(value: Any, key_path: str = "") -> Iterator[tuple[str, Any]]:
    """Walk nested task-arg values while keeping a readable path for errors."""
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
    """Apply a transform to nested task-arg leaves while preserving container shape."""
    if isinstance(value, dict):
        return {key: map_nested_leaf_values(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [map_nested_leaf_values(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(map_nested_leaf_values(item, transform) for item in value)
    return transform(value)


def normalize_identifier(identifier: str) -> str:
    """Normalize names so YAML keys can be matched across casing and separators."""
    return "".join(char for char in identifier.lower() if char.isalnum())


def camel_to_snake(identifier: str) -> str:
    """Turn a class-like name into the module-style name we try during discovery."""
    chars: list[str] = []
    for index, char in enumerate(identifier):
        if char.isupper() and index > 0 and (identifier[index - 1].islower() or identifier[index - 1].isdigit()):
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def strip_suffix(value: str, suffix: str) -> str:
    """Remove a suffix only when the value actually has it."""
    return value[: -len(suffix)] if value.endswith(suffix) else value
