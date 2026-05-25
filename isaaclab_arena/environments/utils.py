# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from enum import Enum
from numbers import Real
from typing import Any


def as_dict(data: Any, spec_name: str) -> dict[str, Any]:
    assert isinstance(data, dict), f"{spec_name} must be a dict, got {type(data).__name__}"
    return data


def parse_list(data: dict[str, Any], key: str, parser: Callable[[Any], Any]) -> list[Any]:
    values = data.get(key, [])
    assert isinstance(values, list), f"Field '{key}' must be a list"
    return [parser(value) for value in values]


def required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    assert isinstance(value, str) and value, f"Missing required string field '{key}'"
    return value


def optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    assert value is None or isinstance(value, str), f"Optional field '{key}' must be a string when set"
    return value


def optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    assert value is None or isinstance(value, dict), f"Optional field '{key}' must be a dict when set"
    return dict(value or {})


def required_number_sequence(data: dict[str, Any], key: str, length: int) -> tuple[float, ...]:
    value = data.get(key)
    assert isinstance(value, (list, tuple)), f"Missing required numeric sequence field '{key}'"
    assert len(value) == length, f"Field '{key}' must contain {length} numbers"
    assert all(
        isinstance(item, Real) and not isinstance(item, bool) for item in value
    ), f"Field '{key}' must contain only numbers"
    return tuple(float(item) for item in value)


def required_enum(data: dict[str, Any], key: str, enum_type: type[Enum]) -> Enum:
    value = data.get(key)
    assert value is not None, f"Missing required field '{key}'"
    parsed = parse_enum(value, key, enum_type)
    assert parsed is not None
    return parsed


def optional_enum(data: dict[str, Any], key: str, enum_type: type[Enum]) -> Enum | None:
    return parse_enum(data.get(key), key, enum_type)


def parse_enum(value: Any, key: str, enum_type: type[Enum]) -> Enum | None:
    if value is None or isinstance(value, enum_type):
        return value
    assert isinstance(value, str), f"Field '{key}' must be a string when set"
    try:
        return enum_type(value)
    except ValueError:
        valid_values = [enum_value.value for enum_value in enum_type]
        raise AssertionError(f"Unknown {key} '{value}'. Expected one of {valid_values}") from None


def assert_env_graph_universal_ids(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
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


def assert_env_graph_references_exist(nodes: list[Any], tasks: list[Any], state_specs: list[Any]) -> None:
    node_ids = {node.id for node in nodes}
    state_spec_ids = {state_spec.id for state_spec in state_specs}

    for node in nodes:
        # `parent` only exists on ArenaEnvGraphObjectReferenceNodeSpec; getattr keeps this
        # helper generic so it doesn't need to import the subclass.
        parent = getattr(node, "parent", None)
        if parent is not None:
            assert parent in node_ids, f"Node '{node.id}' references unknown parent '{parent}'"

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


def _add_id_location(id_locations: dict[str, list[str]], spec_id: str, location: str) -> None:
    id_locations.setdefault(spec_id, []).append(location)
