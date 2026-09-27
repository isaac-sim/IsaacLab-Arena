# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply validated nested overrides to structured configurations."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, TypeVar

from hydra.utils import get_class
from isaaclab.utils.dict import update_class_from_dict

import isaaclab_arena.hydra.config_type_utils as config_type_utils

_ALLOWED_TARGET_MODULE_PREFIXES = (
    "isaaclab.",
    "isaaclab_contrib.",
    "isaaclab_newton.",
    "isaaclab_ov.",
    "isaaclab_physx.",
)
_HYDRA_TARGET_KEY = "_target_"
_ALLOW_CONFIG_OVERRIDE_METADATA_KEY = "allow_config_override"
ConfigT = TypeVar("ConfigT")


def apply_config_override(
    config: ConfigT,
    override: dict[str, Any],
    *,
    override_name: str,
    path: str,
    allow_hydra_targets: bool = True,
) -> ConfigT:
    """Apply a validated nested override to a configclass or dataclass."""
    assert override is not None, f"{override_name} must be provided"
    assert isinstance(override, dict), f"{override_name} must be a mapping, got {type(override).__name__}"

    values = copy.deepcopy(override)
    _validate_override_syntax(values, path=path, allow_hydra_targets=allow_hydra_targets)
    # Build targets post-order in the copy, including concrete containers for typed list entries.
    _materialize_targets(config, values, path=path)
    pending_assignments: list[tuple[Any, str | int, Any]] = []
    # Keep constructed instances out to prevent ``from_dict`` from reprocessing typed config instances as
    # raw mappings again before they are safely assigned to polymorphic fields.
    _extract_materialized_values(config, values, pending_assignments)

    try:
        update_class_from_dict(config, values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {override_name}: {exc}") from exc

    # Commit polymorphic replacements only after the residual merge succeeds.
    for target_obj, key, value in pending_assignments:
        if isinstance(target_obj, list):
            target_obj[key] = value
        else:
            setattr(target_obj, key, value)
    return config


def _validate_override_syntax(value: Any, *, path: str, allow_hydra_targets: bool = True) -> None:
    """Reject unsafe override content before ``_materialize_targets`` runs.

    Disallows ``class_type`` overrides, Hydra control keys other than ``_target_``,
    and OmegaConf ``${...}`` interpolation in strings.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            assert key != "class_type", f"'{child_path}' is derived by Isaac Lab and cannot be overridden"
            assert not key.startswith("_") or (
                allow_hydra_targets and key == _HYDRA_TARGET_KEY
            ), f"Unsupported Hydra control key '{child_path}'"
            _validate_override_syntax(item, path=child_path, allow_hydra_targets=allow_hydra_targets)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_override_syntax(item, path=f"{path}[{index}]", allow_hydra_targets=allow_hydra_targets)
    elif isinstance(value, str):
        assert "${" not in value, f"OmegaConf interpolation is not allowed at '{path}'"


def _materialize_targets(
    target: Any,
    values: dict[str, Any],
    *,
    path: str,
    construct_structured: bool = False,
    strict: bool = False,
) -> None:
    """Materialize targets post-order across annotated dicts and lists.

    Concrete dataclass containers are constructed around materialized children.
    """
    target_cls = target if isinstance(target, type) else type(target)
    fields_by_name = {field.name: field for field in dataclasses.fields(target_cls)}
    for key, value in values.items():
        child_path = f"{path}.{key}"
        if key not in fields_by_name:
            assert not strict, f"Unknown config field '{child_path}'"
            continue
        assert fields_by_name[key].metadata.get(
            _ALLOW_CONFIG_OVERRIDE_METADATA_KEY, True
        ), f"'{child_path}' cannot be overridden"
        annotation = config_type_utils.field_annotation(target_cls, key)
        values[key] = _materialize_value(
            annotation,
            value,
            path=child_path,
            construct_structured=construct_structured,
            strict=strict,
            current_value=None if isinstance(target, type) else getattr(target, key),
        )


def _materialize_value(
    annotation: Any,
    value: Any,
    *,
    path: str,
    construct_structured: bool,
    strict: bool,
    current_value: Any,
) -> Any:
    """Dispatch materialization based on the override value's structure."""
    # Handle typed lists.
    list_element_type = config_type_utils.list_element_type(annotation)
    if list_element_type is not None:
        return _materialize_list(
            list_element_type,
            value,
            path=path,
            construct_structured=construct_structured,
            strict=strict,
            current_value=current_value,
        )

    # Restore exact runtime types from YAML-compatible values and reject incompatible scalars
    # before the residual recursive merge reaches ``update_class_from_dict``.
    if not isinstance(value, dict):
        converted_value = config_type_utils.convert_override_value(annotation, value)
        if not config_type_utils.annotation_accepts_value(annotation, converted_value):
            raise ValueError(f"Invalid value at '{path}': expected {annotation}, got {type(value).__name__}")
        return converted_value

    # Handle explicitly typed configclasses.
    if _HYDRA_TARGET_KEY in value:
        return _materialize_target_mapping(annotation, value, path=path)

    # Handle ordinary nested dataclass mappings.
    return _materialize_dataclass_mapping(
        annotation,
        value,
        path=path,
        construct_structured=construct_structured,
        strict=strict,
        current_value=current_value,
    )


def _materialize_list(
    element_type: Any,
    value: Any,
    *,
    path: str,
    construct_structured: bool,
    strict: bool,
    current_value: Any,
) -> list[Any]:
    """Materialize every element of a typed override list."""
    assert isinstance(value, list), f"Expected a list at '{path}'"

    return [
        _materialize_value(
            element_type,
            item,
            path=f"{path}[{index}]",
            construct_structured=construct_structured,
            strict=strict,
            current_value=(
                current_value[index] if isinstance(current_value, list) and index < len(current_value) else None
            ),
        )
        for index, item in enumerate(value)
    ]


def _materialize_target_mapping(annotation: Any, value: dict[str, Any], *, path: str) -> Any:
    """Materialize an explicitly typed configclass mapping."""
    target_cls = _validated_target_class(value[_HYDRA_TARGET_KEY], annotation, path=path)
    payload = {key: item for key, item in value.items() if key != _HYDRA_TARGET_KEY}
    _materialize_targets(target_cls, payload, path=path, construct_structured=True, strict=True)
    return _construct_configclass(target_cls, payload, path=path)


def _materialize_dataclass_mapping(
    annotation: Any,
    value: dict[str, Any],
    *,
    path: str,
    construct_structured: bool,
    strict: bool,
    current_value: Any,
) -> Any:
    """Traverse an ordinary mapping using its concrete dataclass type."""
    concrete_type = config_type_utils.concrete_dataclass_type(annotation)
    if concrete_type is None and dataclasses.is_dataclass(current_value):
        concrete_type = type(current_value)
    if concrete_type is None:
        assert not config_type_utils.annotation_contains_dataclass(
            annotation
        ), f"Nested config '{path}' requires {_HYDRA_TARGET_KEY!r} when its parent is constructed by Hydra"
        return value

    payload = dict(value)
    nested_target = concrete_type if construct_structured else current_value
    _materialize_targets(nested_target, payload, path=path, construct_structured=construct_structured, strict=strict)
    if construct_structured:
        return _construct_configclass(concrete_type, payload, path=path)
    return payload


def _extract_materialized_values(
    target_obj: Any,
    values: dict[str, Any] | list[Any],
    pending_assignments: list[tuple[Any, str | int, Any]],
) -> None:
    """Remove materialized values from the merge payload and record their assignments."""
    if isinstance(values, list):
        if not isinstance(target_obj, list):
            return
        for index, value in enumerate(values):
            if dataclasses.is_dataclass(value):
                pending_assignments.append((target_obj, index, value))
                values[index] = {}
            elif index < len(target_obj) and isinstance(value, (dict, list)):
                _extract_materialized_values(target_obj[index], value, pending_assignments)
        return

    for key in list(values):
        if not hasattr(target_obj, key) and not isinstance(target_obj, dict):
            continue
        value = values[key]
        child_obj = target_obj[key] if isinstance(target_obj, dict) else getattr(target_obj, key)
        is_new_optional_value = child_obj is None and value is not None and not isinstance(target_obj, dict)
        is_materialized_config = dataclasses.is_dataclass(value)
        if is_new_optional_value or is_materialized_config:
            # Optional fields (for example ``placement_seed: int | None``) have no live child object
            # for recursive merging. Materialized configclasses must also bypass the raw mapping merge.
            pending_assignments.append((target_obj, key, value))
            values.pop(key)
        elif isinstance(value, (dict, list)):
            _extract_materialized_values(child_obj, value, pending_assignments)


def _construct_configclass(target_cls: type, payload: dict[str, Any], *, path: str) -> Any:
    """Construct one Isaac Lab configclass from a converted payload mapping."""
    field_names = {field.name for field in dataclasses.fields(target_cls)}
    filtered_payload = {key: item for key, item in payload.items() if key in field_names}
    try:
        return target_cls(**filtered_payload)
    except Exception as exc:
        raise ValueError(f"Could not construct {target_cls.__qualname__} at '{path}': {exc}") from exc


def _validated_target_class(target_path: Any, expected_type: Any, *, path: str) -> type:
    """Resolve and validate one Hydra target against its annotated field type."""
    assert isinstance(target_path, str) and target_path, f"'{path}.{_HYDRA_TARGET_KEY}' must be a class path string"
    module_name, separator, _ = target_path.rpartition(".")
    assert separator and module_name.startswith(
        _ALLOWED_TARGET_MODULE_PREFIXES
    ), f"Hydra target {target_path!r} at '{path}' is outside the approved Isaac Lab packages"

    try:
        target_cls = get_class(target_path)
    except Exception as exc:
        raise ValueError(f"Could not resolve Hydra target {target_path!r} at '{path}': {exc}") from exc

    assert isinstance(target_cls, type), f"Hydra target {target_path!r} at '{path}' must resolve to a class"
    assert target_cls.__module__.startswith(
        _ALLOWED_TARGET_MODULE_PREFIXES
    ), f"Hydra target {target_path!r} at '{path}' resolves outside the approved Isaac Lab packages"
    assert dataclasses.is_dataclass(
        target_cls
    ), f"Hydra target {target_path!r} at '{path}' must resolve to an Isaac Lab configclass"
    assert config_type_utils.annotation_accepts_type(
        expected_type, target_cls
    ), f"Hydra target {target_path!r} is incompatible with the annotated type of '{path}'"
    return target_cls
