# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply nested or Hydra dotlist overrides to structured configurations."""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from hydra.utils import get_class
from isaaclab_newton.physics import NewtonCfg
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

_ALLOWED_TARGET_MODULE_PREFIXES = (
    "isaaclab.",
    "isaaclab_contrib.",
    "isaaclab_newton.",
    "isaaclab_ov.",
    "isaaclab_physx.",
)
_HYDRA_TARGET_KEY = "_target_"


def dotlist_to_override(overrides: list[str]) -> dict[str, Any]:
    """Parse assignment-only Hydra dotlist values into a nested override mapping."""
    for override in overrides:
        if not override or override[0] in "+~":
            raise ValueError(f"Hydra override operators are not supported by config overrides: '{override}'")
    try:
        values = OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=False)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Could not parse Hydra overrides: {exc}") from exc
    assert isinstance(values, dict)
    return values


def nested_override(values: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted mapping keys and return one nested override mapping."""
    assert isinstance(values, dict), f"Config override must be a mapping, got {type(values).__name__}"
    dotlist: list[str] = []
    _append_dotlist_values(values, "", dotlist)
    return dotlist_to_override(dotlist)


def apply_config_override(config: Any, override: dict[str, Any]) -> Any:
    """Apply a validated nested override to a configclass or mapping in place.

    Args:
        config: Structured configuration or mapping to update.
        override: Nested mapping containing values to update.

    Returns:
        The updated ``config`` object.
    """
    assert isinstance(override, dict), f"Config override must be a mapping, got {type(override).__name__}"
    values = copy.deepcopy(override)
    _validate_override_syntax(values, path="config")

    # Validate the complete override without exposing partial mutations, then replay it onto the
    # original config so untouched nested configclass instances retain their identity.
    candidate = copy.deepcopy(config)
    _apply_override(candidate, copy.deepcopy(values))
    _apply_override(config, values)
    return config


def _apply_override(config: Any, values: dict[str, Any]) -> None:
    """Apply previously validated values without providing atomicity."""
    if isinstance(config, dict):
        _apply_mapping_override(config, values, path="config")
    elif callable(getattr(config, "from_dict", None)):
        _apply_configclass_override(config, values)
    else:
        raise TypeError(f"Config override target must be a configclass or mapping, got {type(config).__name__}")


def _append_dotlist_values(value: Any, prefix: str, dotlist: list[str]) -> None:
    """Append mapping leaves as OmegaConf-compatible dotlist assignments."""
    if isinstance(value, dict):
        for key, child_value in value.items():
            assert key, "Hydra override keys must be non-empty"
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _append_dotlist_values(child_value, child_prefix, dotlist)
        return
    assert prefix, "Hydra override paths must be non-empty"
    dotlist.append(f"{prefix}={json.dumps(value, separators=(',', ':'))}")


def _apply_mapping_override(target: dict[str, Any], values: dict[str, Any], *, path: str) -> None:
    """Apply values recursively to an existing-key mapping."""
    for key, value in values.items():
        child_path = f"{path}.{key}"
        if key not in target:
            raise ValueError(f"Invalid config override: Unknown config field '{child_path}'")
        current_value = target[key]
        if not isinstance(value, dict):
            target[key] = value
        elif isinstance(current_value, dict):
            _apply_mapping_override(current_value, value, path=child_path)
        elif callable(getattr(current_value, "from_dict", None)):
            apply_config_override(current_value, value)
        else:
            raise ValueError(f"Invalid config override: expected a structured value at '{child_path}'")


def _apply_configclass_override(config: Any, values: dict[str, Any]) -> None:
    """Apply an override using the Isaac Lab configclass ``from_dict`` contract."""
    _materialize_targets(config, values, path="config")
    pending_assignments: list[tuple[Any, str | int, Any]] = []
    _extract_materialized_values(config, values, pending_assignments)
    try:
        config.from_dict(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid config override: {exc}") from exc

    for target_obj, key, value in pending_assignments:
        if isinstance(target_obj, list):
            target_obj[key] = value
        else:
            setattr(target_obj, key, value)
        if key == "solver_cfg" and isinstance(target_obj, NewtonCfg):
            target_obj.class_type = target_obj.solver_cfg.class_type


def _validate_override_syntax(value: Any, *, path: str) -> None:
    """Reject unsafe Hydra control keys, derived fields, and interpolation."""
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            assert key != "class_type", f"'{child_path}' is derived by Isaac Lab and cannot be overridden"
            assert not key.startswith("_") or key == _HYDRA_TARGET_KEY, f"Unsupported Hydra control key '{child_path}'"
            _validate_override_syntax(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_override_syntax(item, path=f"{path}[{index}]")
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
    """Materialize targets post-order across annotated dicts and lists."""
    target_cls = target if isinstance(target, type) else type(target)
    field_names = {field.name for field in dataclasses.fields(target_cls)}
    for key, value in values.items():
        child_path = f"{path}.{key}"
        if key not in field_names:
            assert not strict, f"Unknown config field '{child_path}'"
            continue
        annotation = _field_annotation(target_cls, key)
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
    list_element_type = _list_element_type(annotation)
    if list_element_type is not None:
        return _materialize_list(
            list_element_type,
            value,
            path=path,
            construct_structured=construct_structured,
            strict=strict,
            current_value=current_value,
        )
    if not isinstance(value, dict):
        return value
    if _HYDRA_TARGET_KEY in value:
        return _materialize_target_mapping(annotation, value, path=path)
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

    def current_item(index: int) -> Any:
        if isinstance(current_value, list) and index < len(current_value):
            return current_value[index]
        return None

    return [
        _materialize_value(
            element_type,
            item,
            path=f"{path}[{index}]",
            construct_structured=construct_structured,
            strict=strict,
            current_value=current_item(index),
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
    concrete_type = _concrete_dataclass_type(annotation)
    if concrete_type is None and dataclasses.is_dataclass(current_value):
        concrete_type = type(current_value)
    if concrete_type is None:
        assert not _annotation_contains_dataclass(
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
        if dataclasses.is_dataclass(value):
            pending_assignments.append((target_obj, key, value))
            values.pop(key)
        elif isinstance(value, (dict, list)):
            _extract_materialized_values(child_obj, value, pending_assignments)


def _construct_configclass(target_cls: type, payload: dict[str, Any], *, path: str) -> Any:
    """Construct one Isaac Lab configclass from a coerced payload mapping."""
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
    assert _annotation_accepts_type(
        expected_type, target_cls
    ), f"Hydra target {target_path!r} is incompatible with the annotated type of '{path}'"
    return target_cls


def _field_annotation(owner: type, field_name: str) -> Any:
    """Resolve one inherited dataclass field annotation without resolving unrelated fields."""
    for cls in owner.__mro__:
        own_fields = cls.__dict__.get("__configclass_own_fields__")
        if own_fields is not None and field_name not in own_fields:
            continue
        annotation = cls.__dict__.get("__annotations__", {}).get(field_name)
        if annotation is None:
            continue
        if isinstance(annotation, str):
            module_globals = vars(sys.modules[cls.__module__])
            holder = type("_FieldAnnotation", (), {"__annotations__": {"value": annotation}})
            return get_type_hints(holder, globalns=module_globals, localns=vars(cls))["value"]
        return annotation
    raise TypeError(f"Could not resolve the annotated type of '{owner.__name__}.{field_name}'")


def _list_element_type(annotation: Any) -> Any | None:
    """Return the element annotation for ``list[T]``, or ``None``."""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else None
    return None


def _concrete_dataclass_type(annotation: Any) -> type | None:
    """Return a single dataclass type annotation."""
    if _union_members(annotation) is not None:
        return None
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    return None


def _union_members(annotation: Any) -> tuple[Any, ...] | None:
    """Return union member annotations, or ``None``."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return get_args(annotation)
    return None


def _annotation_accepts_type(annotation: Any, target_cls: type) -> bool:
    """Return whether ``target_cls`` is compatible with a field annotation."""
    members = _union_members(annotation)
    if members is not None:
        return any(_annotation_accepts_type(member, target_cls) for member in members)
    return isinstance(annotation, type) and issubclass(target_cls, annotation)


def _annotation_contains_dataclass(annotation: Any) -> bool:
    """Return whether an annotation contains a dataclass type."""
    members = _union_members(annotation)
    if members is not None:
        return any(_annotation_contains_dataclass(member) for member in members)
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)
