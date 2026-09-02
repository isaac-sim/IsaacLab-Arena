# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply validated Hydra overrides to an Isaac Lab environment configuration."""

from __future__ import annotations

import copy
import dataclasses
import sys
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from hydra.utils import get_class, instantiate

_ALLOWED_TARGET_MODULE_PREFIXES = (
    "isaaclab.",
    "isaaclab_contrib.",
    "isaaclab_newton.",
    "isaaclab_ov.",
    "isaaclab_physx.",
)
_HYDRA_TARGET_KEY = "_target_"


def apply_env_cfg_override(env_cfg: Any, override: dict[str, Any] | None) -> Any:
    """Apply a validated environment-config override in place.

    Hydra ``_target_`` nodes first replace polymorphic config fields with concrete
    Isaac Lab configclass instances. Remaining values are then merged against the
    concrete schema and applied through ``from_dict``.

    Args:
        env_cfg: Concrete Isaac Lab environment configuration to update.
        override: Nested override mapping, or ``None`` for no changes.

    Returns:
        The updated ``env_cfg`` instance.
    """
    if override is None:
        return env_cfg
    assert isinstance(override, dict), f"env_cfg_override must be a mapping, got {type(override).__name__}"

    values = copy.deepcopy(override)
    _validate_data_only(values, path="env", allow_target=True)
    _materialize_targets(env_cfg, values, path="env")

    try:
        env_cfg.from_dict(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid env_cfg_override: {exc}") from exc
    return env_cfg


def _materialize_targets(target_obj: Any, values: dict[str, Any], *, path: str) -> None:
    """Replace ``_target_`` mappings with validated configclass instances."""
    if not dataclasses.is_dataclass(target_obj):
        return

    field_names = {field.name for field in dataclasses.fields(target_obj)}
    for key, value in values.items():
        if key not in field_names or not isinstance(value, dict):
            continue

        child_path = f"{path}.{key}"
        child_obj = getattr(target_obj, key)
        target_path = value.pop(_HYDRA_TARGET_KEY, None)
        if target_path is not None:
            expected_type = _field_annotation(type(target_obj), key)
            target_cls = _validated_target_class(target_path, expected_type, path=child_path)
            _validate_nested_targets(target_cls, value, path=child_path)
            try:
                child_obj = instantiate(
                    {_HYDRA_TARGET_KEY: target_path, **value},
                    _convert_="object",
                )
            except Exception as exc:
                raise ValueError(f"Could not instantiate {target_path!r} at '{child_path}': {exc}") from exc
            assert isinstance(child_obj, target_cls)
            setattr(target_obj, key, child_obj)
            value.clear()
            continue

        if value:
            assert (
                child_obj is not None
            ), f"Override '{child_path}' targets None; add {_HYDRA_TARGET_KEY!r} to select a concrete config class"
            _materialize_targets(child_obj, value, path=child_path)


def _validate_nested_targets(target_cls: type, values: dict[str, Any], *, path: str) -> None:
    """Validate nested Hydra targets before recursively instantiating a config tree."""
    field_names = {field.name for field in dataclasses.fields(target_cls)}
    for key, value in values.items():
        child_path = f"{path}.{key}"
        assert key in field_names, f"Unknown config field '{child_path}'"
        if not isinstance(value, dict):
            continue
        annotation = _field_annotation(target_cls, key)
        if _HYDRA_TARGET_KEY not in value:
            assert not _annotation_contains_dataclass(
                annotation
            ), f"Nested config '{child_path}' requires {_HYDRA_TARGET_KEY!r} when its parent is constructed by Hydra"
            continue
        nested_target = value[_HYDRA_TARGET_KEY]
        nested_cls = _validated_target_class(
            nested_target,
            annotation,
            path=child_path,
        )
        _validate_nested_targets(
            nested_cls,
            {nested_key: nested_value for nested_key, nested_value in value.items() if nested_key != _HYDRA_TARGET_KEY},
            path=child_path,
        )


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
        annotation = cls.__dict__.get("__annotations__", {}).get(field_name)
        if annotation is None:
            continue
        if isinstance(annotation, str):
            module_globals = vars(sys.modules[cls.__module__])
            holder = type("_FieldAnnotation", (), {"__annotations__": {"value": annotation}})
            return get_type_hints(holder, globalns=module_globals, localns=vars(cls))["value"]
        return annotation
    raise TypeError(f"Could not resolve the annotated type of '{owner.__name__}.{field_name}'")


def _annotation_accepts_type(annotation: Any, target_cls: type) -> bool:
    """Return whether ``target_cls`` is compatible with a field annotation."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return any(_annotation_accepts_type(member, target_cls) for member in get_args(annotation))

    return isinstance(annotation, type) and issubclass(target_cls, annotation)


def _annotation_contains_dataclass(annotation: Any) -> bool:
    """Return whether an annotation contains a dataclass type."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return any(_annotation_contains_dataclass(member) for member in get_args(annotation))
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def _validate_data_only(value: Any, *, path: str, allow_target: bool = False) -> None:
    """Reject executable or Hydra-control values left after target construction."""
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            assert key != "class_type", f"'{child_path}' is derived by Isaac Lab and cannot be overridden"
            assert not key.startswith("_") or (
                allow_target and key == _HYDRA_TARGET_KEY
            ), f"Unsupported Hydra control key '{child_path}'"
            _validate_data_only(item, path=child_path, allow_target=allow_target)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_data_only(item, path=f"{path}[{index}]", allow_target=allow_target)
    elif isinstance(value, str):
        assert "${" not in value, f"OmegaConf interpolation is not allowed at '{path}'"
