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

from hydra.utils import get_class

_ALLOWED_TARGET_MODULE_PREFIXES = (
    "isaaclab.",
    "isaaclab_contrib.",
    "isaaclab_newton.",
    "isaaclab_ov.",
    "isaaclab_physx.",
)
_HYDRA_TARGET_KEY = "_target_"
_ALLOWED_ARENA_TARGETS = {
    "isaaclab_arena.assets.physics_config.PhysicsUsdFileCfg",
    "isaaclab_arena.assets.physics_config.PrimPhysicsCfg",
    "isaaclab_arena.assets.physics_config.MujocoEqualityPropertiesCfg",
}


def apply_env_cfg_override(env_cfg: Any, override: dict[str, Any] | None) -> Any:
    """Apply a validated environment-config override in place.

    Hydra ``_target_`` nodes first replace polymorphic config fields with concrete
    Isaac Lab configclass instances on a working copy. Remaining values are merged
    against that concrete schema, then published to ``env_cfg`` only after the
    complete override succeeds.

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
    _validate_data_only(values, path="env")
    working_cfg = copy.deepcopy(env_cfg)
    _materialize_targets(working_cfg, values, path="env")

    try:
        working_cfg.from_dict(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid env_cfg_override: {exc}") from exc

    assert type(working_cfg) is type(env_cfg)
    for field in dataclasses.fields(env_cfg):
        setattr(env_cfg, field.name, getattr(working_cfg, field.name))
    return env_cfg


def _materialize_targets(target_obj: Any, values: dict[str, Any], *, path: str) -> None:
    """Materialize targets on a working config and consume their copied mappings."""
    if not dataclasses.is_dataclass(target_obj):
        return

    field_names = _dataclass_field_names(type(target_obj))
    for key, value in values.items():
        if key not in field_names or not isinstance(value, dict):
            continue

        child_path = f"{path}.{key}"
        child_obj = getattr(target_obj, key)
        target_path = value.pop(_HYDRA_TARGET_KEY, None)
        if target_path is not None:
            expected_type = _field_annotation(type(target_obj), key, current_type=type(child_obj))
            target_cls = _validated_target_class(target_path, expected_type, path=child_path)
            _validate_nested_targets(target_cls, value, path=child_path)
            _materialize_nested_values(target_cls, value, path=child_path)
            try:
                child_obj = target_cls(**value)
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


def _materialize_nested_values(target_cls: type, values: dict[str, Any], *, path: str) -> None:
    """Materialize validated target payload values in place."""
    for key, value in values.items():
        annotation = _field_annotation(target_cls, key)
        values[key] = _materialize_nested_value(annotation, value, path=f"{path}.{key}")


def _materialize_nested_value(annotation: Any, value: Any, *, path: str) -> Any:
    """Construct validated configclass values nested in dictionaries and lists."""
    if isinstance(value, list):
        element_annotation = _list_element_annotation(annotation)
        return [
            _materialize_nested_value(element_annotation, item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value

    if _HYDRA_TARGET_KEY in value:
        target_path = value[_HYDRA_TARGET_KEY]
        target_cls = _validated_target_class(target_path, annotation, path=path)
        payload = _override_payload(value)
        _materialize_nested_values(target_cls, payload, path=path)
        try:
            target_obj = target_cls(**payload)
        except Exception as exc:
            raise ValueError(f"Could not instantiate {target_path!r} at '{path}': {exc}") from exc
        assert isinstance(target_obj, target_cls)
        return target_obj

    nested_cls = _dataclass_type(annotation)
    if nested_cls is not None:
        _materialize_nested_values(nested_cls, value, path=path)
        return nested_cls(**value)

    value_annotation = _dict_value_annotation(annotation)
    return {key: _materialize_nested_value(value_annotation, item, path=f"{path}.{key}") for key, item in value.items()}


def _validate_nested_targets(target_cls: type, values: dict[str, Any], *, path: str) -> None:
    """Validate nested Hydra targets before recursively instantiating a config tree."""
    field_names = _dataclass_field_names(target_cls)
    for key, value in values.items():
        child_path = f"{path}.{key}"
        assert key in field_names, f"Unknown config field '{child_path}'"
        annotation = _field_annotation(target_cls, key)
        _validate_nested_value(annotation, value, path=child_path)


def _validate_nested_value(
    annotation: Any,
    value: Any,
    *,
    path: str,
    allow_dataclass_mapping: bool = False,
) -> None:
    """Validate Hydra targets nested in typed dictionaries and lists."""
    if isinstance(value, list):
        element_annotation = _list_element_annotation(annotation)
        for index, item in enumerate(value):
            _validate_nested_value(
                element_annotation,
                item,
                path=f"{path}[{index}]",
                allow_dataclass_mapping=True,
            )
        return
    if not isinstance(value, dict):
        return

    if _HYDRA_TARGET_KEY in value:
        nested_cls = _validated_target_class(value[_HYDRA_TARGET_KEY], annotation, path=path)
        _validate_nested_targets(nested_cls, _override_payload(value), path=path)
        return

    nested_cls = _dataclass_type(annotation)
    if nested_cls is not None:
        assert (
            allow_dataclass_mapping
        ), f"Nested config '{path}' requires {_HYDRA_TARGET_KEY!r} when its parent is constructed by Hydra"
        _validate_nested_targets(nested_cls, value, path=path)
        return

    value_annotation = _dict_value_annotation(annotation)
    for key, item in value.items():
        _validate_nested_value(value_annotation, item, path=f"{path}.{key}")


def _validated_target_class(target_path: Any, expected_type: Any, *, path: str) -> type:
    """Resolve and validate one Hydra target against its annotated field type."""
    assert isinstance(target_path, str) and target_path, f"'{path}.{_HYDRA_TARGET_KEY}' must be a class path string"
    module_name, separator, _ = target_path.rpartition(".")
    assert separator and (
        module_name.startswith(_ALLOWED_TARGET_MODULE_PREFIXES) or target_path in _ALLOWED_ARENA_TARGETS
    ), f"Hydra target {target_path!r} at '{path}' is outside the approved Isaac Lab packages and Arena config classes"

    try:
        target_cls = get_class(target_path)
    except Exception as exc:
        raise ValueError(f"Could not resolve Hydra target {target_path!r} at '{path}': {exc}") from exc

    assert isinstance(target_cls, type), f"Hydra target {target_path!r} at '{path}' must resolve to a class"
    assert (
        target_cls.__module__.startswith(_ALLOWED_TARGET_MODULE_PREFIXES)
        or f"{target_cls.__module__}.{target_cls.__name__}" in _ALLOWED_ARENA_TARGETS
    ), f"Hydra target {target_path!r} at '{path}' resolves outside the approved config classes"
    assert dataclasses.is_dataclass(
        target_cls
    ), f"Hydra target {target_path!r} at '{path}' must resolve to an Isaac Lab configclass"
    assert _annotation_accepts_type(
        expected_type, target_cls
    ), f"Hydra target {target_path!r} is incompatible with the annotated type of '{path}'"
    return target_cls


def _field_annotation(owner: type, field_name: str, current_type: type | None = None) -> Any:
    """Resolve one inherited dataclass field annotation without resolving unrelated fields."""
    for cls in owner.__mro__:
        annotation = cls.__dict__.get("__annotations__", {}).get(field_name)
        if annotation is None:
            continue
        if isinstance(annotation, str):
            module_globals = vars(sys.modules[cls.__module__])
            holder = type("_FieldAnnotation", (), {"__annotations__": {"value": annotation}})
            # Some Isaac Lab fields import their base config type only under TYPE_CHECKING
            # (for example AssetBaseCfg.spawn: SpawnerCfg). An existing value supplies that
            # public type through its MRO without weakening the annotation to Any.
            localns = {base.__name__: base for base in current_type.__mro__} if current_type is not None else {}
            localns.update(vars(cls))
            return get_type_hints(holder, globalns=module_globals, localns=localns)["value"]
        return annotation
    raise TypeError(f"Could not resolve the annotated type of '{owner.__name__}.{field_name}'")


def _union_members(annotation: Any) -> tuple[Any, ...] | None:
    """Return union member annotations, or ``None`` when ``annotation`` is not a union."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return get_args(annotation)
    return None


def _annotation_accepts_type(annotation: Any, target_cls: type) -> bool:
    """Return whether ``target_cls`` is compatible with a field annotation."""
    if annotation is Any:
        return True
    members = _union_members(annotation)
    if members is not None:
        return any(_annotation_accepts_type(member, target_cls) for member in members)
    return isinstance(annotation, type) and issubclass(target_cls, annotation)


def _dataclass_type(annotation: Any) -> type | None:
    """Return the dataclass type contained in an annotation, if any."""
    members = _union_members(annotation)
    if members is not None:
        for member in members:
            member_type = _dataclass_type(member)
            if member_type is not None:
                return member_type
        return None
    return annotation if isinstance(annotation, type) and dataclasses.is_dataclass(annotation) else None


def _list_element_annotation(annotation: Any) -> Any:
    """Return a list annotation's element type, or ``Any``."""
    members = _union_members(annotation)
    if members is not None:
        for member in members:
            element_annotation = _list_element_annotation(member)
            if element_annotation is not Any:
                return element_annotation
        return Any
    args = get_args(annotation)
    return args[0] if get_origin(annotation) is list and args else Any


def _dict_value_annotation(annotation: Any) -> Any:
    """Return a dict annotation's value type, or ``Any``."""
    members = _union_members(annotation)
    if members is not None:
        for member in members:
            value_annotation = _dict_value_annotation(member)
            if value_annotation is not Any:
                return value_annotation
        return Any
    args = get_args(annotation)
    return args[1] if get_origin(annotation) is dict and len(args) == 2 else Any


def _dataclass_field_names(owner: type) -> set[str]:
    """Return dataclass field names for ``owner``."""
    return {field.name for field in dataclasses.fields(owner)}


def _override_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Return override entries excluding the Hydra class selector key."""
    return {key: value for key, value in values.items() if key != _HYDRA_TARGET_KEY}


def _validate_data_only(value: Any, *, path: str) -> None:
    """Reject executable or Hydra-control values left after target construction."""
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            assert key != "class_type", f"'{child_path}' is derived by Isaac Lab and cannot be overridden"
            assert not key.startswith("_") or key == _HYDRA_TARGET_KEY, f"Unsupported Hydra control key '{child_path}'"
            _validate_data_only(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_data_only(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        assert "${" not in value, f"OmegaConf interpolation is not allowed at '{path}'"
