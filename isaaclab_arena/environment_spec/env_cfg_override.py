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
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from hydra.utils import get_class
from isaaclab_newton.physics import NewtonCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

_ALLOWED_TARGET_MODULE_PREFIXES = (
    "isaaclab.",
    "isaaclab_contrib.",
    "isaaclab_newton.",
    "isaaclab_ov.",
    "isaaclab_physx.",
)
_HYDRA_TARGET_KEY = "_target_"


def apply_env_cfg_override(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    override: dict[str, Any],
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply a validated environment-config override in place.

    Materializes nested Hydra targets in a copied override, merges the residual
    values through ``from_dict``, then commits deferred type replacements.

    Args:
        env_cfg: Arena manager-based RL environment configuration to update.
        override: Nested override mapping from graph ``env_cfg_override``.

    Returns:
        The updated ``env_cfg`` instance.
    """
    assert override is not None, "env_cfg_override must be provided"
    assert isinstance(override, dict), f"env_cfg_override must be a mapping, got {type(override).__name__}"

    values = copy.deepcopy(override)
    _validate_override_syntax(values, path="env")
    # Build targets post-order in the copy, including concrete containers for typed list entries.
    _materialize_targets(env_cfg, values, path="env")
    pending_assignments: list[tuple[Any, str | int, Any]] = []
    # Keep constructed instances out to prevent ``from_dict`` from reprocessing typed config instances as
    # raw mappings again before they are safely assigned to polymorphic fields.
    _extract_materialized_values(env_cfg, values, pending_assignments)

    try:
        env_cfg.from_dict(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid env_cfg_override: {exc}") from exc

    # Commit polymorphic replacements only after the residual merge succeeds.
    for target_obj, key, value in pending_assignments:
        if isinstance(target_obj, list):
            target_obj[key] = value
        else:
            setattr(target_obj, key, value)
        if key == "solver_cfg" and isinstance(target_obj, NewtonCfg):
            # NewtonCfg.__post_init__ derives the manager from the initial
            # solver, so synchronize it after replacing solver_cfg.
            target_obj.class_type = target_obj.solver_cfg.class_type
    return env_cfg


def _validate_override_syntax(value: Any, *, path: str) -> None:
    """Reject unsafe override content before ``_materialize_targets`` runs.

    Disallows ``class_type`` overrides, Hydra control keys other than ``_target_``,
    and OmegaConf ``${...}`` interpolation in strings.
    """
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
    """Materialize targets post-order across annotated dicts and lists.

    Concrete dataclass containers are constructed around materialized children.
    """
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
    # Handle list
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

    # Handle non-dict values
    if not isinstance(value, dict):
        return value

    # Handle typed configclasses
    if _HYDRA_TARGET_KEY in value:
        return _materialize_target_mapping(annotation, value, path=path)

    # Handle ordinary mappings
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
        # Isaac Lab copies inherited annotations into each configclass. Use its
        # original field declarations to find the module that owns the imports.
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
    """Return the element annotation for ``list[T]``, or ``None`` when ``annotation`` is not a list."""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else None
    return None


def _concrete_dataclass_type(annotation: Any) -> type | None:
    """Return a single dataclass type annotation, or ``None`` for unions and non-dataclass fields."""
    if _union_members(annotation) is not None:
        return None
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    return None


def _union_members(annotation: Any) -> tuple[Any, ...] | None:
    """Return union member annotations, or ``None`` when ``annotation`` is not a union."""
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
