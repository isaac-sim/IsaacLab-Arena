# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve and validate structured-config annotations."""

from __future__ import annotations

import dataclasses
import sys
import types
from enum import Enum
from functools import cache
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


@cache
def field_annotation(owner: type, field_name: str) -> Any:
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


def list_element_type(annotation: Any) -> Any | None:
    """Return the element annotation for ``list[T]``, or ``None`` when the annotation is not a list."""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else Any
    return None


def concrete_dataclass_type(annotation: Any) -> type | None:
    """Return a single dataclass type annotation, or ``None`` for unions and non-dataclass fields."""
    if union_members(annotation) is not None:
        return None
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    return None


def union_members(annotation: Any) -> tuple[Any, ...] | None:
    """Return union member annotations, or ``None`` when the annotation is not a union."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return get_args(annotation)
    return None


def annotation_accepts_type(annotation: Any, target_cls: type) -> bool:
    """Return whether ``target_cls`` is compatible with a field annotation."""
    members = union_members(annotation)
    if members is not None:
        return any(annotation_accepts_type(member, target_cls) for member in members)
    return isinstance(annotation, type) and issubclass(target_cls, annotation)


def annotation_accepts_value(annotation: Any, value: Any) -> bool:
    """Return whether a plain value is compatible with an annotation."""
    members = union_members(annotation)
    if members is not None:
        return any(annotation_accepts_value(member, value) for member in members)

    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if origin is Literal:
            return value in args
        try:
            value_matches_origin = isinstance(value, origin)
        except TypeError:
            return False
        if not value_matches_origin:
            return False
        if origin in (list, set) and args:
            return all(annotation_accepts_value(args[0], item) for item in value)
        if origin is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:
                return all(annotation_accepts_value(args[0], item) for item in value)
            return len(value) == len(args) and all(
                annotation_accepts_value(item_type, item) for item_type, item in zip(args, value, strict=True)
            )
        return True

    if annotation in (int, float) and isinstance(value, bool):
        return False
    return annotation is Any or (isinstance(annotation, type) and isinstance(value, annotation))


def convert_override_value(annotation: Any, value: Any) -> Any:
    """Convert one serialized override value to a safe annotation-compatible runtime value."""
    members = union_members(annotation)
    if members is not None:
        for member in members:
            converted = convert_override_value(member, value)
            if annotation_accepts_value(member, converted):
                return converted
        return value

    origin = get_origin(annotation)
    if origin in (set, tuple) and isinstance(value, (list, tuple, set)):
        args = get_args(annotation)
        if origin is set:
            element_type = args[0] if args else Any
            return {convert_override_value(element_type, item) for item in value}
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(convert_override_value(args[0], item) for item in value)
        if args and len(args) == len(value):
            return tuple(convert_override_value(item_type, item) for item_type, item in zip(args, value, strict=True))
        return tuple(value)

    if isinstance(annotation, type) and issubclass(annotation, Enum) and not isinstance(value, annotation):
        try:
            return annotation(value)
        except (TypeError, ValueError):
            return value

    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def annotation_contains_dataclass(annotation: Any) -> bool:
    """Return whether an annotation contains a dataclass type."""
    members = union_members(annotation)
    if members is not None:
        return any(annotation_contains_dataclass(member) for member in members)
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)
