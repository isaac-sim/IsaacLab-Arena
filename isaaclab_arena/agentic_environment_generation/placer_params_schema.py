# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate an agent-facing override schema from placement dataclasses."""

from __future__ import annotations

import dataclasses
import sys
import types
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

_EXCLUDED_PATHS = {
    ("reachability_config", "embodiment"),
    ("solver_params", "strategies"),
}


def build_placer_params_override_schema() -> dict[str, Any]:
    """Return the strict schema projected from ObjectPlacerParams."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    return _nullable_schema(_dataclass_override_schema(ObjectPlacerParams))


def _dataclass_override_schema(dataclass_type: type, path: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build a strict partial-override schema from a dataclass."""
    properties = {}
    for field in dataclasses.fields(dataclass_type):
        field_path = (*path, field.name)
        if field_path in _EXCLUDED_PATHS:
            continue
        annotation = _dataclass_field_annotation(dataclass_type, field.name)
        properties[field.name] = _nullable_schema(_annotation_schema(annotation, field_path))
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _dataclass_field_annotation(dataclass_type: type, field_name: str) -> Any:
    """Resolve one dataclass annotation without touching excluded fields."""
    annotation = dataclass_type.__annotations__[field_name]
    if not isinstance(annotation, str):
        return annotation
    holder = type("_FieldAnnotation", (), {"__annotations__": {"value": annotation}})
    return get_type_hints(
        holder,
        globalns=vars(sys.modules[dataclass_type.__module__]),
        localns=vars(dataclass_type),
    )["value"]


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Allow null to mean that an override field is omitted."""
    if any(option.get("type") == "null" for option in schema.get("anyOf", [])):
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _annotation_schema(annotation: Any, path: tuple[str, ...]) -> dict[str, Any]:
    """Convert one supported dataclass annotation to JSON schema."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return {"anyOf": [_annotation_schema(member, path) for member in get_args(annotation)]}
    if dataclasses.is_dataclass(annotation):
        return _dataclass_override_schema(annotation, path)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        values = [member.value for member in annotation]
        value_type = type(values[0]) if values else str
        return {**_annotation_schema(value_type, path), "enum": values}
    if origin in (list, set, tuple):
        args = get_args(annotation)
        item_type = args[0] if args else Any
        return {"type": "array", "items": _annotation_schema(item_type, path)}
    primitive_types = {bool: "boolean", int: "integer", float: "number", str: "string", type(None): "null"}
    if annotation in primitive_types:
        return {"type": primitive_types[annotation]}
    raise TypeError(f"Unsupported placer override annotation at {'.'.join(path)}: {annotation}")
