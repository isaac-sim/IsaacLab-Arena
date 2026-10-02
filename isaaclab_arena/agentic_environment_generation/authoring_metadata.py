# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static authoring declarations on registered classes and factories; no asset construction."""

from __future__ import annotations

import inspect
import json
import sys
import types
from contextlib import suppress
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_type import ObjectType


@dataclass(frozen=True)
class ParameterMetadata:
    """Explicit semantics supplementing a constructor's type and default."""

    units: str | None = None
    """Units of the parameter; omitted when not declared."""
    minimum: float | None = None
    """Inclusive lower bound, when declared by the component."""
    maximum: float | None = None
    """Inclusive upper bound, when declared by the component."""
    description: str | None = None
    """Meaning or usage constraints for this parameter."""
    reference: bool | None = None
    """Whether the serialized value names an environment graph node."""


@dataclass(frozen=True)
class AuthoringMetadata:
    """Optional ``authoring_metadata`` attribute on a registered class or factory."""

    parameters: dict[str, ParameterMetadata] = field(default_factory=dict)
    """Semantics for named constructor parameters."""
    configuration: dict[str, ParameterMetadata] = field(default_factory=dict)
    """Semantics for relative dotted configuration paths, such as ``sampler_cfg.low``."""
    provides: tuple[str, ...] = ()
    """Capabilities provided in addition to inherited affordance classes."""
    requires: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Capabilities required on each named graph-node argument; all must hold."""
    constraints: tuple[str, ...] = ()
    """Additional authoring constraints; documentation, not executable predicates."""
    reset_semantics: str | None = None
    """What is reset, when it is reset, and relevant preserved state."""


def get_authoring_metadata(component: Any) -> AuthoringMetadata:
    """Read the class-local declaration without constructing the component."""
    metadata = getattr(component, "authoring_metadata", AuthoringMetadata())
    assert isinstance(metadata, AuthoringMetadata), "authoring_metadata must be an AuthoringMetadata instance"
    return metadata


def provided_capabilities(component: Any) -> list[str]:
    """Return declared capabilities and inherited affordance class names."""
    capabilities = set(get_authoring_metadata(component).provides)
    capabilities.update(object_type_capabilities(getattr(component, "object_type", None)))
    if isinstance(component, type):
        for base in component.__mro__:
            if base is not AffordanceBase and issubclass(base, AffordanceBase):
                # The asset itself is not a separate affordance.
                if not issubclass(base, Asset):
                    capabilities.add(base.__name__)
    return sorted(capabilities)


def object_type_capabilities(object_type: ObjectType | str | None) -> set[str]:
    """Expose declared physics interfaces without inferring geometry or constructing an asset."""
    if object_type == ObjectType.RIGID:
        return {"rigid", "root_frame"}
    if object_type in (ObjectType.BASE, ObjectType.ARTICULATION):
        return {"root_frame"}
    return set()


def constructor_parameters(component: Any, excluded: frozenset[str] = frozenset()) -> dict[str, dict[str, Any]]:
    """Describe declared keyword parameters using JSON Schema and optional authoring metadata.

    Args:
        component: Registered class or callable factory; it is never called.
        excluded: Parameters represented by structural graph fields instead of ``params``.

    Returns:
        Parameter names mapped to schemas with ``required`` and optional ``default`` keys.
        Unresolved annotations retain ``x-python-type`` without inventing a JSON type.
    """
    constructor = component.__init__ if isinstance(component, type) else component
    signature = inspect.signature(constructor)
    annotations = _type_hints(constructor, signature)
    metadata = get_authoring_metadata(component)
    parameters = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or name in excluded:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        schema = annotation_schema(annotations.get(name, parameter.annotation))
        schema["required"] = parameter.default is inspect.Parameter.empty
        if parameter.default is not inspect.Parameter.empty:
            try:
                schema["default"] = json.loads(json.dumps(parameter.default, default=_json_default, allow_nan=False))
            except (TypeError, ValueError):
                schema["x-default-unavailable"] = True
        declaration = metadata.parameters.get(name)
        if declaration is not None:
            for key, value in asdict(declaration).items():
                if value is not None:
                    schema[{"units": "x-units", "reference": "x-arena-reference"}.get(key, key)] = value
        requirements = set(schema.get("x-required-capabilities", [])) | set(metadata.requires.get(name, ()))
        if requirements:
            schema["x-required-capabilities"] = sorted(requirements)
            schema["x-arena-reference"] = True
        parameters[name] = schema
    unknown = (set(metadata.parameters) | set(metadata.requires)) - set(parameters) - excluded
    assert not unknown, f"Authoring metadata names undeclared parameters: {sorted(unknown)}"
    if isinstance(component, type) and is_dataclass(component):
        for cfg_field in fields(component):
            if cfg_field.name in parameters and cfg_field.default_factory is not MISSING:
                factory = cfg_field.default_factory
                parameters[cfg_field.name]["default_factory"] = getattr(factory, "__qualname__", type(factory).__name__)
    return parameters


def accepts_extra_parameters(component: Any) -> bool:
    """Return whether the declared constructor accepts additional keyword arguments."""
    constructor = component.__init__ if isinstance(component, type) else component
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(constructor).parameters.values())


def annotation_schema(annotation: Any) -> dict[str, Any]:
    """Describe a Python annotation's serialized shape without constructing its type."""
    if annotation in (Any, inspect.Parameter.empty):
        return {}
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return annotation_schema(arguments[0])
    if origin in (Union, types.UnionType):
        choices = [annotation_schema(argument) for argument in arguments]
        schema = {"anyOf": choices}
        references = [choice for choice in choices if choice.get("x-arena-reference")]
        if references:
            schema["x-arena-reference"] = True
            requirements = set.intersection(*(set(choice.get("x-required-capabilities", [])) for choice in references))
            if requirements:
                schema["x-required-capabilities"] = sorted(requirements)
        return schema
    if origin is Literal:
        return {"enum": list(arguments)}
    if annotation is None or annotation is type(None):
        return {"type": "null"}
    if annotation is Path:
        return {"type": "string", "format": "path"}
    scalar_types = {str: "string", bool: "boolean", int: "integer", float: "number"}
    if annotation in scalar_types:
        return {"type": scalar_types[annotation]}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"enum": [member.value for member in annotation]}
    if isinstance(annotation, type) and issubclass(annotation, (Asset, AffordanceBase)):
        schema = {"type": "string", "x-arena-reference": True, "x-python-type": annotation.__name__}
        if issubclass(annotation, AffordanceBase) and not issubclass(annotation, Asset):
            schema["x-required-capabilities"] = [annotation.__name__]
        return schema
    if origin in (list, tuple) or annotation in (list, tuple):
        schema = {"type": "array"}
        if origin is tuple and arguments and arguments[-1] is not Ellipsis:
            schema.update(
                prefixItems=[annotation_schema(arg) for arg in arguments],
                minItems=len(arguments),
                maxItems=len(arguments),
            )
        elif arguments:
            schema["items"] = annotation_schema(arguments[0])
            if origin is list and schema["items"].get("x-arena-reference"):
                schema["x-arena-reference-collection"] = True
        return schema
    if origin is dict or annotation is dict:
        return {"type": "object", **({"additionalProperties": annotation_schema(arguments[1])} if arguments else {})}
    return {"x-python-type": getattr(annotation, "__name__", str(annotation))}


def _type_hints(constructor: Any, signature: inspect.Signature) -> dict[str, Any]:
    try:
        return get_type_hints(constructor, include_extras=True)
    except (NameError, TypeError):
        module = sys.modules.get(getattr(constructor, "__module__", ""))
        namespace = vars(module) if module is not None else {}
        resolved = {}
        for name, parameter in signature.parameters.items():
            annotation = parameter.annotation
            if isinstance(annotation, str):
                with suppress(NameError, SyntaxError, TypeError):
                    annotation = eval(annotation, namespace)  # noqa: S307 — trusted internal annotations
            resolved[name] = annotation
        return resolved


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"No JSON representation for {type(value).__name__}")
