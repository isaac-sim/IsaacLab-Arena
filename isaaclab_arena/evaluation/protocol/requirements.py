# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative requirements for the Evaluation protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from ._validation import validate_name
from .truth import TruthValue, all_truth, any_truth

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def _freeze_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        frozen_mapping: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            frozen_mapping[key] = _freeze_json(item)
        return MappingProxyType(frozen_mapping)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("PredicateSpec.parameters values must be JSON-compatible")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("PredicateSpec.parameters cannot contain non-finite floats")
    return value


def _json_identity(value: JsonValue):
    if isinstance(value, Mapping):
        return ("object", tuple(sorted((key, _json_identity(item)) for key, item in value.items())))
    if isinstance(value, tuple):
        return ("array", tuple(_json_identity(item) for item in value))
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("integer", value)
    if isinstance(value, float):
        return ("number", value.hex())
    return ("string", value)


@dataclass(frozen=True)
class RoleRef:
    """Semantic task role resolved to a scene entity by a later binding stage."""

    name: str
    """Task-local semantic role name."""

    def __post_init__(self) -> None:
        validate_name(self.name, "RoleRef.name")


@dataclass(frozen=True)
class PredicateSpec:
    """Stable predicate identity and its fully resolved declaration parameters."""

    predicate_id: str
    """Stable, namespaced predicate identifier."""

    predicate_version: str
    """Version of the predicate semantics."""

    roles: Mapping[str, RoleRef] = field(default_factory=dict)
    """Predicate argument names mapped to semantic task roles."""

    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    """Resolved JSON-compatible parameters that affect predicate semantics."""

    def __post_init__(self) -> None:
        validate_name(self.predicate_id, "PredicateSpec.predicate_id")
        validate_name(self.predicate_version, "PredicateSpec.predicate_version")

        roles = dict(self.roles)
        for argument, role in roles.items():
            validate_name(argument, "PredicateSpec role argument")
            if not isinstance(role, RoleRef):
                raise TypeError("PredicateSpec.roles values must be RoleRef instances")

        parameters: dict[str, JsonValue] = {}
        for name, value in self.parameters.items():
            validate_name(name, "PredicateSpec parameter name")
            parameters[name] = _freeze_json(value)

        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def __hash__(self) -> int:
        """Hash this immutable declaration independently of mapping insertion order."""
        return hash((
            self.predicate_id,
            self.predicate_version,
            tuple(sorted(self.roles.items())),
            _json_identity(self.parameters),
        ))

    def __eq__(self, other: object) -> bool:
        """Compare declarations using JSON type-sensitive parameter identity.

        Args:
            other: Candidate declaration to compare.
        """
        if not isinstance(other, PredicateSpec):
            return NotImplemented
        return (
            self.predicate_id == other.predicate_id
            and self.predicate_version == other.predicate_version
            and self.roles == other.roles
            and _json_identity(self.parameters) == _json_identity(other.parameters)
        )


@dataclass(frozen=True)
class PredicateRequirement:
    """Named leaf requirement backed by one predicate declaration."""

    name: str
    """Stable task-local name used to associate evidence with this requirement."""

    predicate: PredicateSpec
    """Predicate whose evidence determines this requirement."""

    def __post_init__(self) -> None:
        validate_name(self.name, "PredicateRequirement.name")
        if not isinstance(self.predicate, PredicateSpec):
            raise TypeError("PredicateRequirement.predicate must be a PredicateSpec")


def _normalize_child_requirements(
    requirements: tuple[RequirementSpec, ...],
    operator_name: str,
) -> tuple[RequirementSpec, ...]:
    """Normalize and validate a logical operator's children.

    Args:
        requirements: Child requirement declarations.
        operator_name: Logical operator name to include in validation errors.
    """
    normalized = tuple(requirements)
    if not normalized:
        raise ValueError(f"{operator_name} requires at least one child requirement")
    if not all(isinstance(requirement, (PredicateRequirement, AllOf, AnyOf)) for requirement in normalized):
        raise TypeError(f"{operator_name}.requirements must contain declarative requirements")
    return normalized


@dataclass(frozen=True)
class AllOf:
    """Requirement satisfied only when every child is true."""

    requirements: tuple[RequirementSpec, ...]
    """Child requirements."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirements", _normalize_child_requirements(self.requirements, "AllOf"))


@dataclass(frozen=True)
class AnyOf:
    """Requirement satisfied when at least one child is true."""

    requirements: tuple[RequirementSpec, ...]
    """Child requirements."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirements", _normalize_child_requirements(self.requirements, "AnyOf"))


RequirementSpec: TypeAlias = PredicateRequirement | AllOf | AnyOf


def evaluate_requirement(requirement: RequirementSpec, evidence: Mapping[str, TruthValue]) -> TruthValue:
    """Reduce named predicate evidence through a declarative requirement tree.

    Missing evidence is explicitly represented as ``TruthValue.UNKNOWN``.

    Args:
        requirement: Requirement tree to reduce.
        evidence: Truth values keyed by predicate requirement name.
    """
    if isinstance(requirement, PredicateRequirement):
        value = evidence.get(requirement.name, TruthValue.UNKNOWN)
        if not isinstance(value, TruthValue):
            raise TypeError("Requirement evidence values must be TruthValue instances")
        return value
    if isinstance(requirement, AllOf):
        return all_truth(evaluate_requirement(child, evidence) for child in requirement.requirements)
    if isinstance(requirement, AnyOf):
        return any_truth(evaluate_requirement(child, evidence) for child in requirement.requirements)
    raise TypeError(f"Unsupported requirement type: {type(requirement).__name__}")
