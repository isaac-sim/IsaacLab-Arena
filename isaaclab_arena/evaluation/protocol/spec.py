# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Versioned Evaluation specification and canonical identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._validation import validate_name
from .requirements import AllOf, AnyOf, PredicateRequirement, RequirementSpec

_SPEC_HASH_DOMAIN = b"isaaclab_arena.evaluation_spec.v1\0"
_REQUIREMENT_TYPES = (PredicateRequirement, AllOf, AnyOf)


def _validate_requirement(requirement: RequirementSpec, field_name: str) -> None:
    if not isinstance(requirement, _REQUIREMENT_TYPES):
        raise TypeError(f"{field_name} must be a declarative requirement")


@dataclass(frozen=True)
class MilestoneSpec:
    """Named progress milestone derived from the same evidence as success."""

    name: str
    """Stable task-local milestone name."""

    requirement: RequirementSpec
    """Condition that marks this milestone as reached."""

    def __post_init__(self) -> None:
        validate_name(self.name, "MilestoneSpec.name")
        _validate_requirement(self.requirement, "MilestoneSpec.requirement")


@dataclass(frozen=True)
class FailureSpec:
    """Named semantic failure condition."""

    code: str
    """Stable machine-readable failure reason code."""

    requirement: RequirementSpec
    """Condition that identifies the failure."""

    stage: str | None = None
    """Optional task stage in which the failure is meaningful."""

    def __post_init__(self) -> None:
        validate_name(self.code, "FailureSpec.code")
        _validate_requirement(self.requirement, "FailureSpec.requirement")
        if self.stage is not None:
            validate_name(self.stage, "FailureSpec.stage")


@dataclass(frozen=True)
class EvaluationSpec:
    """Simulator-independent declaration of task evaluation semantics."""

    success: RequirementSpec
    """Condition that defines task success."""

    milestones: tuple[MilestoneSpec, ...] = ()
    """Ordered progress conditions."""

    failures: tuple[FailureSpec, ...] = ()
    """Semantic failure conditions."""

    protocol_version: str = "1"
    """Version of the Evaluation protocol schema."""

    def __post_init__(self) -> None:
        validate_name(self.protocol_version, "EvaluationSpec.protocol_version")
        _validate_requirement(self.success, "EvaluationSpec.success")

        milestones = tuple(self.milestones)
        failures = tuple(self.failures)
        if not all(isinstance(milestone, MilestoneSpec) for milestone in milestones):
            raise TypeError("EvaluationSpec.milestones must contain MilestoneSpec instances")
        if not all(isinstance(failure, FailureSpec) for failure in failures):
            raise TypeError("EvaluationSpec.failures must contain FailureSpec instances")
        _validate_unique((milestone.name for milestone in milestones), "milestone name")
        _validate_unique((failure.code for failure in failures), "failure code")
        _validate_requirement_declarations(self.success, milestones, failures)

        object.__setattr__(self, "milestones", milestones)
        object.__setattr__(self, "failures", failures)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""
        return {
            "failures": [
                {
                    "code": failure.code,
                    "requirement": _requirement_to_dict(failure.requirement),
                    "stage": failure.stage,
                }
                for failure in self.failures
            ],
            "milestones": [
                {
                    "name": milestone.name,
                    "requirement": _requirement_to_dict(milestone.requirement),
                }
                for milestone in self.milestones
            ],
            "protocol_version": self.protocol_version,
            "success": _requirement_to_dict(self.success),
        }

    def to_canonical_json(self) -> str:
        """Serialize this specification to deterministic canonical JSON."""
        return json.dumps(
            self.to_canonical_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_canonical_bytes(self) -> bytes:
        """Serialize this specification to deterministic UTF-8 bytes."""
        return self.to_canonical_json().encode("utf-8")

    @property
    def spec_hash(self) -> str:
        """SHA-256 identity of evaluation semantics, excluding runtime role bindings."""
        return hashlib.sha256(_SPEC_HASH_DOMAIN + self.to_canonical_bytes()).hexdigest()


def _validate_unique(values, description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {description}: {value!r}")
        seen.add(value)


def _iter_leaf_requirements(requirement: RequirementSpec):
    if isinstance(requirement, PredicateRequirement):
        yield requirement
        return
    for child in requirement.requirements:
        yield from _iter_leaf_requirements(child)


def _validate_requirement_declarations(
    success: RequirementSpec,
    milestones: tuple[MilestoneSpec, ...],
    failures: tuple[FailureSpec, ...],
) -> None:
    declarations: dict[str, object] = {}
    predicate_revisions: dict[str, str] = {}
    roots = (
        success,
        *(milestone.requirement for milestone in milestones),
        *(failure.requirement for failure in failures),
    )
    for root in roots:
        for leaf in _iter_leaf_requirements(root):
            existing = declarations.setdefault(leaf.name, leaf.predicate)
            if existing != leaf.predicate:
                raise ValueError(f"Requirement name {leaf.name!r} is reused with a different predicate")
            predicate_id = leaf.predicate.predicate_id
            predicate_version = leaf.predicate.predicate_version
            existing_version = predicate_revisions.setdefault(predicate_id, predicate_version)
            if existing_version != predicate_version:
                raise ValueError(
                    f"Predicate {predicate_id!r} uses multiple revisions: {existing_version!r} and"
                    f" {predicate_version!r}"
                )


def _requirement_to_dict(requirement: RequirementSpec) -> dict[str, Any]:
    if isinstance(requirement, PredicateRequirement):
        return {
            "name": requirement.name,
            "predicate": {
                "id": requirement.predicate.predicate_id,
                "parameters": _json_to_builtin(requirement.predicate.parameters),
                "roles": {name: role.name for name, role in requirement.predicate.roles.items()},
                "version": requirement.predicate.predicate_version,
            },
            "type": "predicate",
        }
    if isinstance(requirement, AllOf):
        return {
            "requirements": [_requirement_to_dict(child) for child in requirement.requirements],
            "type": "all_of",
        }
    if isinstance(requirement, AnyOf):
        return {
            "requirements": [_requirement_to_dict(child) for child in requirement.requirements],
            "type": "any_of",
        }
    raise TypeError(f"Unsupported requirement type: {type(requirement).__name__}")


def _json_to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_to_builtin(item) for item in value]
    return value
