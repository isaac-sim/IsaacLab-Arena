# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-semantic result contracts for the Evaluation protocol."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from ._validation import validate_name
from .truth import TruthValue


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


class EvaluationOutcome(StrEnum):
    """Task-semantic outcome, independent of episode completion cause."""

    SUCCESS = "success"
    FAILURE = "failure"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class RequirementResult:
    """Observed truth value of one named predicate requirement."""

    name: str
    """Requirement name from the Evaluation specification."""

    value: TruthValue
    """Current three-valued result."""

    def __post_init__(self) -> None:
        validate_name(self.name, "RequirementResult.name")
        if not isinstance(self.value, TruthValue):
            raise TypeError("RequirementResult.value must be a TruthValue")


@dataclass(frozen=True)
class EvaluationResult:
    """Pure task-semantic snapshot produced from an Evaluation specification.

    This contract intentionally does not classify timeout, shutdown, or other
    episode completion causes. A runtime integration may wrap this snapshot
    after its lifecycle and precedence policies have been defined.
    """

    outcome: EvaluationOutcome
    """Current task-semantic outcome."""

    progress: float
    """Normalized progress derived from declared milestones."""

    spec_hash: str
    """Canonical hash of the Evaluation specification."""

    predicate_revisions: Mapping[str, str]
    """Predicate IDs mapped to the semantic revisions used for this result."""

    completion_step: int | None = None
    """First simulation or control step at which the success requirement became true."""

    binding_hash: str | None = None
    """Hash of separately compiled role bindings, when available."""

    highest_milestone: str | None = None
    """Highest reached milestone name, when any."""

    failure_code: str | None = None
    """Selected semantic failure reason, when outcome is failure."""

    failure_stage: str | None = None
    """Task stage associated with the selected semantic failure."""

    requirement_results: Mapping[str, RequirementResult] = field(default_factory=dict)
    """Per-requirement truth values used to explain the result."""

    protocol_version: str = "1"
    """Evaluation protocol version that produced this result."""

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EvaluationOutcome):
            raise TypeError("EvaluationResult.outcome must be an EvaluationOutcome")
        if isinstance(self.progress, bool) or not isinstance(self.progress, (int, float)):
            raise TypeError("EvaluationResult.progress must be a finite number")
        if not math.isfinite(self.progress):
            raise TypeError("EvaluationResult.progress must be a finite number")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("EvaluationResult.progress must be between 0.0 and 1.0")
        object.__setattr__(self, "progress", float(self.progress))
        if self.completion_step is not None:
            if isinstance(self.completion_step, bool) or not isinstance(self.completion_step, int):
                raise TypeError("EvaluationResult.completion_step must be an integer or None")
            if self.completion_step < 0:
                raise ValueError("EvaluationResult.completion_step cannot be negative")
        if self.outcome is EvaluationOutcome.SUCCESS and self.completion_step is None:
            raise ValueError("Success outcome requires a completion_step")
        _validate_sha256(self.spec_hash, "EvaluationResult.spec_hash")
        if self.binding_hash is not None:
            _validate_sha256(self.binding_hash, "EvaluationResult.binding_hash")
        validate_name(self.protocol_version, "EvaluationResult.protocol_version")
        if self.highest_milestone is not None:
            validate_name(self.highest_milestone, "EvaluationResult.highest_milestone")
        if self.failure_code is not None:
            validate_name(self.failure_code, "EvaluationResult.failure_code")
        if self.failure_stage is not None:
            validate_name(self.failure_stage, "EvaluationResult.failure_stage")

        if self.outcome is EvaluationOutcome.FAILURE and self.failure_code is None:
            raise ValueError("Failure outcome requires a failure_code")
        if self.outcome is not EvaluationOutcome.FAILURE and self.failure_code is not None:
            raise ValueError("failure_code is only valid for a failure outcome")
        if self.failure_stage is not None and self.failure_code is None:
            raise ValueError("failure_stage requires a failure_code")

        predicate_revisions = dict(self.predicate_revisions)
        if not predicate_revisions:
            raise ValueError("EvaluationResult.predicate_revisions cannot be empty")
        for predicate_id, predicate_version in predicate_revisions.items():
            validate_name(predicate_id, "EvaluationResult.predicate_revisions key")
            validate_name(predicate_version, "EvaluationResult predicate revision")
        object.__setattr__(self, "predicate_revisions", MappingProxyType(predicate_revisions))

        requirement_results = dict(self.requirement_results)
        for name, requirement_result in requirement_results.items():
            validate_name(name, "EvaluationResult.requirement_results key")
            if not isinstance(requirement_result, RequirementResult):
                raise TypeError("EvaluationResult.requirement_results values must be RequirementResult instances")
            if name != requirement_result.name:
                raise ValueError("EvaluationResult.requirement_results keys must match RequirementResult.name")
        object.__setattr__(self, "requirement_results", MappingProxyType(requirement_results))

    def __hash__(self) -> int:
        """Hash this immutable result independently of mapping insertion order."""
        return hash((
            self.outcome,
            self.completion_step,
            self.progress,
            self.spec_hash,
            tuple(sorted(self.predicate_revisions.items())),
            self.binding_hash,
            self.highest_milestone,
            self.failure_code,
            self.failure_stage,
            tuple(sorted(self.requirement_results.items())),
            self.protocol_version,
        ))
