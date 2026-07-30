# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python contracts for declaring and reporting task evaluation."""

from .requirements import (
    AllOf,
    AnyOf,
    JsonScalar,
    JsonValue,
    PredicateRequirement,
    PredicateSpec,
    RequirementSpec,
    RoleRef,
    evaluate_requirement,
)
from .result import EvaluationOutcome, EvaluationResult, RequirementResult
from .spec import EvaluationSpec, FailureSpec, MilestoneSpec
from .truth import TruthValue, all_truth, any_truth

__all__ = [
    "AllOf",
    "AnyOf",
    "EvaluationSpec",
    "EvaluationOutcome",
    "EvaluationResult",
    "FailureSpec",
    "JsonScalar",
    "JsonValue",
    "MilestoneSpec",
    "PredicateRequirement",
    "PredicateSpec",
    "RequirementSpec",
    "RequirementResult",
    "RoleRef",
    "TruthValue",
    "all_truth",
    "any_truth",
    "evaluate_requirement",
]
