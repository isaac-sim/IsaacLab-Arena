# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure-Python Evaluation result contracts."""

from dataclasses import FrozenInstanceError

import pytest

from isaaclab_arena.evaluation.protocol import EvaluationOutcome, EvaluationResult, RequirementResult, TruthValue

_SPEC_HASH = "a" * 64
_BINDING_HASH = "b" * 64
_PREDICATE_REVISIONS = {
    "arena.manipulation.grasped": "1",
    "arena.spatial.object_in_receptacle": "1",
}


def _success_result(**overrides) -> EvaluationResult:
    values = {
        "outcome": EvaluationOutcome.SUCCESS,
        "completion_step": 42,
        "progress": 1.0,
        "spec_hash": _SPEC_HASH,
        "predicate_revisions": _PREDICATE_REVISIONS,
    }
    values.update(overrides)
    return EvaluationResult(**values)


def test_success_result_carries_semantic_provenance_and_requirement_values():
    requirement_results = {
        "grasped": RequirementResult("grasped", TruthValue.TRUE),
        "placed": RequirementResult("placed", TruthValue.TRUE),
    }

    result = EvaluationResult(
        outcome=EvaluationOutcome.SUCCESS,
        completion_step=42,
        progress=1.0,
        spec_hash=_SPEC_HASH,
        predicate_revisions=_PREDICATE_REVISIONS,
        binding_hash=_BINDING_HASH,
        highest_milestone="grasped",
        requirement_results=requirement_results,
    )
    requirement_results["stable"] = RequirementResult("stable", TruthValue.TRUE)

    assert result.protocol_version == "1"
    assert result.outcome is EvaluationOutcome.SUCCESS
    assert result.completion_step == 42
    assert result.predicate_revisions == _PREDICATE_REVISIONS
    assert result.requirement_results == {
        "grasped": RequirementResult("grasped", TruthValue.TRUE),
        "placed": RequirementResult("placed", TruthValue.TRUE),
    }
    with pytest.raises(TypeError):
        result.requirement_results["stable"] = RequirementResult("stable", TruthValue.TRUE)
    with pytest.raises(TypeError):
        result.predicate_revisions["arena.spatial.object_in_receptacle"] = "2"
    with pytest.raises(FrozenInstanceError):
        result.progress = 0.5


def test_failure_result_carries_reason_without_defining_resolution_precedence():
    result = EvaluationResult(
        outcome=EvaluationOutcome.FAILURE,
        progress=0.5,
        spec_hash=_SPEC_HASH,
        predicate_revisions={"arena.manipulation.dropped": "1"},
        binding_hash=None,
        failure_code="object_dropped",
        failure_stage="transport",
        requirement_results={"dropped": RequirementResult("dropped", TruthValue.TRUE)},
    )

    assert result.failure_code == "object_dropped"
    assert result.failure_stage == "transport"


def test_indeterminate_result_represents_missing_evidence_explicitly():
    result = EvaluationResult(
        outcome=EvaluationOutcome.INDETERMINATE,
        progress=0.0,
        spec_hash=_SPEC_HASH,
        predicate_revisions={"arena.spatial.object_in_receptacle": "1"},
        requirement_results={"placed": RequirementResult("placed", TruthValue.UNKNOWN)},
    )

    assert result.requirement_results["placed"].value is TruthValue.UNKNOWN


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _success_result(completion_step=None),
        lambda: _success_result(completion_step=-1),
        lambda: _success_result(completion_step=True),
        lambda: _success_result(progress=True),
        lambda: _success_result(progress=-0.1),
        lambda: _success_result(progress=1.1),
        lambda: _success_result(spec_hash="not-a-sha256"),
        lambda: _success_result(failure_code="object_dropped"),
        lambda: _success_result(predicate_revisions={}),
        lambda: _success_result(predicate_revisions={"arena.spatial.placed": ""}),
        lambda: EvaluationResult(
            outcome=EvaluationOutcome.FAILURE,
            progress=0.5,
            spec_hash=_SPEC_HASH,
            predicate_revisions={"arena.manipulation.dropped": "1"},
        ),
        lambda: EvaluationResult(
            outcome=EvaluationOutcome.INDETERMINATE,
            progress=0.0,
            spec_hash=_SPEC_HASH,
            predicate_revisions={"arena.spatial.object_in_receptacle": "1"},
            failure_stage="transport",
        ),
        lambda: _success_result(
            requirement_results={"placed": RequirementResult("different_name", TruthValue.TRUE)},
        ),
    ],
)
def test_result_rejects_structurally_inconsistent_states(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RequirementResult("", TruthValue.TRUE),
        lambda: RequirementResult("placed", "true"),
    ],
)
def test_requirement_result_validates_its_public_contract(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()
