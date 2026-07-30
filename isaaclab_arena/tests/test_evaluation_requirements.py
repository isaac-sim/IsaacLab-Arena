# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for declarative Evaluation requirements."""

from dataclasses import FrozenInstanceError

import pytest

from isaaclab_arena.evaluation.protocol import (
    AllOf,
    AnyOf,
    PredicateRequirement,
    PredicateSpec,
    RoleRef,
    TruthValue,
    evaluate_requirement,
)


def _predicate_requirement(name: str) -> PredicateRequirement:
    return PredicateRequirement(
        name=name,
        predicate=PredicateSpec(
            predicate_id="arena.spatial.object_in_receptacle",
            predicate_version="1",
            roles={"subject": RoleRef("object"), "receptacle": RoleRef("target")},
            parameters={"xy_tolerance_m": 0.02},
        ),
    )


def test_predicate_declaration_is_immutable_and_detached_from_input_mappings():
    roles = {"subject": RoleRef("object")}
    parameter_bounds = [0.05, 0.2]
    parameter_options = {"axis": "z"}
    parameters = {
        "minimum_height_m": 0.05,
        "bounds_m": parameter_bounds,
        "options": parameter_options,
    }
    predicate = PredicateSpec(
        predicate_id="arena.spatial.lifted",
        predicate_version="1",
        roles=roles,
        parameters=parameters,
    )

    roles["subject"] = RoleRef("replacement")
    parameters["minimum_height_m"] = 1.0
    parameter_bounds[0] = 1.0
    parameter_options["axis"] = "x"

    assert predicate.roles["subject"] == RoleRef("object")
    assert predicate.parameters["minimum_height_m"] == 0.05
    assert predicate.parameters["bounds_m"] == (0.05, 0.2)
    assert predicate.parameters["options"] == {"axis": "z"}
    assert hash(predicate) == hash(
        PredicateSpec(
            predicate_id="arena.spatial.lifted",
            predicate_version="1",
            roles={"subject": RoleRef("object")},
            parameters={
                "minimum_height_m": 0.05,
                "bounds_m": (0.05, 0.2),
                "options": {"axis": "z"},
            },
        )
    )
    with pytest.raises(TypeError):
        predicate.roles["subject"] = RoleRef("replacement")
    with pytest.raises(FrozenInstanceError):
        predicate.predicate_version = "2"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RoleRef(""),
        lambda: RoleRef(1),
        lambda: PredicateSpec("", "1"),
        lambda: PredicateSpec("arena.spatial.lifted", ""),
        lambda: PredicateSpec("arena.spatial.lifted", "1", roles={"subject": "object"}),
        lambda: PredicateSpec("arena.spatial.lifted", "1", parameters={"threshold": object()}),
        lambda: PredicateSpec("arena.spatial.lifted", "1", parameters={"nested": {1: "invalid"}}),
        lambda: PredicateSpec("arena.spatial.lifted", "1", parameters={"threshold": float("nan")}),
        lambda: PredicateRequirement("", PredicateSpec("arena.spatial.lifted", "1")),
        lambda: AllOf(()),
        lambda: AllOf((object(),)),
        lambda: AnyOf(()),
        lambda: AnyOf((object(),)),
    ],
)
def test_invalid_declarations_fail_at_construction(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_requirement_reducer_evaluates_nested_expression():
    placed = _predicate_requirement("placed")
    released = _predicate_requirement("released")
    stable = _predicate_requirement("stable")
    requirement = AllOf((placed, AnyOf((released, stable))))

    assert (
        evaluate_requirement(
            requirement,
            {
                "placed": TruthValue.TRUE,
                "released": TruthValue.FALSE,
                "stable": TruthValue.TRUE,
            },
        )
        is TruthValue.TRUE
    )
    assert (
        evaluate_requirement(
            requirement,
            {
                "placed": TruthValue.TRUE,
                "released": TruthValue.FALSE,
                "stable": TruthValue.UNKNOWN,
            },
        )
        is TruthValue.UNKNOWN
    )
    assert (
        evaluate_requirement(
            requirement,
            {
                "placed": TruthValue.FALSE,
                "released": TruthValue.UNKNOWN,
                "stable": TruthValue.UNKNOWN,
            },
        )
        is TruthValue.FALSE
    )


def test_missing_predicate_evidence_is_unknown():
    assert evaluate_requirement(_predicate_requirement("placed"), {}) is TruthValue.UNKNOWN


def test_requirement_reducer_rejects_non_truth_evidence():
    with pytest.raises(TypeError):
        evaluate_requirement(_predicate_requirement("placed"), {"placed": "true"})
