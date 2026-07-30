# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for EvaluationSpec validation, serialization, and identity."""

import dataclasses
import json

import pytest

from isaaclab_arena.evaluation.protocol import (
    AllOf,
    EvaluationSpec,
    FailureSpec,
    JsonValue,
    MilestoneSpec,
    PredicateRequirement,
    PredicateSpec,
    RoleRef,
)


def _requirement(
    name: str,
    predicate_id: str,
    *,
    roles: dict[str, RoleRef] | None = None,
    parameters: dict[str, JsonValue] | None = None,
    version: str = "1",
) -> PredicateRequirement:
    return PredicateRequirement(
        name=name,
        predicate=PredicateSpec(
            predicate_id=predicate_id,
            predicate_version=version,
            roles=roles or {},
            parameters=parameters or {},
        ),
    )


def _pick_and_place_spec(*, tolerance_m: float = 0.02, predicate_version: str = "1") -> EvaluationSpec:
    grasped = _requirement(
        "grasped",
        "arena.manipulation.grasped",
        roles={"subject": RoleRef("object"), "agent": RoleRef("robot")},
    )
    placed = _requirement(
        "placed",
        "arena.spatial.object_in_receptacle",
        roles={"receptacle": RoleRef("target"), "subject": RoleRef("object")},
        parameters={"xy_tolerance_m": tolerance_m, "require_release": True},
        version=predicate_version,
    )
    dropped = _requirement(
        "dropped",
        "arena.manipulation.dropped",
        roles={"subject": RoleRef("object")},
        parameters={"minimum_height_m": 0.4},
    )
    return EvaluationSpec(
        success=AllOf((placed,)),
        milestones=(MilestoneSpec("grasped", grasped),),
        failures=(FailureSpec("object_dropped", dropped, stage="transport"),),
    )


def test_evaluation_spec_has_deterministic_canonical_serialization():
    spec = _pick_and_place_spec()

    payload = json.loads(spec.to_canonical_json())

    assert payload == {
        "failures": [{
            "code": "object_dropped",
            "requirement": {
                "name": "dropped",
                "predicate": {
                    "id": "arena.manipulation.dropped",
                    "parameters": {"minimum_height_m": 0.4},
                    "roles": {"subject": "object"},
                    "version": "1",
                },
                "type": "predicate",
            },
            "stage": "transport",
        }],
        "milestones": [{
            "name": "grasped",
            "requirement": {
                "name": "grasped",
                "predicate": {
                    "id": "arena.manipulation.grasped",
                    "parameters": {},
                    "roles": {"agent": "robot", "subject": "object"},
                    "version": "1",
                },
                "type": "predicate",
            },
        }],
        "protocol_version": "1",
        "success": {
            "requirements": [{
                "name": "placed",
                "predicate": {
                    "id": "arena.spatial.object_in_receptacle",
                    "parameters": {"require_release": True, "xy_tolerance_m": 0.02},
                    "roles": {"receptacle": "target", "subject": "object"},
                    "version": "1",
                },
                "type": "predicate",
            }],
            "type": "all_of",
        },
    }
    assert spec.to_canonical_bytes() == spec.to_canonical_json().encode("utf-8")
    assert len(spec.spec_hash) == 64
    assert all(character in "0123456789abcdef" for character in spec.spec_hash)


def test_spec_hash_ignores_mapping_insertion_order():
    first = _requirement(
        "placed",
        "arena.spatial.object_in_receptacle",
        roles={"subject": RoleRef("object"), "receptacle": RoleRef("target")},
        parameters={
            "xy_tolerance_m": 0.02,
            "require_release": True,
            "stability": {"linear_mps": 0.01, "angular_radps": 0.1},
            "axes": ("x", "y"),
        },
    )
    second = _requirement(
        "placed",
        "arena.spatial.object_in_receptacle",
        roles={"receptacle": RoleRef("target"), "subject": RoleRef("object")},
        parameters={
            "axes": ("x", "y"),
            "stability": {"angular_radps": 0.1, "linear_mps": 0.01},
            "require_release": True,
            "xy_tolerance_m": 0.02,
        },
    )

    assert EvaluationSpec(success=first).spec_hash == EvaluationSpec(success=second).spec_hash


def test_spec_hash_changes_when_evaluation_semantics_change():
    baseline = _pick_and_place_spec()

    assert _pick_and_place_spec(tolerance_m=0.03).spec_hash != baseline.spec_hash
    assert _pick_and_place_spec(predicate_version="2").spec_hash != baseline.spec_hash


def test_spec_hash_is_independent_of_external_role_bindings():
    spec = _pick_and_place_spec()
    first_bindings = {"object": "red_cube", "target": "blue_bowl", "robot": "franka"}
    second_bindings = {"object": "green_cube", "target": "tray", "robot": "ur10"}

    assert first_bindings != second_bindings
    assert spec.spec_hash == spec.spec_hash
    assert "red_cube" not in spec.to_canonical_json()
    assert "green_cube" not in spec.to_canonical_json()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EvaluationSpec(
            success=_requirement("placed", "arena.spatial.placed"),
            milestones=(
                MilestoneSpec("progress", _requirement("grasped", "arena.manipulation.grasped")),
                MilestoneSpec("progress", _requirement("lifted", "arena.manipulation.lifted")),
            ),
        ),
        lambda: EvaluationSpec(
            success=_requirement("placed", "arena.spatial.placed"),
            failures=(
                FailureSpec("dropped", _requirement("dropped", "arena.manipulation.dropped")),
                FailureSpec("dropped", _requirement("collision", "arena.safety.collision")),
            ),
        ),
    ],
)
def test_spec_rejects_duplicate_milestone_names_and_failure_codes(factory):
    with pytest.raises(ValueError):
        factory()


def test_spec_rejects_conflicting_reuse_of_requirement_name():
    with pytest.raises(ValueError, match="placed"):
        EvaluationSpec(
            success=_requirement("placed", "arena.spatial.placed"),
            milestones=(
                MilestoneSpec(
                    "placed",
                    _requirement("placed", "arena.spatial.object_in_receptacle"),
                ),
            ),
        )


@pytest.mark.parametrize(("first_value", "second_value"), [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_spec_rejects_type_distinct_json_values_for_one_requirement_name(first_value, second_value):
    first = _requirement("placed", "arena.spatial.placed", parameters={"threshold": first_value})
    second = _requirement("placed", "arena.spatial.placed", parameters={"threshold": second_value})

    assert first.predicate != second.predicate
    assert EvaluationSpec(success=first).spec_hash != EvaluationSpec(success=second).spec_hash
    with pytest.raises(ValueError, match="placed"):
        EvaluationSpec(success=first, milestones=(MilestoneSpec("placed", second),))


def test_spec_rejects_multiple_revisions_of_one_predicate_id():
    with pytest.raises(ValueError, match="arena.spatial.placed"):
        EvaluationSpec(
            success=_requirement("placed_v1", "arena.spatial.placed", version="1"),
            milestones=(
                MilestoneSpec(
                    "placed_v2",
                    _requirement("placed_v2", "arena.spatial.placed", version="2"),
                ),
            ),
        )


def test_evaluation_spec_is_frozen():
    spec = _pick_and_place_spec()

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.protocol_version = "2"
