# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate agent command boundaries and per-environment context isolation."""

import pytest
from pydantic import TypeAdapter, ValidationError

from isaaclab_arena.inference.backend import build_strict_schema
from isaaclab_arena_vlm_agent_policy.commands import ActionChunkCommand, GoalCommand, MoveToCommand, WaitCommand
from isaaclab_arena_vlm_agent_policy.context import AgentContext, AgentContextCfg, ExecutionFeedback


def _goal():
    return {
        "kind": "move_to",
        "target": {
            "reference_frame": "robot_root",
            "controlled_frame": "gripper",
            "position_m": [0.4, 0.0, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "max_steps": 180,
        "gripper": 1.0,
    }


def test_goal_discrimination_and_json_round_trip():
    adapter = TypeAdapter(GoalCommand)
    command = adapter.validate_python(_goal())
    assert isinstance(command, MoveToCommand)
    assert command == adapter.validate_json(command.model_dump_json())
    assert isinstance(adapter.validate_python({"kind": "wait", "steps": 2}), WaitCommand)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "unknown", "steps": 2})


def test_pose_schema_uses_homogeneous_arrays_and_strict_objects():
    schema = build_strict_schema(MoveToCommand)
    pose_schema = schema["$defs"]["PoseTarget"]
    assert pose_schema["additionalProperties"] is False
    for name, length in (("position_m", 3), ("quaternion_xyzw", 4)):
        vector_schema = pose_schema["properties"][name]
        assert vector_schema["items"] == {"type": "number"}
        assert "prefixItems" not in vector_schema
        assert vector_schema["minItems"] == vector_schema["maxItems"] == length


@pytest.mark.parametrize(
    "field,value", [("max_steps", 0), ("max_steps", True), ("max_steps", 1.5), ("gripper", 1.1), ("extra", 1)]
)
def test_invalid_goal_fields_are_rejected(field, value):
    data = _goal()
    data[field] = value
    with pytest.raises(ValidationError):
        MoveToCommand.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quaternion_xyzw", [0, 0, 0, 0]),
        ("quaternion_xyzw", [0, 0, 0, 2]),
        ("position_m", [float("nan"), 0, 0]),
        ("position_m", [float("inf"), 0, 0]),
        ("position_m", [0, 0]),
        ("reference_frame", ""),
    ],
)
def test_invalid_pose_is_rejected(field, value):
    data = _goal()
    data["target"][field] = value
    with pytest.raises(ValidationError):
        MoveToCommand.model_validate(data)


def test_chunk_requires_references_and_explicit_positive_timing():
    with pytest.raises(ValidationError):
        ActionChunkCommand(references=[])
    reference = {"target": _goal()["target"], "duration_steps": 3}
    chunk = ActionChunkCommand(references=[reference])
    assert chunk.references[0].duration_steps == 3
    reference["duration_steps"] = 0
    with pytest.raises(ValidationError):
        ActionChunkCommand(references=[reference])


def test_context_links_execution_and_isolates_environments():
    context = AgentContext()
    context.record_decision(0, "decision-0", WaitCommand(steps=2))
    context.record_execution(0, "decision-0", ExecutionFeedback("succeeded", 2))
    context.record_decision(1, "decision-1", WaitCommand(steps=3))
    assert context.query(0)["events"][1]["decision_id"] == "decision-0"
    context.reset([0])
    assert context.query(0) == {"events": [], "notes": {}}
    assert context.query(1)["events"][0]["decision_id"] == "decision-1"
    context.reset()
    assert context.query(1) == {"events": [], "notes": {}}


def test_context_copies_inputs_and_outputs_and_bounds_events():
    context = AgentContext(AgentContextCfg(max_events=2))
    progress = {"completed": ["approach"]}
    context.record_task_progress(0, progress)
    context.update_notes(0, progress)
    progress["completed"].append("grasp")
    snapshot = context.query(0)
    assert snapshot["events"][0]["progress"]["completed"] == ["approach"]
    snapshot["notes"]["completed"].append("place")
    assert context.query(0)["notes"]["completed"] == ["approach"]
    context.record_decision(0, "one", WaitCommand(steps=1))
    context.record_decision(0, "two", WaitCommand(steps=1))
    assert [event["decision_id"] for event in context.query(0)["events"]] == ["one", "two"]
    context.reset([0])
    assert not context.query(0)["notes"]
