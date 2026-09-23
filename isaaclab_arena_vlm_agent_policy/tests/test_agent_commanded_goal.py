# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena_vlm_agent_policy.embodiment_adapter.droid_eef_action_adapter import DroidGoalActionAdapter
from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_goal_policy import (
    AgentCommandedGoalPolicy,
    AgentCommandedGoalPolicyCfg,
    pose_error,
)
from isaaclab_arena_vlm_agent_policy.utils import encoded_intrinsics

POSE = np.array([0.4, 0.0, 0.3, 0.0, 0, 0, 1.0])


def test_standalone_experiment_uses_goal_policy_and_fixed_layout():
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file

    config = load_arena_experiment_from_config_file(
        "isaaclab_arena_environments/experiment_configs/droid_pnp_agent_commanded_goal_experiment.yaml", device="cuda:0"
    )
    goal = config.runs["droid_pnp_agent_commanded_goal"]
    assert len(config.runs) == 1
    assert goal.environment.embodiment == "droid_absolute_ik"
    assert goal.environment.episode_length_s == 70.0
    assert goal.environment_builder.placement_seed == 42
    assert goal.environment_builder.resolve_on_reset is False
    assert isinstance(goal.policy, AgentCommandedGoalPolicyCfg)
    assert goal.policy.decision_history == 16


def request(command="move_to", **kwargs):
    return dict(
        command=command,
        position=POSE[:3].tolist() if command == "move_to" else None,
        quaternion_xyzw=None,
        gripper=None,
        steps=12,
        note="Observed target",
        **kwargs,
    )


@pytest.mark.parametrize("max_position_step_m,max_rotation_step_rad", [(0.004, 0.04), (0.008, 0.08)])
def test_reference_limits_and_shortest_rotation(max_position_step_m, max_rotation_step_rad):
    target = POSE.copy()
    target[:3] += [0.1, 0.1, 0.0]
    target[3:7] = [0, 0, -np.sin(0.1), -np.cos(0.1)]
    adapter = DroidGoalActionAdapter(
        max_position_step_m=max_position_step_m, max_rotation_step_rad=max_rotation_step_rad
    )
    original_target = target.copy()
    advanced = adapter.compute_next_reference_pose(POSE, target)
    np.testing.assert_array_equal(target, original_target)
    translation, rotation = pose_error(advanced, POSE)
    assert translation == pytest.approx(max_position_step_m)
    assert rotation == pytest.approx(max_rotation_step_rad)
    for _ in range(40):
        advanced = adapter.compute_next_reference_pose(advanced, target)
    np.testing.assert_allclose(pose_error(advanced, target), [0, 0], atol=1e-7)


def test_reference_interpolates_between_different_rotation_axes():
    current = POSE.copy()
    target = POSE.copy()
    current[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    target[3:7] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    adapter = DroidGoalActionAdapter(max_rotation_step_rad=np.pi / 3)
    advanced = adapter.compute_next_reference_pose(current, target)
    expected = (current[3:7] + target[3:7]) / np.sqrt(3)
    np.testing.assert_allclose(advanced[3:7], expected, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.01])
@pytest.mark.parametrize("sign", [-1, 1])
def test_reference_reaches_nearby_rotation_without_overshoot(angle, sign):
    target = POSE.copy()
    target[3:7] = sign * np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])
    advanced = DroidGoalActionAdapter().compute_next_reference_pose(POSE, target)
    np.testing.assert_allclose(advanced[3:7], target[3:7], atol=1e-12)
    assert np.isfinite(advanced).all()


def test_intrinsics_match_encoded_image_coordinates():
    matrix = np.array([[1000, 0, 640], [0, 900, 360], [0, 0, 1]])
    scaled, shape = encoded_intrinsics(matrix, (720, 1280), 384)
    assert shape == [216, 384]
    point = np.array([0.2, -0.1, 1.0])
    np.testing.assert_allclose((np.array(scaled) @ point)[:2], (matrix @ point)[:2] * 0.3)
    unchanged, shape = encoded_intrinsics(matrix, (216, 384), 1024)
    np.testing.assert_array_equal(unchanged, matrix)
    assert shape == [216, 384]


@pytest.mark.parametrize("field", ["max_position_step_m", "max_rotation_step_rad"])
@pytest.mark.parametrize("value", [0, -0.01, float("nan"), float("inf")])
def test_invalid_reference_limits(field, value):
    with pytest.raises(AssertionError, match=field):
        DroidGoalActionAdapter(**{field: value})


@pytest.mark.parametrize("fault", ["position", "steps", "gripper", "quaternion_xyzw", "hold", "missing"])
def test_invalid_goal_never_reaches_controller(fault):
    payload = request()
    if fault == "position":
        payload[fault] = [0.3, 0, 0.09]
    elif fault == "steps":
        payload[fault] = 241
    elif fault == "gripper":
        payload[fault] = 0.5
    elif fault == "quaternion_xyzw":
        payload[fault] = [0, 0, 0, 0]
    elif fault == "hold":
        payload["command"] = "wait"
    else:
        del payload["note"]
    with pytest.raises(AssertionError):
        DroidGoalActionAdapter().decode_command(payload, {"eef_pose_root_xyz_xyzw": POSE, "last_gripper_command": 1})


def test_hold_preserves_measured_pose_and_commanded_gripper():
    decoded = DroidGoalActionAdapter().decode_command(
        request("wait"), {"eef_pose_root_xyz_xyzw": POSE, "last_gripper_command": 1}
    )
    np.testing.assert_allclose(decoded[:7], POSE)
    assert decoded[7] == 1 and decoded[9] == 0


@pytest.mark.parametrize("tracks", [True, False])
@pytest.mark.parametrize("decision_history", [0, 16])
def test_goal_completion_history_and_reset(monkeypatch, tmp_path, tracks, decision_history):
    import isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy as agent_module

    requests = []

    def create(**kwargs):
        # Preserve each request before correction/history handling can mutate it.
        requests.append(json.loads(json.dumps(kwargs)))
        payload = request()
        payload["position"] = [0.408, 0, 0.3]
        payload["steps"] = 8
        message = SimpleNamespace(content=json.dumps(payload), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    monkeypatch.setattr(
        agent_module,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None
        ),
    )
    monkeypatch.setenv("TEST_GOAL_KEY", "test-only")
    measured = [POSE.copy()]

    class Adapter(agent_module.VLMObservationAdapter):
        camera_keys = ("camera",)

        def extract_proprioception(self, env, observation):
            return [{"eef_pose_root_xyz_xyzw": measured[0].tolist()}]

    policy = AgentCommandedGoalPolicy(
        AgentCommandedGoalPolicyCfg(
            api_key_env_var="TEST_GOAL_KEY",
            trace_directory=str(tmp_path),
            decision_history=decision_history,
            system_prompt='Goal experiment prompt with literal JSON: {"command": "wait"}.\nUse current images.',
        ),
        observation_adapter=Adapter(image_max_edge=384),
        action_adapter=DroidGoalActionAdapter(),
    )
    policy.action_adapter.validate_environment = lambda env: None
    policy.set_task_description("Pick up the object")
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=1, device="cpu", step_dt=1 / 15))
    observation = {"camera_obs": {"camera": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)}}
    try:
        count = 6 if tracks else 8
        for _ in range(count):
            action = policy.get_action(env, observation)[0].numpy()
            if tracks:
                measured[0] = action[:7].astype(float)
        assert len(requests) == 1
        policy.get_action(env, observation)
        assert len(requests) == 2
        messages = requests[-1]["messages"]
        assert [x["role"] for x in messages] == (
            ["system", "user", "assistant", "user"] if decision_history else ["system", "user"]
        )
        assert messages[0]["content"] == policy.config.system_prompt
        state = json.loads(messages[-1]["content"][0]["text"])["proprioception"]
        assert state["previous_command_result"]["outcome"] == ("converged" if tracks else "incomplete")
        assert state["previous_command_result"]["executed_steps"] == count
        policy.reset(torch.tensor([0]))
        assert policy._goals == [None] and not policy._decisions
        policy.get_action(env, observation)
        assert len(requests[-1]["messages"]) == 2
    finally:
        policy.close()
