# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_chunk_policy import (
    AgentCommandedChunkPolicy,
    AgentCommandedChunkPolicyCfg,
)
from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_goal_policy import (
    AgentCommandedGoalPolicy,
    AgentCommandedGoalPolicyCfg,
)
from isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy import AgentActionAdapter, VLMObservationAdapter


class OtherRobotObservationAdapter(VLMObservationAdapter):
    camera_keys = ("overhead",)

    def extract_tracking_state(self, env, observation):
        return [{"eef_pose_root_xyz_xyzw": [0.4, 0, 0.3, 0, 0, 0, 1]}]

    def extract_proprioception(self, env, observation):
        return [{**self.extract_tracking_state(env, observation)[0], "camera_calibration": {"overhead": "test"}}]


class OtherRobotActionAdapter(AgentActionAdapter):
    response_schema = {"type": "object"}

    def __init__(self, step_fraction=0.5):
        self.step_fraction = step_fraction
        self.reference_steps = 0

    def compute_next_reference_pose(self, reference_pose, goal_pose):
        self.reference_steps += 1
        reference = goal_pose.copy()
        reference[:3] = reference_pose[:3] + self.step_fraction * (goal_pose[:3] - reference_pose[:3])
        return reference

    def validate_environment(self, env):
        assert env.unwrapped.robot_kind == "other"

    def decode_command(self, payload, proprioception):
        return np.asarray(payload["command"], dtype=np.float32)

    def command_to_action(self, env, command):
        # This robot expects the gripper first and uses -1 for open.
        return torch.tensor([2 * command[7] - 1, *command[:7]], dtype=torch.float32)


@pytest.mark.parametrize("kind", ["goal", "chunk"])
@pytest.mark.parametrize("selection", ["config", "runtime"])
def test_policy_runs_with_other_robot_adapters(monkeypatch, tmp_path, kind, selection):
    import isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy as agent_module

    command = [0.4, 0, 0.3, 0, 0, 0, 1, 0]
    decoded = [0.44, *command[1:], 6, 1] if kind == "goal" else command * 15
    calls = []

    def create(**kwargs):
        state = json.loads(kwargs["messages"][-1]["content"][0]["text"])["proprioception"]
        assert state["camera_calibration"] == {"overhead": "test"}
        calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps({"command": decoded}), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    monkeypatch.setenv("OTHER_ROBOT_TEST_KEY", "test-only")
    monkeypatch.setattr(
        agent_module,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None
        ),
    )
    policy_type, cfg_type = (
        (AgentCommandedGoalPolicy, AgentCommandedGoalPolicyCfg)
        if kind == "goal"
        else (AgentCommandedChunkPolicy, AgentCommandedChunkPolicyCfg)
    )
    config = cfg_type(
        system_prompt="Use the other robot's calibrated camera.",
        observation_adapter=f"{__name__}.OtherRobotObservationAdapter" if selection == "config" else "",
        action_adapter=f"{__name__}.OtherRobotActionAdapter" if selection == "config" else "",
        action_adapter_kwargs={"step_fraction": 0.25} if selection == "config" else {},
        api_key_env_var="OTHER_ROBOT_TEST_KEY",
        image_max_edge=64,
        trace_directory=str(tmp_path),
    )
    adapters = (
        {}
        if selection == "config"
        else {
            "observation_adapter": OtherRobotObservationAdapter(image_max_edge=64),
            "action_adapter": OtherRobotActionAdapter(step_fraction=0.25),
        }
    )
    policy = policy_type(config, **adapters)
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=1, device="cpu", step_dt=1 / 15, robot_kind="other"))
    observation = {"camera_obs": {"overhead": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)}}
    try:
        assert policy.observation_adapter.image_max_edge == 64
        policy.set_task_description("Move")
        for step in range(2):
            action = policy.get_action(env, observation)
            expected = [-1, *command[:7]]
            if kind == "goal":
                expected[1] = 0.44 - 0.04 * 0.75 ** (step + 1)
            np.testing.assert_allclose(action.numpy()[0], expected, atol=1e-6)
        assert policy.action_adapter.reference_steps == (2 if kind == "goal" else 0)
        assert len(calls) == 1
        policy.reset()
        policy.get_action(env, observation)
        assert len(calls) == 2
    finally:
        policy.close()


@pytest.mark.parametrize("class_path", ["", "builtins.dict"])
def test_missing_or_invalid_adapter_fails_before_inference(class_path):
    config = AgentCommandedGoalPolicyCfg(system_prompt="Test", observation_adapter=class_path)
    with pytest.raises(AssertionError, match="VLMObservationAdapter"):
        AgentCommandedGoalPolicy(config)
