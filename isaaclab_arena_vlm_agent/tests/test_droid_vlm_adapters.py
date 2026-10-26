# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise absolute adapters with step, chunk, and goal policies."""

import copy
import json
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena_vlm_agent.absolute_action_adapter import AbsoluteChunkActionAdapter, AbsoluteGoalActionAdapter
from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices
from isaaclab_arena_vlm_agent.observation_adapter import ImageStateObservationAdapter
from isaaclab_arena_vlm_agent.pose_target_motion_generator import PoseTargetMotionGenerator
from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMChunkPolicy, VLMGoalPolicy, VLMGoalPolicyCfg, VLMPolicyCfg


@pytest.fixture
def inputs(monkeypatch):
    indices = AbsoluteCommandIndices(
        8, (PoseCommandIndices(slice(0, 7), "robot", "tool"),), (), (GripperCommandIndex(7, "gripper", 0.0, 1.0),)
    )
    for module in ("absolute_action_adapter", "pose_target_motion_generator"):
        monkeypatch.setattr(f"isaaclab_arena_vlm_agent.{module}.read_absolute_command_indices", lambda env: indices)
    embodiment = SimpleNamespace(
        get_observation_cfg=lambda: SimpleNamespace(
            policy=SimpleNamespace(concatenate_terms=False, gripper_pos=SimpleNamespace(func=Mock())),
            camera_obs=SimpleNamespace(front=SimpleNamespace(func=Mock(), params={"data_type": "rgb"})),
        ),
        get_action_cfg=lambda: SimpleNamespace(arm=SimpleNamespace(asset_name="robot", body_name="tool")),
    )
    world = SimpleNamespace(get_body_pose_in_root=Mock(return_value=torch.tensor([[0.4, 0, 0.3, 0, 0, 0, 1]] * 2)))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=2,
            action_dim=8,
            device="cpu",
            arena_world=world,
            observation_manager=SimpleNamespace(
                active_terms={"policy": ["gripper_pos"]}, group_obs_concatenate={"policy": False}
            ),
            action_manager=SimpleNamespace(
                action=torch.zeros(2, 8),
                active_terms=["arm"],
                get_term=lambda name: SimpleNamespace(cfg=embodiment.get_action_cfg().arm),
            ),
            episode_length_buf=torch.zeros(2, dtype=torch.long),
        )
    )
    observation = {
        "policy": {"gripper_pos": torch.zeros(2, 1)},
        "camera_obs": {"front": torch.zeros(2, 4, 4, 3, dtype=torch.uint8)},
    }
    gripper = SimpleNamespace(
        get_closing_error_m=lambda base: torch.zeros(2), get_opening_width_m=lambda world: torch.zeros(2)
    )
    return embodiment, env, observation, gripper


def _target(x=0.6):
    return {"values": [x, 0, 0.3, 0, 0, 0, 1, 0]}


def _policy(mode, inputs, infer):
    embodiment, _, _, gripper = inputs
    config = VLMPolicyCfg(system_prompt="Return absolute robot-root targets.")
    if mode == "step":
        policy_type, adapter = VLMChunkPolicy, AbsoluteChunkActionAdapter(1)
    elif mode == "chunk":
        policy_type, adapter = VLMChunkPolicy, AbsoluteChunkActionAdapter(2)
    else:
        policy_type = VLMGoalPolicy
        adapter = AbsoluteGoalActionAdapter(PoseTargetMotionGenerator(gripper))
        config = VLMGoalPolicyCfg(system_prompt="Return absolute robot-root targets.", max_steps_per_goal=2)
    policy = policy_type(
        config,
        inference_backend=SimpleNamespace(infer=infer),
        observation_adapter=ImageStateObservationAdapter(embodiment),
        action_adapter=adapter,
    )
    policy.set_task_description("Move the gripper")
    return policy


@pytest.mark.parametrize("mode", ["step", "chunk", "goal"])
def test_policy_execution_and_partial_resets(inputs, mode):
    _, env, observation, _ = inputs
    requests = []

    def infer(request):
        requests.append(copy.deepcopy(request))
        if mode == "goal":
            return _target()
        return {"commands": [_target(), _target(0.7)] if mode == "chunk" else [_target()]}

    policy = _policy(mode, inputs, infer)
    first = policy.get_action(env, observation)
    torch.testing.assert_close(first[:, 0], torch.tensor([0.6, 0.6]))
    content = requests[0].messages[1]["content"]
    assert json.loads(content[1]["text"])["policy"]["gripper_pos"] == [0.0]
    assert content[2]["text"] == "camera=front"
    policy.reset(torch.tensor([0]))
    second = policy.get_action(env, observation)
    assert len(requests) == (4 if mode == "step" else 3)
    if mode == "chunk":
        torch.testing.assert_close(second[:, 0], torch.tensor([0.6, 0.7]))


@pytest.mark.parametrize("mode", ["step", "chunk", "goal"])
@pytest.mark.parametrize(
    "values,message",
    [
        ([0] * 7, "Expected 8 values"),
        ([0, 0, 0, 0, 0, 0.71, 0.71, 0], "unit XYZW quaternion"),
        ([0, 0, 0, 0, 0, 0, 1, 0.5], "Value 7 must be"),
        ([float("inf"), 0, 0, 0, 0, 0, 1, 0], "must be finite"),
    ],
)
def test_invalid_commands_request_specific_corrections(inputs, mode, values, message):
    _, env, observation, _ = inputs
    replies = [{"values": values}, _target(), _target(0.8)]
    if mode != "goal":
        replies = [{"commands": [reply, _target()] if mode == "chunk" else [reply]} for reply in replies]
    infer = Mock(side_effect=replies)
    actions = _policy(mode, inputs, infer).get_action(env, observation)
    assert torch.isfinite(actions).all()
    assert infer.call_count == 3
    correction = infer.call_args_list[1].args[0].messages[-1]["content"]
    assert message in correction


def test_pose_goal_requires_explicit_motion_generator():
    with pytest.raises(TypeError):
        AbsoluteGoalActionAdapter()


def test_goal_completion_uses_live_pose_without_prompt_state(inputs):
    _, env, _, _ = inputs
    infer = Mock(return_value=_target())
    policy = _policy("goal", inputs, infer)
    policy.config.max_steps_per_goal = 10
    policy.observation_adapter.extract_vlm_input_from_observation = Mock(return_value=({}, {}))
    policy.get_action(env, {})
    assert infer.call_count == 2
    env.unwrapped.arena_world.get_body_pose_in_root.return_value[0, 0] = 0.6
    env.unwrapped.episode_length_buf += 1
    policy.get_action(env, {})
    assert infer.call_count == 3
