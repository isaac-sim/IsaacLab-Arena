# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise hosted inference with small robot-independent adapters and a mocked client."""

import base64
import copy
import io
import json
import numpy as np
import subprocess
import sys
import textwrap
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, create_model

from isaaclab_arena.inference.backend import InferenceBackend, build_strict_schema
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.image_utils import extract_rgb_images_from_camera_observations
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.observation_adapter import CAMERA_OBSERVATION_GROUP, VLMObservationAdapter
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError
from isaaclab_arena_vlm_agent.vlm_agent_policy import (
    VLMChunkPolicy,
    VLMGoalPolicy,
    VLMGoalPolicyCfg,
    VLMPolicyCfg,
    VLMStepPolicy,
)


class ObservationAdapter(VLMObservationAdapter):
    camera_keys = ("front_view",)

    def __init__(self, include_joint_positions=True):
        self.include_joint_positions = include_joint_positions

    def extract_joint_positions(self, observation, env_id):
        return observation["position"][env_id].tolist()

    def extract_vlm_input_from_observation(self, env, observation, env_id):
        prompt_values = {}
        if self.include_joint_positions:
            prompt_values["joint_positions"] = self.extract_joint_positions(observation, env_id)
        images = extract_rgb_images_from_camera_observations(
            observation[CAMERA_OBSERVATION_GROUP], self.camera_keys, env_id
        )
        return images, prompt_values


class JointTargetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: list[float]


class ActionAdapter(AgentActionAdapter):
    response_model = JointTargetResponse

    def __init__(self, scale=1.0, action_dim=2):
        self.scale = scale
        self.joint_names = tuple(f"joint_{i}" for i in range(action_dim))

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def decode_command(self, response, arena_world, env_id):
        positions = arena_world.get_joint_positions("robot", self.joint_names)[env_id].cpu().numpy()
        target = np.asarray(response.target, dtype=np.float32)
        if target.shape != (len(positions),):
            raise InvalidAgentResponseError("Wrong command dimensions")
        return target - positions

    def command_to_action(self, env, command):
        return super().command_to_action(env, command * self.scale)


def response(raw):
    message = SimpleNamespace(content=raw, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


@pytest.fixture
def backend(monkeypatch):
    import isaaclab_arena.inference.backend as backend_module

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())), close=Mock())
    client.chat.completions.create.return_value = response("OK")
    monkeypatch.setattr(backend_module, "OpenAI", Mock(return_value=client))
    backend = InferenceBackend(api_key="test-only", endpoint="internal", model="test-model", max_retries=1)
    client.chat.completions.create.reset_mock()
    yield backend
    client.close()


def make_policy(backend, **kwargs):
    config = VLMPolicyCfg(
        system_prompt="Return the joint targets using the supplied schema.",
        image_max_width_or_height=8,
        **kwargs,
    )
    policy = VLMStepPolicy(
        config,
        inference_backend=backend,
        observation_adapter=ObservationAdapter(),
        action_adapter=ActionAdapter(),
    )
    policy.set_task_description("Move to the target")
    return policy


def inputs(action_dim=2):
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, device="cpu", action_dim=action_dim))
    positions = torch.arange(2 * action_dim, dtype=torch.float32).reshape(2, action_dim)

    def get_joint_positions(scene_key, joint_names):
        assert scene_key == "robot"
        indices = [int(name.removeprefix("joint_")) for name in joint_names]
        return positions[:, indices].clone()

    env.unwrapped.arena_world = SimpleNamespace(get_joint_positions=Mock(side_effect=get_joint_positions))
    images = torch.zeros((2, 8, 16, 3), dtype=torch.uint8)
    images[0, ..., 0] = 255
    images[1, ..., 2] = 255
    return env, {"position": positions, CAMERA_OBSERVATION_GROUP: {"front_view": images}}


@pytest.mark.parametrize("action_dim", [2, 5])
def test_policy_uses_adapter_actions_for_each_environment(backend, action_dim):
    client = backend.client
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        state = json.loads(kwargs["messages"][-1]["content"][1]["text"])
        return response(json.dumps({"target": (np.asarray(state["joint_positions"]) + 0.5).tolist()}))

    client.chat.completions.create.side_effect = create
    config = VLMPolicyCfg(
        system_prompt="Use this robot's joint units.",
        image_max_width_or_height=8,
    )
    policy = VLMStepPolicy(
        config,
        inference_backend=backend,
        observation_adapter=ObservationAdapter(),
        action_adapter=ActionAdapter(scale=2, action_dim=action_dim),
    )
    env, observation = inputs(action_dim)
    try:
        policy.set_task_description("Move each joint")
        torch.testing.assert_close(policy.get_action(env, observation), torch.ones((2, action_dim)))
        assert not policy.is_remote
        assert len(calls) == 2
        for env_id, call in enumerate(calls):
            assert call["messages"][0] == {"role": "system", "content": config.system_prompt}
            assert call["response_format"]["json_schema"]["schema"] == build_strict_schema(ActionAdapter.response_model)
            content = call["messages"][1]["content"]
            assert json.loads(content[0]["text"]) == {"task": "Move each joint"}
            assert json.loads(content[1]["text"]) == {"joint_positions": observation["position"][env_id].tolist()}
            assert content[2]["text"] == "camera=front_view"
            image = Image.open(io.BytesIO(base64.b64decode(content[3]["image_url"]["url"].split(",", 1)[1])))
            assert image.size == (8, 4)
            assert image.getpixel((0, 0))[env_id * 2] > 240
        observation["position"].add_(1)
        policy.reset(torch.tensor([0]))
        policy.get_action(env, observation)
        assert len(calls) == 4
        assert all(len(call["messages"]) == 2 for call in calls)
    finally:
        policy.close()
    client.close.assert_not_called()  # The backend belongs to the caller, not the policy.


@pytest.mark.parametrize(
    "bad_raw,needs_correction",
    [
        ("not JSON", False),
        ('{"target": [1]}', True),
        ('{"target": [1, NaN]}', True),
        (None, False),
        ('{"wrong_field": [1, 2]}', True),
        ('{"target": "not a list"}', True),
        ('{"target": [1, 2], "extra": 1}', True),
    ],
)
def test_invalid_responses_are_corrected_before_actions(backend, bad_raw, needs_correction):
    client = backend.client
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 1:
            return response(bad_raw)
        return response('{"target": [1, 2]}')

    client.chat.completions.create.side_effect = create
    policy = make_policy(backend, max_attempts=2)
    env, observation = inputs()
    try:
        action = policy.get_action(env, observation)
        torch.testing.assert_close(action, torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
        assert len(calls) == 3
        if needs_correction:
            assert "Invalid actions" in calls[1]["messages"][-1]["content"]
        else:
            assert calls[1]["messages"] == calls[0]["messages"]  # Backend retries preserve image inputs.
        assert len(calls[2]["messages"]) == 2  # Correction from env 0 must not leak into env 1.
    finally:
        policy.close()


def test_invalid_commands_fail_after_bounded_attempts(backend):
    client = backend.client
    client.chat.completions.create.return_value = response('{"target": [1]}')
    policy = make_policy(backend, max_attempts=2)
    policy.action_adapter.command_to_action = Mock()
    try:
        with pytest.raises(InvalidAgentResponseError, match="Wrong command dimensions"):
            policy.get_action(*inputs())
        assert client.chat.completions.create.call_count == 2
        policy.action_adapter.command_to_action.assert_not_called()
    finally:
        policy.close()


def test_environment_validation_precedes_inference(backend):
    client = backend.client
    policy = make_policy(backend)
    env, observation = inputs()
    env.unwrapped.action_dim = 0
    try:
        with pytest.raises(AssertionError, match="Robot action space"):
            policy.get_action(env, observation)
        client.chat.completions.create.assert_not_called()
    finally:
        policy.close()


@pytest.mark.parametrize("prompt", ["", " \n\t", None])
def test_missing_prompt_is_rejected(prompt):
    with pytest.raises(AssertionError, match="policy.system_prompt"):
        VLMPolicyCfg(system_prompt=prompt)


def test_policy_configs_require_prompt_and_goal_budget():
    with pytest.raises(TypeError, match="system_prompt"):
        VLMPolicyCfg()
    with pytest.raises(TypeError, match="max_steps_per_goal"):
        VLMGoalPolicyCfg(system_prompt="Control joints")


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True, None])
def test_goal_config_rejects_invalid_step_budget(max_steps):
    with pytest.raises(AssertionError, match="max_steps_per_goal"):
        VLMGoalPolicyCfg(system_prompt="Control joints", max_steps_per_goal=max_steps)


class JointChunkAdapter(ChunkActionAdapter):

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def __init__(self, chunk_size: int, action_dim=2):
        super().__init__(chunk_size)
        self.joint_names = tuple(f"joint_{i}" for i in range(action_dim))
        self.response_model = create_model(
            "JointChunkResponse",
            __config__=ConfigDict(extra="forbid", strict=True),
            actions=(list[list[float]], Field(min_length=chunk_size, max_length=chunk_size)),
        )

    def decode_command(self, response, arena_world, env_id):
        positions = arena_world.get_joint_positions("robot", self.joint_names)[env_id].cpu().numpy()
        if any(len(row) != len(positions) for row in response.actions):
            raise InvalidAgentResponseError("Wrong command dimensions")
        return np.asarray(response.actions, dtype=np.float32)


class JointGoalAdapter(GoalActionAdapter):
    response_model = JointTargetResponse

    def __init__(self, action_dim=2, *, motion_generator=None):
        if motion_generator is None:
            motion_generator = JointMotionGenerator(action_dim=action_dim)
        super().__init__(motion_generator=motion_generator)
        self.joint_names = tuple(f"joint_{i}" for i in range(action_dim))

    def is_robot_action_space_supported(self, env):
        return env.unwrapped.action_dim > 0

    def decode_command(self, response, arena_world, env_id):
        positions = arena_world.get_joint_positions("robot", self.joint_names)[env_id].cpu().numpy()
        target = np.asarray(response.target, dtype=np.float32)
        if target.shape != (len(positions),):
            raise InvalidAgentResponseError("Wrong goal dimensions")
        return target


class JointMotionGenerator(MotionGenerator):
    def __init__(self, action_dim=2, max_step=0.25):
        self.joint_names = tuple(f"joint_{i}" for i in range(action_dim))
        self.max_step = max_step

    def goal_reached(self, goal, env, env_id):
        arena_world = env.unwrapped.arena_world
        positions = arena_world.get_joint_positions("robot", self.joint_names)[env_id].cpu().numpy()
        return np.allclose(goal, positions)

    def compute_command(self, goal, env, env_id):
        arena_world = env.unwrapped.arena_world
        positions = arena_world.get_joint_positions("robot", self.joint_names)[env_id].cpu().numpy()
        return np.clip(goal - positions, -self.max_step, self.max_step)


def make_multi_step_policy(backend, adapter, *, max_steps_per_goal=2):
    if isinstance(adapter, ChunkActionAdapter):
        policy_type = VLMChunkPolicy
        config = VLMPolicyCfg(system_prompt="Control these joints.", max_attempts=1)
    else:
        policy_type = VLMGoalPolicy
        config = VLMGoalPolicyCfg(
            system_prompt="Control these joints.", max_attempts=1, max_steps_per_goal=max_steps_per_goal
        )
    policy = policy_type(
        config,
        inference_backend=backend,
        observation_adapter=ObservationAdapter(),
        action_adapter=adapter,
    )
    policy.set_task_description("Move")
    policy.reset()  # Reset before the first observation must be harmless.
    return policy


@pytest.mark.parametrize("action_dim", [2, 5])
def test_chunks_replay_in_order_and_reset_independently(backend, action_dim):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        start = 3 * (len(calls) - 1) + 1
        return response(json.dumps({"actions": [[value] * action_dim for value in range(start, start + 3)]}))

    backend.client.chat.completions.create.side_effect = create
    policy = make_multi_step_policy(backend, JointChunkAdapter(chunk_size=3, action_dim=action_dim))
    env, observation = inputs(action_dim)
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[1.0] * action_dim, [4.0] * action_dim])
    )
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[2.0] * action_dim, [5.0] * action_dim])
    )
    assert len(calls) == 2
    policy.reset(torch.tensor([0]))
    observation["position"].add_(10)
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[7.0] * action_dim, [6.0] * action_dim])
    )
    assert len(calls) == 3
    current_state = json.loads(calls[2]["messages"][1]["content"][1]["text"])
    assert current_state["joint_positions"] == observation["position"][0].tolist()
    torch.testing.assert_close(
        policy.get_action(env, observation), torch.tensor([[8.0] * action_dim, [10.0] * action_dim])
    )
    assert len(calls) == 4
    policy.reset()
    policy.get_action(env, observation)
    assert len(calls) == 6


@pytest.mark.parametrize(
    "commands",
    [
        [],
        [[1, 2]],
        [1, 2, 3],
        [[1, 2], [3, float("nan")], [4, 5]],
        [[1, 2], [3], [4, 5]],
    ],
)
def test_whole_chunk_is_validated_before_execution(backend, commands):
    backend.client.chat.completions.create.return_value = response(json.dumps({"actions": commands}))
    adapter = JointChunkAdapter(chunk_size=3)
    adapter.command_to_action = Mock()
    policy = make_multi_step_policy(backend, adapter)
    with pytest.raises(InvalidAgentResponseError):
        policy.get_action(*inputs())
    adapter.command_to_action.assert_not_called()
    assert backend.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("action_dim", [2, 5])
def test_goals_use_measured_feedback_step_limit_and_partial_reset(backend, action_dim):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        state = json.loads(kwargs["messages"][1]["content"][1]["text"])
        return response(json.dumps({"target": (np.asarray(state["joint_positions"]) + 1).tolist()}))

    backend.client.chat.completions.create.side_effect = create
    policy = make_multi_step_policy(backend, JointGoalAdapter(action_dim=action_dim))
    env, observation = inputs(action_dim)
    torch.testing.assert_close(policy.get_action(env, observation), torch.full((2, action_dim), 0.25))
    assert len(calls) == 2
    # Env 0 reaches its goal; env 1 is close enough to require a smaller command.
    observation["position"][0].add_(1)
    observation["position"][1].add_(0.9)
    expected = torch.stack((torch.full((action_dim,), 0.25), torch.full((action_dim,), 0.1)))
    torch.testing.assert_close(policy.get_action(env, observation), expected)
    assert len(calls) == 3
    # Env 1 exhausts its budget while env 0 continues its second goal.
    torch.testing.assert_close(policy.get_action(env, observation), torch.full((2, action_dim), 0.25))
    assert len(calls) == 4
    policy.reset(torch.tensor([0]))
    policy.get_action(env, observation)
    assert len(calls) == 5
    policy.reset()
    policy.get_action(env, observation)
    assert len(calls) == 7


@pytest.mark.parametrize("mode", ["step", "goal"])
def test_actions_use_live_world_without_joint_positions_in_prompt(backend, mode):
    backend.client.chat.completions.create.return_value = response('{"target": [1, 2]}')
    if mode == "step":
        policy = make_policy(backend)
    else:
        policy = make_multi_step_policy(backend, JointGoalAdapter(), max_steps_per_goal=5)
    policy.observation_adapter = ObservationAdapter(include_joint_positions=False)
    policy.observation_adapter.extract_vlm_input_from_observation = Mock(
        wraps=policy.observation_adapter.extract_vlm_input_from_observation
    )
    env, observation = inputs()
    world_positions = observation["position"]
    observation["position"] = world_positions.clone()  # Keep prompt observations stale.
    first = policy.get_action(env, observation)
    expected_step = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    torch.testing.assert_close(first, expected_step if mode == "step" else expected_step * 0.25)
    world_positions[0].add_(0.9)
    world_positions[1].sub_(0.9)
    torch.testing.assert_close(policy.get_action(env, observation), expected_step * 0.1)
    expected_calls = 4 if mode == "step" else 2
    assert backend.client.chat.completions.create.call_count == expected_calls
    assert policy.observation_adapter.extract_vlm_input_from_observation.call_count == expected_calls
    content = backend.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert len(content) == 3  # Task, camera label, and image; no joint positions.
    assert content[1] == {"type": "text", "text": "camera=front_view"}
    assert content[2]["type"] == "image_url"
    env.unwrapped.arena_world.get_joint_positions.assert_called_with("robot", ("joint_0", "joint_1"))


@pytest.mark.parametrize("max_step", [0.1, 0.25])
def test_goal_decoder_can_use_different_motion_generators(backend, max_step):
    backend.client.chat.completions.create.return_value = response('{"target": [1, 2]}')
    policy = make_multi_step_policy(backend, JointGoalAdapter(motion_generator=JointMotionGenerator(max_step=max_step)))
    expected = torch.tensor([[max_step, max_step], [-max_step, -max_step]])
    torch.testing.assert_close(policy.get_action(*inputs()), expected)
    assert backend.client.chat.completions.create.call_count == 2


def test_invalid_goals_are_rejected_before_motion_generation(backend):
    backend.client.chat.completions.create.return_value = response('{"target": [1]}')
    generator = JointMotionGenerator()
    generator.compute_command = Mock()
    policy = make_multi_step_policy(backend, JointGoalAdapter(motion_generator=generator))
    with pytest.raises(InvalidAgentResponseError, match="Wrong goal dimensions"):
        policy.get_action(*inputs())
    generator.compute_command.assert_not_called()
    assert backend.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("command", [np.array([np.inf, 0]), np.zeros(1), np.zeros((1, 2)), np.empty(0)])
def test_motion_generator_output_is_validated_before_conversion(backend, command):
    backend.client.chat.completions.create.return_value = response('{"target": [1, 2]}')
    generator = JointMotionGenerator()
    generator.compute_command = Mock(return_value=command)
    adapter = JointGoalAdapter(motion_generator=generator)
    adapter.command_to_action = Mock()
    policy = make_multi_step_policy(backend, adapter)
    with pytest.raises(AssertionError, match="finite goal command"):
        policy.get_action(*inputs())
    adapter.command_to_action.assert_not_called()


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"wrong_field": [1, 2]}, "validation errors"),
        ({"target": [1]}, "Wrong command dimensions"),
        ({"target": [1, float("nan")]}, "Expected finite commands"),
    ],
)
def test_response_rejection_sends_correction_message(backend, payload, message):
    backend.client.chat.completions.create.return_value = response(json.dumps(payload))
    policy = make_policy(backend, max_attempts=2)
    with pytest.raises(InvalidAgentResponseError, match=message) as caught:
        policy.get_action(*inputs())
    correction = backend.client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert correction == f"Invalid actions: {caught.value}. Return a corrected prediction."
    assert backend.client.chat.completions.create.call_count == 2


def test_invalid_model_actions_are_rejected_with_python_optimization():
    # This child imports no simulation runtime; -O must not disable model-output rejection.
    script = textwrap.dedent("""\
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock
        from isaaclab_arena_vlm_agent.tests.test_vlm_agent_policy import (
            ActionAdapter, ObservationAdapter,
        )
        from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError
        from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMStepPolicy, VLMPolicyCfg

        for command in (np.ones((1, 2)), np.empty(0), np.array([1, np.nan])):
            backend = SimpleNamespace(infer=Mock(return_value={"target": [1, 2]}))
            adapter = ActionAdapter()
            adapter.decode_command = Mock(return_value=command)
            policy = VLMStepPolicy(
                VLMPolicyCfg(system_prompt="Control joints", max_attempts=2),
                inference_backend=backend, observation_adapter=ObservationAdapter(), action_adapter=adapter,
            )
            try:
                policy._infer([], SimpleNamespace(), 0)
            except InvalidAgentResponseError as error:
                if "Expected finite commands" not in str(error) or backend.infer.call_count != 2:
                    raise RuntimeError("Incorrect rejection message or retry count")
            else:
                raise RuntimeError("Invalid model actions were accepted with assertions disabled")
        print("optimized rejection checks passed")
    """)
    result = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "optimized rejection checks passed" in result.stdout


@pytest.mark.parametrize("error_type", [KeyError, TypeError, AssertionError, ValueError])
@pytest.mark.parametrize("method", ["decode_command", "is_command_valid"])
def test_adapter_implementation_errors_are_not_retried(backend, error_type, method):
    backend.client.chat.completions.create.return_value = response('{"target": [1, 2]}')
    policy = make_policy(backend, max_attempts=3)
    error = error_type("adapter implementation bug")
    setattr(policy.action_adapter, method, Mock(side_effect=error))
    with pytest.raises(error_type) as caught:
        policy.get_action(*inputs())
    assert caught.value is error
    assert backend.client.chat.completions.create.call_count == 1


def test_empty_choices_retry_without_correction_message(backend):
    backend.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[]),
        response('{"target": [1, 2]}'),
        response('{"target": [1, 2]}'),
    ]
    policy = make_policy(backend)
    policy.get_action(*inputs())
    calls = backend.client.chat.completions.create.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["messages"] == calls[1].kwargs["messages"]


def test_missing_camera_keys_fail_before_inference(backend):
    policy = make_policy(backend)
    policy.observation_adapter.camera_keys = ()
    with pytest.raises(AssertionError, match="camera observation key"):
        policy.get_action(*inputs())
    backend.client.chat.completions.create.assert_not_called()
