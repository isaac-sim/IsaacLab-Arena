# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import numpy as np
from collections import deque
from types import SimpleNamespace

import pytest
from PIL import Image

from isaaclab_arena_vlm_agent_policy.embodiment_adapter.droid_eef_action_adapter import (
    DroidGoalActionAdapter,
    DroidProprioceptionAdapter,
)
from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_goal_policy import (
    AgentCommandedGoalPolicy,
    AgentCommandedGoalPolicyCfg,
)
from isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy import VLMAgentPolicyCfg, encode_image


def test_image_encoding_preserves_aspect_ratio_and_color():
    pixels = np.zeros((720, 1280, 4), dtype=np.uint8)
    pixels[..., 0] = 255
    url = encode_image(pixels)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == (384, 216)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0))[0] > 250


@pytest.mark.parametrize("rotate_local_x", [False, True])
def test_droid_quaternion_boundary_in_rotated_root_frame(rotate_local_x):
    """Check known physical orientations through observation, model output, and IK input."""
    import torch

    from isaaclab.utils.math import quat_apply

    s = np.sqrt(0.5)
    # Root is translated and rotated +90 degrees around world Z.
    # A local +90-degree X rotation then has world XYZW quaternion [0.5]*4.
    world_quat = [0.5, 0.5, 0.5, 0.5] if rotate_local_x else [0, 0, s, s]
    expected_native_quat = [s, 0, 0, s] if rotate_local_x else [0, 0, 0, 1]
    root = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.tensor([[1.0, 2.0, 3.0]])),
        root_quat_w=SimpleNamespace(torch=torch.tensor([[0, 0, s, s]], dtype=torch.float32)),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene={"robot": SimpleNamespace(data=root)},
            num_envs=1,
            device="cpu",
            step_dt=1 / 15,
        )
    )
    observation = {
        "policy": {
            "eef_pos": torch.tensor([[1.0, 2.4, 3.3]]),
            "eef_quat": torch.tensor([world_quat], dtype=torch.float32),
            "joint_pos": torch.zeros((1, 7)),
            "gripper_pos": torch.ones((1, 1)),
        }
    }
    state = DroidProprioceptionAdapter().extract_proprioception(env, observation)[0]
    expected_pose = np.array([0.4, 0.0, 0.3, *expected_native_quat])
    np.testing.assert_allclose(state["eef_pose_root_xyz_xyzw"], expected_pose, atol=1e-6)
    adapter = DroidGoalActionAdapter()
    payload = {
        "command": "move_to",
        "position": expected_pose[:3].tolist(),
        "quaternion_xyzw": expected_pose[3:].tolist(),
        "gripper": 1,
        "steps": 12,
        "note": "Frame check",
    }
    command = adapter.decode_command(payload, state)
    saved_command = command.copy()
    action = adapter.command_to_action(env, command)
    np.testing.assert_allclose(action.numpy(), [0.4, 0, 0.3, *expected_native_quat, 1], atol=1e-6)
    # Check actual Isaac Lab rotation semantics, not just a round-trip permutation.
    rotated_y = quat_apply(action[None, 3:7], torch.tensor([[0.0, 1.0, 0.0]]))
    expected_y = [0, 0, 1] if rotate_local_x else [0, 1, 0]
    np.testing.assert_allclose(rotated_y.numpy()[0], expected_y, atol=1e-6)
    np.testing.assert_array_equal(command, saved_command)


def test_inference_corrects_invalid_goal_and_retains_only_accepted_response():
    requests = []
    pose = [0.4, 0, 0.3, 0, 0, 0, 1]
    payload = {
        "command": "wait",
        "position": None,
        "quaternion_xyzw": None,
        "gripper": None,
        "steps": 12,
        "note": "Hold pose",
    }

    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs)))
        response = dict(payload, steps=0) if len(requests) == 1 else payload
        message = SimpleNamespace(content=json.dumps(response), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    policy = AgentCommandedGoalPolicy.__new__(AgentCommandedGoalPolicy)
    policy.config = AgentCommandedGoalPolicyCfg(
        decision_history=2, system_prompt="Follow this experiment's goal contract."
    )
    policy.action_adapter = DroidGoalActionAdapter()
    policy.observation_adapter = DroidProprioceptionAdapter()
    policy.task_description = "Hold pose"
    policy._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    policy._histories = [deque([(0, {"wrist_camera_rgb": "image0"})], maxlen=1)]
    policy._steps = [0]
    policy._decisions = {}
    policy._trace = io.StringIO()
    state = {"eef_pose_root_xyz_xyzw": pose, "last_gripper_command": 0}
    result = policy._infer(0, state)
    assert result.shape == (10,)
    assert len(requests) == 2
    assert "Invalid actions" in requests[-1]["messages"][-1]["content"]
    assert all(r["messages"][0] == {"role": "system", "content": policy.config.system_prompt} for r in requests)
    records = [json.loads(line) for line in policy._trace.getvalue().splitlines()]
    assert "validation_error" in records[0] and "prediction" in records[1]
    assert len(policy._decisions[0]) == 1
    assert json.loads(policy._decisions[0][0][1]) == payload
    images = [part for part in requests[0]["messages"][1]["content"] if part["type"] == "image_url"]
    assert [part["image_url"]["url"] for part in images] == ["image0"]
    policy._infer(0, state)
    policy._infer(0, state)
    assert len(policy._decisions[0]) == 2


def test_reset_clears_only_selected_environment_history_and_goal():
    import torch

    policy = AgentCommandedGoalPolicy.__new__(AgentCommandedGoalPolicy)
    policy._histories = [deque([1]), deque([2])]
    policy._steps = [17, 18]
    policy._decisions = {0: deque(["first"]), 1: deque(["second"])}
    policy._goals = ["first goal", "second goal"]
    policy._grippers = [1, 1]
    policy.reset(torch.tensor([0]))
    assert not policy._histories[0] and list(policy._histories[1]) == [2]
    assert policy._steps == [0, 18]
    assert 0 not in policy._decisions and list(policy._decisions[1]) == ["second"]
    assert policy._goals == [None, "second goal"] and policy._grippers == [0, 1]


@pytest.mark.parametrize("prompt", ["", " \n\t", None])
def test_missing_or_blank_experiment_prompt_is_rejected(prompt):
    with pytest.raises(AssertionError, match="policy.system_prompt"):
        VLMAgentPolicyCfg(system_prompt=prompt)


@pytest.mark.parametrize("fault", [None, "relative", "link", "scale", "offset", "dimension"])
def test_absolute_eef_action_contract(fault):
    arm = SimpleNamespace(
        action_dim=7,
        cfg=SimpleNamespace(
            body_name="base_link",
            scale=1.0,
            body_offset=None,
            max_joint_velocity=0.8,
            controller=SimpleNamespace(use_relative_mode=False),
        ),
    )
    if fault == "relative":
        arm.cfg.controller.use_relative_mode = True
    elif fault == "link":
        arm.cfg.body_name = "panda_link0"
    elif fault == "scale":
        arm.cfg.scale = 0.5
    elif fault == "offset":
        arm.cfg.body_offset = (0, 0, 0.1)
    elif fault == "dimension":
        arm.action_dim = 6
    env = SimpleNamespace(unwrapped=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: arm)))
    adapter = DroidGoalActionAdapter()
    if fault is None:
        adapter.validate_environment(env)
    else:
        with pytest.raises(AssertionError):
            adapter.validate_environment(env)
