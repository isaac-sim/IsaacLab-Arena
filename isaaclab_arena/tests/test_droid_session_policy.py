# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check session policy behavior through its public action and lifecycle methods."""

import json
import numpy as np
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, RelativeJointPositionAction

from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
from isaaclab_arena.policy.droid_session_policy import DroidSessionPolicy, DroidSessionPolicyCfg
from isaaclab_arena_examples.robot_tool_control.session_client import inspect_session, submit_actions
from isaaclab_arena_openpi.policy.droid_adapter import Pi0DroidAdapter


def _environment(arm_action_type=JointPositionAction):
    # Use the real action types without constructing an articulation or a simulator.
    arm_action = object.__new__(arm_action_type)
    arm_action._num_joints = 7
    arm_action.cfg = SimpleNamespace(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
        scale=1.0,
        offset=0.0,
        clip=None,
    )
    gripper_action = object.__new__(BinaryJointPositionZeroToOneAction)
    gripper_action.cfg = SimpleNamespace(asset_name="robot", clip=None)
    action_terms = {"arm_action": arm_action, "gripper_action": gripper_action}
    joint_names = [f"panda_joint{index}" for index in range(1, 8)]
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_names=joint_names,
            joint_pos_limits=wp.from_numpy(np.tile([-3.0, 3.0], (1, 7, 1)).astype(np.float32)),
        ),
        find_joints=lambda joint_expressions, preserve_order: (list(range(7)), joint_names),
    )
    environment = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=1 / 15,
        scene={"robot": robot},
        action_manager=SimpleNamespace(active_terms=list(action_terms), get_term=action_terms.__getitem__),
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        episode_index=0,
    )
    environment.get_episode_index = lambda env_id: environment.episode_index
    environment.unwrapped = environment
    return environment


def _observation():
    return {
        "camera_obs": {
            "external_camera_rgb": torch.arange(24 * 32 * 3, dtype=torch.int32).to(torch.uint8).reshape(1, 24, 32, 3),
            "wrist_camera_rgb": torch.full((1, 24, 32, 3), 127, dtype=torch.uint8),
        },
        "policy": {"joint_pos": torch.zeros(1, 7), "gripper_pos": torch.zeros(1, 1)},
    }


@pytest.fixture
def configured_policy(tmp_path):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Use only the current episode's observations.\n")
    policy = DroidSessionPolicy(DroidSessionPolicyCfg(str(prompt_path), 2, 4.0))
    policy.set_output_directory(tmp_path / "policy")
    session_directory = next((tmp_path / "policy").iterdir())
    policy.reset()
    policy.set_task_description("Put the cube in the bowl.")
    yield policy, session_directory
    policy.close()


def _respond_to_next_action(policy, session_directory, environment, actions):
    observation = _observation()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending_action = executor.submit(policy.get_action, environment, observation)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if pending_action.done():
                pending_action.result()
            inspection = inspect_session(session_directory)
            if "request" in inspection:
                request_path = Path(inspection["request_path"])
                submit_actions(request_path, actions)
                return pending_action.result(timeout=3.0), inspection["request"]
            time.sleep(0.01)
    raise AssertionError("No action request was published")


def test_policy_returns_chunk_rows_and_refetches_with_openpi_inputs(configured_policy):
    from PIL import Image

    policy, session_directory = configured_policy
    environment = _environment()
    chunk = [[0.1] * 7 + [0], [0.2] * 7 + [1]]
    first_action, request = _respond_to_next_action(policy, session_directory, environment, chunk)
    second_action = policy.get_action(environment, _observation())
    torch.testing.assert_close(first_action, torch.tensor([chunk[0]]))
    torch.testing.assert_close(second_action, torch.tensor([chunk[1]]))
    assert len(list(session_directory.rglob("request.json"))) == 1
    assert inspect_session(session_directory)["session"]["active_request"] is None
    adapter = Pi0DroidAdapter()
    expected = adapter.pack_request(adapter.extract(_observation(), 0), policy.task_description)
    assert request["observation"]["joint_position"] == expected["observation/joint_position"].tolist()
    assert request["observation"]["gripper_position"] == expected["observation/gripper_position"].tolist()
    assert request["task_instruction"] == expected["prompt"]
    np.testing.assert_array_equal(
        np.asarray(Image.open(session_directory / request["images"]["exterior_image"])),
        expected["observation/exterior_image_1_left"],
    )
    environment.episode_length_buf += 2
    third_action, next_request = _respond_to_next_action(policy, session_directory, environment, chunk[::-1])
    torch.testing.assert_close(third_action, torch.tensor([chunk[1]]))
    assert next_request["request_id"] != request["request_id"]
    assert next_request["observation"]["simulation_step"] == 2


def test_reset_discards_chunk_without_starting_an_extra_episode(configured_policy):
    policy, session_directory = configured_policy
    environment = _environment()
    assert not list(session_directory.rglob("episode.json"))
    _respond_to_next_action(policy, session_directory, environment, [[0.1] * 7 + [0], [0.2] * 7 + [0]])
    policy.reset(torch.tensor([0]))
    assert inspect_session(session_directory)["session"]["last_event"]["event"] == "episode_ended"
    assert len(list(session_directory.rglob("episode.json"))) == 1
    environment.episode_index = 1
    next_action, request = _respond_to_next_action(
        policy,
        session_directory,
        environment,
        [[0.3] * 7 + [1], [0.4] * 7 + [1]],
    )
    assert request["episode_index"] == 1
    torch.testing.assert_close(next_action, torch.tensor([[0.3] * 7 + [1]]))
    policy.reset(torch.tensor([0]))
    policy.close()
    policy.close()
    assert len(list(session_directory.rglob("episode.json"))) == 2
    assert inspect_session(session_directory)["session"]["status"] == "closed"
    assert inspect_session(session_directory)["session"]["last_event"]["event"] == "episode_ended"


@pytest.mark.parametrize(
    "actions, error",
    [
        ([[0] * 8], "Expected actions of shape"),
        ([[0] * 7 + [0.5]] * 2, "Gripper actions"),
        ([[4] + [0] * 7] * 2, "position limits"),
        ([["0"] * 8] * 2, "real numbers"),
        ([[True] + [0] * 7] * 2, "real numbers"),
    ],
)
def test_invalid_chunk_records_error_before_returning_actions(configured_policy, actions, error):
    policy, session_directory = configured_policy
    with pytest.raises(AssertionError, match=error):
        _respond_to_next_action(policy, session_directory, _environment(), actions)
    error_path = next(session_directory.rglob("error.json"))
    assert error in json.loads(error_path.read_text())["message"]
    policy.close()
    assert inspect_session(session_directory)["session"]["last_event"]["event"] == "episode_stopped"


@pytest.mark.parametrize("incompatibility", ["relative_actions", "parallel_envs", "joint_offset"])
def test_incompatible_environment_publishes_no_request(configured_policy, incompatibility):
    policy, session_directory = configured_policy
    environment = _environment(
        RelativeJointPositionAction if incompatibility == "relative_actions" else JointPositionAction
    )
    if incompatibility == "parallel_envs":
        environment.num_envs = 2
    if incompatibility == "joint_offset":
        environment.action_manager.get_term("arm_action").cfg.use_default_offset = True
    with pytest.raises(AssertionError):
        policy.get_action(environment, _observation())
    assert not list(session_directory.rglob("request.json"))
    assert (session_directory / "error.json").is_file()
