# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the Arena-owned bimanual YAM CAP policy adapter."""

import numpy as np
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena_environments.isaac_cap.cap_policy import CapYamPolicy, CapYamPolicyCfg

pytestmark = pytest.mark.isaac_cap


def _robot(joints: list[float]):
    return SimpleNamespace(
        joint_names=[*(f"joint{index}" for index in range(1, 7)), "left_finger"],
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([joints], dtype=torch.float32))),
    )


def _environment():
    return SimpleNamespace(
        scene={
            "left_robot": _robot([1, 2, 3, 4, 5, 6, 0.037524]),
            "right_robot": _robot([7, 8, 9, 10, 11, 12, 0.0]),
        },
        action_manager=SimpleNamespace(total_action_dim=14),
        device="cpu",
    )


def _policy() -> CapYamPolicy:
    return CapYamPolicy(CapYamPolicyCfg())


def test_yam_hold_action_matches_bimanual_action_layout():
    action = _policy()._hold_action(_environment())

    assert action.shape == (14,)
    assert torch.equal(action[:6], torch.arange(1, 7, dtype=torch.float32))
    assert action[6] == 0.0
    assert torch.equal(action[7:13], torch.arange(7, 13, dtype=torch.float32))
    assert action[13] == 1.0


def test_yam_observation_frame_includes_both_arms_and_camera_aliases():
    policy = _policy()
    policy._camera = lambda _env, name: {"camera": name}

    frame = policy._observation_frame(_environment())

    assert frame["left"]["joint_pos"] == [1, 2, 3, 4, 5, 6, 1.0]
    assert frame["right"]["joint_pos"] == [7, 8, 9, 10, 11, 12, 0.0]
    assert frame["overhead"]["camera"] == "top_camera"
    assert frame["eye_in_hand_left"]["camera"] == "left_wrist_camera"
    assert frame["eye_in_hand_right"]["camera"] == "right_wrist_camera"
    assert frame["side"]["camera"] == "side_camera"


def test_yam_reply_updates_only_valid_channels():
    policy = _policy()
    action = policy._hold_action(_environment())
    reply = {
        "left": {
            "arm_valid": True,
            "joint_pos": np.arange(6, dtype=np.float32),
            "gripper_valid": True,
            "gripper": 0.25,
        },
        "right": {
            "arm_valid": False,
            "gripper_valid": False,
        },
    }

    policy._apply_reply(reply, action)

    assert torch.equal(action[:6], torch.arange(6, dtype=torch.float32))
    assert action[6] == 0.75
    assert torch.equal(action[7:13], torch.arange(7, 13, dtype=torch.float32))
    assert action[13] == 1.0
    assert torch.equal(policy._last_grippers, torch.tensor([0.75, 1.0]))
