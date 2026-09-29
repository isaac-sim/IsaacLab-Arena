# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check the USB-C GaP client's bimanual observation and action contract."""

import numpy as np
import torch
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.isaac_cap


def test_usbc_cap_disconnect_termination():
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.task import cap_episode_finished

    env = SimpleNamespace(num_envs=2, device="cpu")
    assert cap_episode_finished(env).tolist() == [False, False]
    env.cap_episode_finished = True
    assert cap_episode_finished(env).tolist() == [True, True]


def test_usbc_gap_policy_contract():
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    joint_names = [*(f"joint{i}" for i in range(1, 7)), "left_finger"]
    left_joints = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0375]])
    right_joints = torch.tensor([[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.0]])

    def robot(joints):
        return SimpleNamespace(joint_names=joint_names, data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joints)))

    terms = {
        "left_gripper_action": SimpleNamespace(cfg=SimpleNamespace(offset=0.0375, scale=-0.0375)),
        "right_gripper_action": SimpleNamespace(cfg=SimpleNamespace(offset=0.0375, scale=-0.0375)),
    }
    env = SimpleNamespace(
        scene={"left_robot": robot(left_joints), "right_robot": robot(right_joints)},
        action_manager=SimpleNamespace(get_term=terms.__getitem__),
    )
    policy = object.__new__(CapPolicy)
    policy.config = CapPolicyCfg(
        robot_profile="yam_bimanual",
        camera_mapping={
            "top_camera": "topdown",
            "right_wrist_camera": "wrist",
            "left_wrist_camera": "wrist_support",
        },
    )
    policy._last_gripper = None
    hold = policy._hold_action(env)
    assert hold.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.0])

    policy._camera = lambda _env, name: {"source": name}
    frame = policy._observation_frame(env, hold)
    assert frame["left"]["joint_pos"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    assert frame["right"]["joint_pos"] == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.0])
    assert frame["topdown"] == {"source": "top_camera"}
    assert frame["wrist"] == {"source": "right_wrist_camera"}
    assert frame["wrist_support"] == {"source": "left_wrist_camera"}

    reply = {
        "left": {
            "joint_pos": [1, 2, 3, 4, 5, 6],
            "gripper": 0.25,
            "arm_valid": True,
            "gripper_valid": False,
        },
        "right": {
            "joint_pos": [6, 5, 4, 3, 2, 1],
            "gripper": 0.75,
            "arm_valid": False,
            "gripper_valid": True,
        },
    }
    policy._apply_reply(reply, hold)
    assert hold[:6].tolist() == pytest.approx([1, 2, 3, 4, 5, 6])
    assert hold[6].item() == pytest.approx(0.0)
    assert hold[7:13].tolist() == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    assert hold[13].item() == pytest.approx(0.25)
    assert np.asarray(policy._last_gripper).tolist() == pytest.approx([0.0, 0.25])
