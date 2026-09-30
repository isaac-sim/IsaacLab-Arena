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


def _bimanual_env():
    joint_names = [*(f"joint{i}" for i in range(1, 7)), "left_finger"]
    left_joints = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0375]])
    right_joints = torch.tensor([[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.0]])

    def robot(joints):
        return SimpleNamespace(joint_names=joint_names, data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joints)))

    terms = {
        "left_gripper_action": SimpleNamespace(cfg=SimpleNamespace(offset=0.0375, scale=-0.0375)),
        "right_gripper_action": SimpleNamespace(cfg=SimpleNamespace(offset=0.0375, scale=-0.0375)),
    }
    return SimpleNamespace(
        scene={"left_robot": robot(left_joints), "right_robot": robot(right_joints)},
        action_manager=SimpleNamespace(get_term=terms.__getitem__),
        num_envs=1,
        device="cpu",
        step_dt=0.25,
        cap_episode_finished=False,
    )


def test_usbc_cap_disconnect_termination():
    from isaaclab_arena_environments.isaac_cap.cap_policy import cap_episode_finished as shared_predicate
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.task import cap_episode_finished

    assert cap_episode_finished is shared_predicate
    env = SimpleNamespace(num_envs=2, device="cpu")
    assert cap_episode_finished(env).tolist() == [False, False]
    env.cap_episode_finished = True
    assert cap_episode_finished(env).tolist() == [True, True]


def test_usbc_gap_policy_rejects_unknown_robot_profile():
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    with pytest.raises(ValueError, match="Unsupported CAP robot profile"):
        CapPolicy(CapPolicyCfg(robot_profile="unknown"))


def test_usbc_gap_policy_contract():
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    env = _bimanual_env()
    policy = CapPolicy(
        CapPolicyCfg(
            robot_profile="yam_bimanual",
            camera_mapping={
                "top_camera": "topdown",
                "right_wrist_camera": "wrist",
                "left_wrist_camera": "wrist_support",
            },
        )
    )
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


def test_usbc_disconnect_holds_grippers_until_episode_finishes():
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.task import cap_episode_finished

    env = _bimanual_env()
    policy = CapPolicy(CapPolicyCfg(robot_profile="yam_bimanual", settle_s=0.5))
    policy._socket = SimpleNamespace(close=lambda: None)
    policy._observation_frame = lambda _env, _action: {}
    calls = 0

    def exchange(_frame):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "left": {"arm_valid": False, "gripper_valid": True, "gripper": 0.25},
                "right": {"arm_valid": False, "gripper_valid": True, "gripper": 0.75},
            }
        raise EOFError("CAP graph closed the connection")

    policy._exchange = exchange
    wrapped_env = SimpleNamespace(unwrapped=env)
    commanded = policy.get_action(wrapped_env, {})[0]
    assert commanded[[6, 13]].tolist() == pytest.approx([0.75, 0.25])

    disconnected = policy.get_action(wrapped_env, {})[0]
    assert calls == 2
    assert disconnected[[6, 13]].tolist() == pytest.approx([0.75, 0.25])
    assert not cap_episode_finished(env).item()

    settled = policy.get_action(wrapped_env, {})[0]
    assert calls == 2
    assert settled[[6, 13]].tolist() == pytest.approx([0.75, 0.25])
    assert cap_episode_finished(env).item()

    policy.reset()
    assert not cap_episode_finished(env).item()
    assert policy._last_gripper is None
