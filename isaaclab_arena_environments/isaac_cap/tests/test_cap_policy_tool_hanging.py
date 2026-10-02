# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check tool hanging's configuration of the shared bimanual CAP client."""

import torch
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.isaac_cap


def test_tool_hanging_uses_shared_client_with_its_camera_rig_and_gripper_range():
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    config_path = Path(__file__).parents[1] / "tool_hanging/experiment_configs/tool_hanging_cap_remote_experiment.yaml"
    policy_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))["shared"]["policy"]
    assert policy_values.pop("type") == "cap_remote"
    policy = CapPolicy(CapPolicyCfg(**policy_values))
    assert policy.config.robot_profile == "yam_bimanual"

    names = [*(f"joint{index}" for index in range(1, 7)), "left_finger"]

    def robot(joints):
        return SimpleNamespace(
            joint_names=names,
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([joints], dtype=torch.float32))),
        )

    # The wide-gripper variant changes the action term's offset and scale.
    terms = {
        f"{side}_gripper_action": SimpleNamespace(cfg=SimpleNamespace(offset=0.0475, scale=-0.0475))
        for side in ("left", "right")
    }
    env = SimpleNamespace(
        scene={
            "left_robot": robot([1, 2, 3, 4, 5, 6, 0.0475]),
            "right_robot": robot([7, 8, 9, 10, 11, 12, 0.0]),
        },
        action_manager=SimpleNamespace(get_term=terms.__getitem__),
    )
    hold = policy._hold_action(env)
    assert hold.tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 1])

    policy._camera = lambda _env, name: {"camera": name}
    frame = policy._observation_frame(env, hold)
    assert frame["left"]["joint_pos"] == pytest.approx([1, 2, 3, 4, 5, 6, 1])
    assert frame["right"]["joint_pos"] == pytest.approx([7, 8, 9, 10, 11, 12, 0])
    assert frame["overhead"] == {"camera": "top_camera"}
    assert frame["eye_in_hand_left"] == {"camera": "left_wrist_camera"}
    assert frame["eye_in_hand_right"] == {"camera": "right_wrist_camera"}
    assert frame["side"] == {"camera": "side_camera"}
