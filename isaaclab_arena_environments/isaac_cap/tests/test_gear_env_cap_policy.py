# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the gear environment's Arena-owned CAP policy adapter."""

import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _robot(joints: list[float]):
    import torch

    return SimpleNamespace(
        joint_names=[*(f"fr3_joint{index}" for index in range(1, 8)), "left_driver_joint"],
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([joints], dtype=torch.float32))),
    )


def _gear_environment():
    gripper_action = SimpleNamespace(cfg=SimpleNamespace(scale=0.8))
    return SimpleNamespace(
        scene={"robot": _robot([1, 2, 3, 4, 5, 6, 7, 0.4])},
        action_manager=SimpleNamespace(get_term=lambda _name: gripper_action),
        device="cpu",
    )


def _gear_policy(config_name: str):
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    config_path = Path(__file__).parents[1] / "gear_insertion_v2/experiment_configs" / config_name
    policy_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))["shared"]["policy"]
    assert policy_values.pop("type") == "cap_remote"
    return CapPolicy(CapPolicyCfg(**policy_values))


def _test_gear_env_cap_policy(_simulation_app) -> bool:
    import numpy as np
    import torch

    config_names = (
        "gear_easy_cap_remote_experiment.yaml",
        "gear_easy_pair_cap_remote_experiment.yaml",
        "gear_medium_train_cap_remote_experiment.yaml",
    )
    policies = [_gear_policy(config_name) for config_name in config_names]
    assert all(policy.config.camera_mapping == {"top_camera": "overhead"} for policy in policies)
    policy = policies[0]
    policy._camera = lambda _env, name: {"camera": name}
    policy._tip_reach = lambda _robot: 0.157
    environment = _gear_environment()
    action = policy._hold_action(environment)

    assert action.shape == (8,)
    assert torch.equal(action[:7], torch.arange(1, 8, dtype=torch.float32))
    assert float(action[7]) == pytest.approx(0.5)

    frame = policy._observation_frame(environment, action)

    assert frame["left"]["joint_pos"] == [1, 2, 3, 4, 5, 6, 7, 0.5]
    assert frame["overhead"]["camera"] == "top_camera"
    assert "eye_in_hand" not in frame
    assert "agentview" not in frame
    assert frame["_isaac_cap"]["gripper"]["tip_reach_m"] == pytest.approx(0.157)

    policy._apply_reply(
        {
            "left": {"joint_pos": np.arange(7, dtype=np.float32)},
            "arm_valid": True,
            "gripper_valid": False,
        },
        action,
    )

    assert torch.equal(action[:7], torch.arange(7, dtype=torch.float32))
    assert float(action[7]) == pytest.approx(0.5)

    policy._apply_reply(
        {
            "left": {"gripper": 0.25},
            "arm_valid": False,
            "gripper_valid": True,
        },
        action,
    )

    assert torch.equal(action[:7], torch.arange(7, dtype=torch.float32))
    assert float(action[7]) == pytest.approx(0.75)
    assert float(policy._last_gripper) == pytest.approx(0.75)
    return True


def test_gear_env_cap_policy():
    assert run_function_with_persistent_simulation_app(_test_gear_env_cap_policy)
