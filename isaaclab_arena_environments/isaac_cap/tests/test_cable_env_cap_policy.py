# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the cable environments' Arena-owned CAP policy adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.isaac_cap

_EXPERIMENT_CONFIG_DIRECTORY = Path(__file__).parents[1] / "cable_routing_v2" / "experiment_configs"


def _robot(side: str, joints: list[float]):
    import torch

    return SimpleNamespace(
        joint_names=[*(f"{side}_joint{index}" for index in range(1, 9))],
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([joints], dtype=torch.float32))),
    )


def _cable_environment():
    return SimpleNamespace(
        scene={
            "left_robot": _robot("left", [1, 2, 3, 4, 5, 6, 0.0475, 0.0475]),
            "right_robot": _robot("right", [7, 8, 9, 10, 11, 12, 0.0, 0.0]),
        },
        action_manager=SimpleNamespace(total_action_dim=14),
        device="cpu",
    )


def _cable_policy():
    from isaaclab_arena_environments.isaac_cap.cable_routing_v2.cap_policy import CapYamI2rtPolicy, CapYamI2rtPolicyCfg

    return CapYamI2rtPolicy(
        CapYamI2rtPolicyCfg(
            gripper_open_position=0.0475,
            wire_arm_order=["right", "left"],
            camera_mapping={
                "top_camera": "overhead",
                "right_wrist_camera": "wrist",
                "cable_camera": "cable",
            },
            workspace={"surface_z": 0.767, "cable_radius_m": 0.0035},
        )
    )


def test_cable_env_cap_policy():
    import numpy as np
    import torch

    policy = _cable_policy()
    policy._camera = lambda _env, name: {"camera": name}
    environment = _cable_environment()
    action = policy._hold_action(environment)

    assert action.shape == (14,)
    assert torch.equal(action[:6], torch.arange(1, 7, dtype=torch.float32))
    assert float(action[6]) == pytest.approx(0.0)
    assert torch.equal(action[7:13], torch.arange(7, 13, dtype=torch.float32))
    assert float(action[13]) == pytest.approx(1.0)

    frame = policy._observation_frame(environment, action)

    assert frame["left"]["joint_pos"] == [7, 8, 9, 10, 11, 12, 0.0]
    assert frame["right"]["joint_pos"] == [1, 2, 3, 4, 5, 6, 1.0]
    assert frame["overhead"]["camera"] == "top_camera"
    assert frame["wrist"]["camera"] == "right_wrist_camera"
    assert frame["cable"]["camera"] == "cable_camera"
    assert frame["_isaac_cap"]["workspace"] == {"surface_z": 0.767, "cable_radius_m": 0.0035}

    policy._apply_reply(
        {
            "left": {"joint_pos": np.arange(20, 26, dtype=np.float32), "gripper": 0.25},
            "right": {"joint_pos": np.arange(30, 36, dtype=np.float32), "gripper": 0.75},
            "arm_valid": {"left": True, "right": False},
            "gripper_valid": {"left": False, "right": True},
        },
        action,
    )

    assert torch.equal(action[:6], torch.arange(1, 7, dtype=torch.float32))
    assert float(action[6]) == pytest.approx(0.25)
    assert torch.equal(action[7:13], torch.arange(20, 26, dtype=torch.float32))
    assert float(action[13]) == pytest.approx(1.0)


def test_cable_cap_experiment_configs_use_i2rt_adapter():
    """Both cable experiments resolve to the isolated YAM-I2RT policy."""
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena_environments.isaac_cap.cable_routing_v2.cap_policy import CapYamI2rtPolicyCfg

    for variant in ("easy", "medium"):
        experiment = load_arena_experiment_from_config_file(
            _EXPERIMENT_CONFIG_DIRECTORY / f"cable_{variant}_cap_remote_experiment.yaml",
            device="cuda:0",
        )
        policy = next(iter(experiment.runs.values())).policy

        assert isinstance(policy, CapYamI2rtPolicyCfg)
        assert policy.wire_arm_order == ["right", "left"]
        assert policy.camera_mapping
        assert policy.workspace
