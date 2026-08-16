# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml
from dataclasses import fields
from pathlib import Path

import pytest

from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "isaaclab_arena_environments" / "G1_Factory" / "policy_configs" / "benchmark"

EXPECTED_CONFIGS = {
    "LMBoxLift": "Walk to table and pick up the box",
    "LMBoxLiftFloor": "Walk to the box, pick it up from the floor",
    "LMBoxTableToShelfPnP": "Move box from table to shelf",
    "LMDrillLift": "Walk to the table and pick up the power drill",
    "LMDrillLiftObs": "Pick up the drill with a shelf obstacle",
    "LMDrillPnP": "Move power drill from one table to another",
    "LMPickDrillFromHolder": "Pick up the drill from holder",
    "LMPushButton": "Walk to the console and press red button",
    "LMPushShelfForward": "Push high wheeled shelf to marked area",
}


def _load_policy_config(env_name: str) -> Gr00tClosedloopPolicyCfg:
    config_path = CONFIG_DIR / f"{env_name}_gr00t_closedloop.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    valid_fields = {field.name for field in fields(Gr00tClosedloopPolicyCfg)}
    unknown_fields = sorted(set(config_data) - valid_fields)
    assert unknown_fields == [], f"{config_path} contains fields not supported by Gr00tClosedloopPolicyCfg"

    return Gr00tClosedloopPolicyCfg(**config_data)


def test_expected_policy_configs_exist():
    expected_files = {f"{env_name}_gr00t_closedloop.yaml" for env_name in EXPECTED_CONFIGS}
    actual_files = {path.name for path in CONFIG_DIR.glob("*_gr00t_closedloop.yaml")}

    assert actual_files == expected_files


@pytest.mark.parametrize("env_name", sorted(EXPECTED_CONFIGS))
def test_policy_config_loads_against_current_gr00t_schema(env_name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)

    policy_config = _load_policy_config(env_name)

    assert policy_config.language_instruction == EXPECTED_CONFIGS[env_name]
    assert policy_config.action_horizon == 16
    assert policy_config.action_chunk_length == 16
    assert policy_config.embodiment_tag == "NEW_EMBODIMENT"
    assert policy_config.task_mode_name == "g1_locomanipulation"
    assert policy_config.pov_cam_name_sim == ["robot_head_cam_rgb"]
    assert tuple(policy_config.original_image_size) == (480, 640, 3)
    assert tuple(policy_config.target_image_size) == (480, 640, 3)
    assert policy_config.modality_config_path == "isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py"
    assert (
        str(policy_config.policy_joints_config_path)
        == "isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml"
    )
    assert str(policy_config.action_joints_config_path) == "isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml"
    assert str(policy_config.state_joints_config_path) == "isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml"
