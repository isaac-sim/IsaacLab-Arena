# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test that the default image carries the dependencies the IK-reachability code needs.

These are pure import checks—they need no SimulationApp—and guard against cuRobo and the
``isaaclab_arena_curobo`` package silently going missing from the image.
"""

import importlib
import subprocess
import sys
import yaml

import pytest

pytestmark = pytest.mark.curobo_deps

# cuRobo core modules the sim-free IK oracle relies on.
CUROBO_DEPENDENCY_MODULES = [
    "curobo",
    "curobo.inverse_kinematics",
    "curobo.scene",
    "curobo.types",
]

# The cuRobo-gated package modules, which must import without a running SimulationApp.
ARENA_CUROBO_MODULES = [
    "isaaclab_arena_curobo.utils.frame_utils",
    "isaaclab_arena_curobo.utils.ik_solver_utils",
    "isaaclab_arena_curobo.utils.robot_cfg_utils",
    "isaaclab_arena_curobo.ik_solver",
    "isaaclab_arena_curobo.ik_reachability_validator",
]


def test_validator_registration_does_not_import_curobo():
    """Registering the validator leaves cuRobo unloaded until a reachability solver is constructed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import isaaclab_arena_curobo, sys; assert 'curobo' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", CUROBO_DEPENDENCY_MODULES + ARENA_CUROBO_MODULES)
def test_module_importable(module_name):
    """Each cuRobo dependency and gated arena module imports cleanly in the cuRobo image."""
    importlib.import_module(module_name)


def test_curobo_reports_version():
    """cuRobo exposes a version string, i.e. it is a real install rather than a stub on the path."""
    import curobo

    assert getattr(curobo, "__version__", None), "curobo is importable but reports no __version__"


def test_robot_yaml_uses_downloaded_urdf_path(monkeypatch, tmp_path):
    """The configured URDF is resolved independently from the packaged robot YAML."""
    import isaaclab_arena_curobo.utils.robot_cfg_utils as robot_cfg_utils
    from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg

    robot_yaml_path = tmp_path / "robot.yml"
    robot_yaml_path.write_text(
        yaml.safe_dump({
            "robot_cfg": {
                "kinematics": {
                    "format_version": 2.0,
                    "base_link": "robot_base",
                    "tool_frames": ["tool"],
                    "grasp_contact_link_names": ["hand"],
                    "urdf_path": "relative/robot.urdf",
                    "cspace": {"default_joint_position": [0.0]},
                }
            }
        })
    )
    robot_urdf_path = tmp_path / "robot.urdf"
    robot_urdf_path.touch()
    monkeypatch.setattr(
        robot_cfg_utils,
        "retrieve_file_path",
        lambda path: str(robot_yaml_path if path == "robot.yml" else robot_urdf_path),
    )
    config = CuroboEmbodimentCfg(
        robot_cfg_template="robot.yml",
        robot_urdf="robot.urdf",
        robot_name="robot",
        gripper_joint_names=[],
        gripper_open_joint_pos={},
        gripper_closed_joint_pos={},
        hand_link_names=["hand"],
    )

    kinematics = robot_cfg_utils.load_robot_yaml(config)["robot_cfg"]["kinematics"]

    assert kinematics["format_version"] == 2.0
    assert kinematics["tool_frames"] == ["tool"]
    assert kinematics["grasp_contact_link_names"] == ["hand"]
    assert kinematics["urdf_path"] == str(robot_urdf_path)
    assert kinematics["cspace"]["default_joint_position"] == [0.0]


def test_droid_robot_yaml_uses_curobo_v2_schema():
    """The packaged DROID configuration contains no retired cuRoboV1 fields."""
    from isaaclab_arena_curobo.embodiment_curobo_registry import DROID_CUROBO_CFG

    with open(DROID_CUROBO_CFG.robot_cfg_template) as f:
        kinematics = yaml.safe_load(f)["robot_cfg"]["kinematics"]

    retired_fields = {
        "ee_link",
        "isaac_usd_path",
        "link_names",
        "usd_flip_joint_limits",
        "usd_flip_joints",
        "usd_path",
        "usd_robot_root",
        "use_usd_kinematics",
    }
    assert kinematics["format_version"] == 2.0
    assert kinematics["tool_frames"] == ["base_link"]
    assert not retired_fields.intersection(kinematics)
    assert "default_joint_position" in kinematics["cspace"]
    assert "retract_config" not in kinematics["cspace"]
