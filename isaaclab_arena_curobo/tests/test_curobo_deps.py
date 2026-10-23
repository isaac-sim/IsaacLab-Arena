# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test that the default image carries the dependencies the IK-reachability code needs.

These are pure import checks—they need no SimulationApp—and guard against cuRobo and the
``isaaclab_arena_curobo`` package silently going missing from the image.
"""

import importlib
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


@pytest.mark.parametrize("module_name", CUROBO_DEPENDENCY_MODULES + ARENA_CUROBO_MODULES)
def test_module_importable(module_name):
    """Each cuRobo dependency and gated arena module imports cleanly in the cuRobo image."""
    importlib.import_module(module_name)


def test_curobo_reports_version():
    """cuRobo exposes a version string, i.e. it is a real install rather than a stub on the path."""
    import curobo

    assert getattr(curobo, "__version__", None), "curobo is importable but reports no __version__"


def test_v1_robot_yaml_is_updated_for_curobo_v2(monkeypatch, tmp_path):
    """Arena's existing robot descriptions are converted to the cuRoboV2 schema."""
    import isaaclab_arena_curobo.utils.robot_cfg_utils as robot_cfg_utils
    from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg

    robot_yaml_path = tmp_path / "robot.yml"
    robot_yaml_path.write_text(
        yaml.safe_dump({
            "robot_cfg": {
                "kinematics": {
                    "base_link": "robot_base",
                    "ee_link": "old_tool",
                    "link_names": ["old_tool"],
                    "use_usd_kinematics": False,
                    "cspace": {"retract_config": [0.0]},
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
        ee_link_name="tool",
        gripper_joint_names=[],
        gripper_open_joint_pos={},
        gripper_closed_joint_pos={},
        hand_link_names=["hand"],
    )

    kinematics = robot_cfg_utils.load_patched_robot_yaml(config)["robot_cfg"]["kinematics"]

    assert kinematics["format_version"] == 2.0
    assert kinematics["tool_frames"] == ["tool"]
    assert kinematics["grasp_contact_link_names"] == ["hand"]
    assert kinematics["urdf_path"] == str(robot_urdf_path)
    assert kinematics["cspace"]["default_joint_position"] == [0.0]
    assert "ee_link" not in kinematics
    assert "link_names" not in kinematics
    assert "retract_config" not in kinematics["cspace"]
