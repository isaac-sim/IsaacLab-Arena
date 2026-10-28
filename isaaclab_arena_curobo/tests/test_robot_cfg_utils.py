# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading an embodiment's cuRobo robot yaml."""

from __future__ import annotations

import yaml

import pytest

pytestmark = pytest.mark.curobo_deps


def _write_robot_cfg(tmp_path, kinematics: dict):
    """Write a robot yaml plus a placeholder URDF, and return a cuRobo embodiment config pointing at them."""
    from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg

    robot_cfg_path = tmp_path / "robot_curobo.yaml"
    robot_cfg_path.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kinematics}}))
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='robot'/>")
    return CuroboEmbodimentCfg(
        robot_cfg_template=str(robot_cfg_path),
        robot_urdf=str(urdf_path),
        robot_name="robot",
        ee_link_name="tool",
        gripper_joint_names=[],
        gripper_open_joint_pos={},
        gripper_closed_joint_pos={},
        hand_link_names=[],
    ), str(urdf_path)


def test_v2_robot_yaml_gets_the_downloaded_urdf_path(tmp_path):
    """A v2-layout config loads as written, with its URDF path pointed at the retrieved URDF."""
    from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml

    curobo_cfg, urdf_path = _write_robot_cfg(
        tmp_path, {"format_version": 2.0, "urdf_path": "urdf/robot.urdf", "tool_frames": ["tool"]}
    )

    kinematics = load_patched_robot_yaml(curobo_cfg)["robot_cfg"]["kinematics"]

    assert kinematics["urdf_path"] == urdf_path
    assert kinematics["tool_frames"] == ["tool"]


def test_v1_robot_yaml_is_rejected_by_name(tmp_path):
    """A v1-layout config (no format_version) fails with the file named, not deep inside cuRobo's loader."""
    from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml

    curobo_cfg, _ = _write_robot_cfg(tmp_path, {"urdf_path": "urdf/robot.urdf", "ee_link": "tool"})

    with pytest.raises(AssertionError, match="robot_curobo.yaml"):
        load_patched_robot_yaml(curobo_cfg)
