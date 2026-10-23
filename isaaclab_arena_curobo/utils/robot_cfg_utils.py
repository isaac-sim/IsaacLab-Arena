# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading and patching an embodiment's cuRobo robot config."""

from __future__ import annotations

import yaml
from typing import TYPE_CHECKING

from isaaclab.utils.assets import retrieve_file_path

if TYPE_CHECKING:
    from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg


def load_patched_robot_yaml(curobo_cfg: CuroboEmbodimentCfg) -> dict:
    """Load an embodiment's robot YAML and update its v1 fields for cuRoboV2."""
    robot_cfg_path = retrieve_file_path(curobo_cfg.robot_cfg_template)
    robot_urdf_path = retrieve_file_path(curobo_cfg.robot_urdf)
    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)

    kinematics = robot_yaml["robot_cfg"]["kinematics"]
    kinematics["urdf_path"] = robot_urdf_path
    kinematics["format_version"] = 2.0
    kinematics["tool_frames"] = [curobo_cfg.ee_link_name]
    kinematics["grasp_contact_link_names"] = list(curobo_cfg.hand_link_names)

    # cuRoboV2 is URDF-only and rejects the retired v1 USD and end-effector fields.
    for field_name in (
        "ee_link",
        "isaac_usd_path",
        "link_names",
        "usd_flip_joint_limits",
        "usd_flip_joints",
        "usd_path",
        "usd_robot_root",
        "use_usd_kinematics",
    ):
        kinematics.pop(field_name, None)

    cspace = kinematics["cspace"]
    if "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
    return robot_yaml
