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

_MIN_ROBOT_CFG_FORMAT_VERSION = 2.0
"""Oldest ``robot_cfg.kinematics.format_version`` cuRobo v2 reads; v1-layout files carry none."""


def load_patched_robot_yaml(curobo_cfg: CuroboEmbodimentCfg) -> dict:
    """Load an embodiment's cuRobo v2 robot yaml and splice in its downloaded URDF path."""
    robot_cfg_path = retrieve_file_path(curobo_cfg.robot_cfg_template)
    robot_urdf_path = retrieve_file_path(curobo_cfg.robot_urdf)
    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)
    kinematics = robot_yaml["robot_cfg"]["kinematics"]
    format_version = kinematics.get("format_version", 1.0)
    assert format_version >= _MIN_ROBOT_CFG_FORMAT_VERSION, (
        f"cuRobo robot config '{curobo_cfg.robot_cfg_template}' has format_version {format_version}, but cuRobo v2 "
        f"needs >= {_MIN_ROBOT_CFG_FORMAT_VERSION} (v1 files use ee_link/retract_config instead of "
        "tool_frames/default_joint_position)."
    )
    kinematics["urdf_path"] = robot_urdf_path
    return robot_yaml
