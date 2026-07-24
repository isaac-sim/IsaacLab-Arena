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
    """Load an embodiment's cuRobo robot yaml and splice in its downloaded URDF path."""
    robot_cfg_path = retrieve_file_path(curobo_cfg.robot_cfg_template)
    robot_urdf_path = retrieve_file_path(curobo_cfg.robot_urdf)
    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = robot_urdf_path
    return robot_yaml
