# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena_g1.g1_env.g1_supplemental_info import (
    G1SupplementalInfo,
    G1SupplementalInfoWaistLowerAndUpperBody,
    G1SupplementalInfoWaistUpperBody,
)
from isaaclab_arena_g1.g1_env.robot_model import RobotModel


def _retrieve_urdf_assets(urdf_url: str) -> tuple[str, str]:
    """Retrieve a URDF and its relative mesh files, returning the local URDF and asset paths."""
    urdf_path_local = retrieve_file_path(urdf_url, force_download=False)
    urdf_root = ET.parse(urdf_path_local).getroot()

    mesh_paths = {mesh_path for mesh in urdf_root.iter("mesh") if (mesh_path := mesh.get("filename")) is not None}
    for mesh_path in sorted(mesh_paths):
        retrieve_file_path(urljoin(urdf_url, mesh_path), force_download=False)

    return urdf_path_local, str(Path(urdf_path_local).parent)


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body"] = "lower_body",
):
    """
    Instantiate a G1 robot model with configurable waist location, and summarize the supplemental info.

    Args:
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)

    Returns:
        RobotModel: Configured G1 robot model
    """

    urdf_url = f"{ARENA_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf"
    urdf_path_local, asset_path_local = _retrieve_urdf_assets(urdf_url)

    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"
    # Choose supplemental info based on waist location preference
    if waist_location == "lower_body":
        robot_model_supplemental_info = G1SupplementalInfo()
    elif waist_location == "upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistUpperBody()
    elif waist_location == "lower_and_upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistLowerAndUpperBody()

    robot_model = RobotModel(
        urdf_path_local,
        asset_path_local,
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
