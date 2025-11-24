# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_arena_g1.g1_env.g1_supplemental_info import (
    G1SupplementalInfo,
    G1SupplementalInfoWaistLowerAndUpperBody,
    G1SupplementalInfoWaistUpperBody,
)
from isaaclab_arena_g1.g1_env.robot_model import RobotModel


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

    # NOTE(peterd, 9/25/2025): Update OV paths once release location is finalized
    robot_model_config = {
        "asset_path": "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/robot_model/g1/",
        "urdf_path": "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf",
    }

    asset_path_local = retrieve_file_path(robot_model_config["asset_path"], force_download=True)
    urdf_path_local = retrieve_file_path(robot_model_config["urdf_path"], force_download=True)

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
