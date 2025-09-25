# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

from isaac_arena.embodiments.g1.g1_supplemental_info import (
    G1SupplementalInfo,
    G1SupplementalInfoWaistLowerAndUpperBody,
    G1SupplementalInfoWaistUpperBody,
)
from isaac_arena.embodiments.g1.robot_model import RobotModel


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
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
