# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from dataclasses import dataclass


@dataclass
class Pose:
    """Transform taking frame A to frame B.

    T_A_B = (t_B_A, q_B_A)

    p_B = p_A + t_B_A
    q_B = q_A * q_B_A
    """

    position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation vector from frame A to frame B."""

    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion from frame A to frame B. Order is (w, x, y, z)."""

    @staticmethod
    def identity() -> "Pose":
        return Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
