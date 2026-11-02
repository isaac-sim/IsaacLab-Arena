# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One camera's rendered images and calibration, shared by VLM observation adapters."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Any

DEPTH_DATA_TYPE = "distance_to_image_plane"
"""Camera data type holding depth along the optical axis, in meters."""


@dataclass
class CameraView:
    """One camera's rendered images and optical-frame calibration for one environment.

    The optical frame C has +X right, +Y down, and +Z forward.
    """

    rgb: np.ndarray
    """Rendered HWC uint8 pixels."""

    intrinsic_matrix: np.ndarray
    """3x3 intrinsics in rendered pixels."""

    t_W_C: np.ndarray
    """Optical-frame origin in the world frame, in meters."""

    q_W_C: np.ndarray
    """XYZW rotation from the optical frame into the world frame."""

    poses_in_robot_roots: dict[str, dict[str, list[float]]]
    """Optical-frame position_root_m and quaternion_root_xyzw in each commanded robot root, by asset name."""

    depth: np.ndarray | None = None
    """Rendered HxW depth along the optical axis in meters, when the camera renders it."""

    def to_camera_geometry(self) -> dict[str, Any]:
        """Return this camera's calibration as the JSON camera_geometry entry sent to the model."""
        return {
            "image_size_px": [self.rgb.shape[1], self.rgb.shape[0]],
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "optical_frame": "x_right_y_down_z_forward",
            "position_world_m": self.t_W_C.tolist(),
            "quaternion_world_xyzw": self.q_W_C.tolist(),
            "poses_in_robot_roots": self.poses_in_robot_roots,
        }

    def measure_point(self, u: float, v: float, root_name: str) -> tuple[np.ndarray | None, float]:
        """Return the surface point under a rendered pixel in a robot root frame R, and its depth.

        Args:
            u: Rendered-image column, in pixels.
            v: Rendered-image row, in pixels.
            root_name: Robot asset whose root frame expresses the point.

        Returns:
            The point in meters, or None when the pixel is outside the image or sees no surface,
            and the depth read at that pixel.
        """
        assert self.depth is not None, f"Measuring points requires {DEPTH_DATA_TYPE} depth"
        height, width = self.depth.shape
        if not (0.0 <= u < width and 0.0 <= v < height):
            return None, float("nan")
        # Intrinsics place pixel centers at half-integer coordinates, so pixel i covers [i, i + 1).
        z_C = float(self.depth[int(np.floor(v)), int(np.floor(u))])
        if not np.isfinite(z_C) or z_C <= 0.0:
            return None, z_C
        p_C = np.linalg.solve(self.intrinsic_matrix, np.array([u, v, 1.0])) * z_C
        pose = self.poses_in_robot_roots[root_name]
        p_R = Rotation.from_quat(pose["quaternion_root_xyzw"]).apply(p_C) + np.asarray(pose["position_root_m"])
        return p_R, z_C
