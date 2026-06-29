# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""SE(3) and rotation utility functions for the data-generation pipeline."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat

from isaaclab_arena_datagen.geometry.rotation import Rotation
from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3
from isaaclab_arena_datagen.geometry.translation import Translation


def se3_from_pos_quat(translation_3: torch.Tensor, quat_4: torch.Tensor) -> TransformSE3:
    """Build a TransformSE3 from position (3,) and quaternion (w,x,y,z) (4,)."""
    rotation = matrix_from_quat(quat_4.reshape(4))
    return TransformSE3(
        rotation=Rotation(R=rotation.unsqueeze(0)),
        translation=Translation(t=translation_3.unsqueeze(0)),
    )


def rotation_from_normal(normal_3: np.ndarray) -> np.ndarray:
    """Build a 3x3 rotation matrix whose z-column equals the unit normal *normal_3*."""
    norm = np.linalg.norm(normal_3)
    if norm < 1e-12:
        raise ValueError("Cannot build rotation from a zero-length normal vector.")
    z = normal_3 / norm
    ref = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def compute_se3_origin_from_surface(
    points_mesh_n3: np.ndarray,
    normals_mesh_n3: np.ndarray,
    T_origin_from_mesh: np.ndarray,
) -> np.ndarray:
    """Compute per-point SE(3) from the body origin to each surface sample.

    For each surface point, builds a transform whose translation is the
    point position and whose z-axis aligns with the outward surface normal,
    all expressed relative to the body origin frame.

    Args:
        points_mesh_n3: ``(N, 3)`` surface point positions in the mesh frame.
        normals_mesh_n3: ``(N, 3)`` outward normals in the mesh frame.
        T_origin_from_mesh: ``(4, 4)`` transform from mesh frame to body origin.

    Returns:
        ``(N, 3, 4)`` float32 array of T_origin_from_surface SE(3) transforms.
    """
    num_points = points_mesh_n3.shape[0]
    T_origin_from_mesh = T_origin_from_mesh.astype(np.float64)

    result_n34 = np.zeros((num_points, 3, 4), dtype=np.float32)

    for i in range(num_points):
        normal_mesh_3 = normals_mesh_n3[i].astype(np.float64)
        n_len = np.linalg.norm(normal_mesh_3)
        if n_len < 1e-12:
            normal_mesh_3 = np.array([0.0, 0.0, 1.0])
        else:
            normal_mesh_3 = normal_mesh_3 / n_len

        rotation_mesh_from_surface = rotation_from_normal(normal_mesh_3)

        T_mesh_from_surface = np.eye(4, dtype=np.float64)
        T_mesh_from_surface[:3, :3] = rotation_mesh_from_surface
        T_mesh_from_surface[:3, 3] = points_mesh_n3[i].astype(np.float64)

        T_origin_from_surface = T_origin_from_mesh @ T_mesh_from_surface
        result_n34[i] = T_origin_from_surface[:3, :].astype(np.float32)

    return result_n34
