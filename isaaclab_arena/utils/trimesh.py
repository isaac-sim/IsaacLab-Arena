# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for transforming and bounding trimesh meshes."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose


def mesh_in_world_frame(mesh: trimesh.Trimesh, pose: Pose) -> trimesh.Trimesh:
    """Return a copy of mesh transformed by pose."""
    transformed = mesh.copy()
    qx, qy, qz, qw = pose.rotation_xyzw
    transform = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
    transform[:3, 3] = np.asarray(pose.position_xyz, dtype=np.float64)
    transformed.apply_transform(transform)
    return transformed


def bounding_box_from_mesh(mesh: trimesh.Trimesh) -> AxisAlignedBoundingBox:
    """Return the axis-aligned bounds of mesh."""
    bounds = mesh.bounds
    return AxisAlignedBoundingBox(
        min_point=tuple(float(v) for v in bounds[0]),
        max_point=tuple(float(v) for v in bounds[1]),
    )
