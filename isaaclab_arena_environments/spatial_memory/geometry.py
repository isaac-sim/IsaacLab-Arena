# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched box geometry for the spatial-memory suite: surface sampling, frustum tests and ray-box occlusion.

Poses are ``(x, y, z, qx, qy, qz, qw)``. All functions are pure torch, so they run without a simulator.
"""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils

# Face ids in box-local coordinates: +x, -x, +y, -y, +z, -z.
FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ = range(6)
ALL_FACES = (FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ)


def sample_box_face_points(
    size_xyz: tuple[float, float, float],
    faces: tuple[int, ...] = ALL_FACES,
    spacing_m: float = 0.03,
    min_points_per_axis: int = 2,
    outward_offset_m: float = 0.002,
) -> torch.Tensor:
    """Return a grid of points on the selected faces of a box, in the box frame.

    Points sit ``outward_offset_m`` outside each face so that a ray reaching a point on a front face does not
    count the box itself as an occluder.

    Args:
        size_xyz: Full box extents in metres.
        faces: Face ids to sample (``FACE_PX`` ... ``FACE_NZ``).
        spacing_m: Approximate spacing between neighboring points.
        min_points_per_axis: Minimum number of points along each face axis.
        outward_offset_m: Distance to push each point outside its face.

    Returns:
        Points with shape (P, 3) in the box frame.
    """
    half = [0.5 * s for s in size_xyz]
    face_points = []
    for face in faces:
        axis = face // 2
        sign = 1.0 if face % 2 == 0 else -1.0
        u_axis, v_axis = (a for a in range(3) if a != axis)
        num_u = max(min_points_per_axis, int(math.ceil(size_xyz[u_axis] / spacing_m)) + 1)
        num_v = max(min_points_per_axis, int(math.ceil(size_xyz[v_axis] / spacing_m)) + 1)
        u = torch.linspace(-half[u_axis], half[u_axis], num_u)
        v = torch.linspace(-half[v_axis], half[v_axis], num_v)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
        points = torch.zeros(num_u * num_v, 3)
        points[:, axis] = sign * (half[axis] + outward_offset_m)
        points[:, u_axis] = grid_u.reshape(-1)
        points[:, v_axis] = grid_v.reshape(-1)
        face_points.append(points)
    return torch.cat(face_points, dim=0)


def transform_points(T_A_B: torch.Tensor, points_B: torch.Tensor) -> torch.Tensor:
    """Map points from frame ``B`` into frame ``A``.

    Args:
        T_A_B: Poses with shape (..., 7).
        points_B: Points with shape (..., P, 3); leading dimensions broadcast against ``T_A_B``.

    Returns:
        Points in frame ``A`` with shape (..., P, 3).
    """
    t_A_B = T_A_B[..., :3].unsqueeze(-2)
    q_A_B = T_A_B[..., 3:7].unsqueeze(-2)
    return math_utils.quat_apply(q_A_B, points_B) + t_A_B


def inverse_transform_points(T_A_B: torch.Tensor, points_A: torch.Tensor) -> torch.Tensor:
    """Map points from frame ``A`` into frame ``B`` given ``T_A_B``.

    Args:
        T_A_B: Poses with shape (..., 7).
        points_A: Points with shape (..., P, 3).

    Returns:
        Points in frame ``B`` with shape (..., P, 3).
    """
    t_A_B = T_A_B[..., :3].unsqueeze(-2)
    q_A_B = T_A_B[..., 3:7].unsqueeze(-2)
    return math_utils.quat_apply_inverse(q_A_B, points_A - t_A_B)


def compose_pose_tensors(T_A_B: torch.Tensor, T_B_C: torch.Tensor) -> torch.Tensor:
    """Return ``T_A_C = T_A_B * T_B_C`` for poses with shape (..., 7)."""
    t_A_C, q_A_C = math_utils.combine_frame_transforms(
        T_A_B[..., :3].reshape(-1, 3),
        T_A_B[..., 3:7].reshape(-1, 4),
        T_B_C[..., :3].expand(*T_A_B.shape[:-1], 3).reshape(-1, 3),
        T_B_C[..., 3:7].expand(*T_A_B.shape[:-1], 4).reshape(-1, 4),
    )
    return torch.cat([t_A_C, q_A_C], dim=-1).reshape(*T_A_B.shape[:-1], 7)


def points_in_pinhole_frustum(
    points_C: torch.Tensor,
    focal_x_px: float,
    focal_y_px: float,
    width_px: int,
    height_px: int,
    near_m: float,
    far_m: float,
) -> torch.Tensor:
    """Return whether points in a ROS camera frame (z forward, x right, y down) project inside the image.

    Args:
        points_C: Points with shape (..., 3) in the camera frame.
        focal_x_px: Horizontal focal length in pixels.
        focal_y_px: Vertical focal length in pixels.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        near_m: Near clipping distance.
        far_m: Far clipping distance.

    Returns:
        Boolean tensor with shape (...,).
    """
    z = points_C[..., 2]
    in_depth = (z > near_m) & (z < far_m)
    safe_z = torch.where(in_depth, z, torch.ones_like(z))
    u = focal_x_px * points_C[..., 0] / safe_z
    v = focal_y_px * points_C[..., 1] / safe_z
    return in_depth & (u.abs() <= 0.5 * width_px) & (v.abs() <= 0.5 * height_px)


def segments_blocked_by_boxes(
    origins_W: torch.Tensor,
    points_W: torch.Tensor,
    T_W_boxes: torch.Tensor,
    half_extents: torch.Tensor,
    box_enabled: torch.Tensor | None = None,
    end_margin: float = 1.0e-3,
) -> torch.Tensor:
    """Return whether the segment from each origin to each point passes through any box.

    A box whose interior contains the origin does not block (the camera may sit inside a gripper model).

    Args:
        origins_W: Segment origins with shape (N, 3).
        points_W: Segment end points with shape (N, P, 3).
        T_W_boxes: Box poses with shape (N, K, 7).
        half_extents: Box half extents with shape (K, 3) or (N, K, 3).
        box_enabled: Optional mask with shape (N, K); disabled boxes never block.
        end_margin: Fraction of the segment near the end point that is ignored.

    Returns:
        Boolean tensor with shape (N, P).
    """
    num_envs, num_points, _ = points_W.shape
    num_boxes = T_W_boxes.shape[1]
    if half_extents.dim() == 2:
        half_extents = half_extents.unsqueeze(0).expand(num_envs, num_boxes, 3)
    # Express the origin and each segment direction in every box frame X: (N, K, 3) and (N, K, P, 3).
    q_W_X = T_W_boxes[..., 3:7]
    origins_X = math_utils.quat_apply_inverse(q_W_X, origins_W.unsqueeze(1) - T_W_boxes[..., :3])
    direction_W = points_W - origins_W.unsqueeze(1)
    q_expanded = q_W_X.unsqueeze(2).expand(num_envs, num_boxes, num_points, 4)
    directions_X = math_utils.quat_apply_inverse(
        q_expanded, direction_W.unsqueeze(1).expand(num_envs, num_boxes, num_points, 3)
    )
    origins_X = origins_X.unsqueeze(2)
    half = half_extents.unsqueeze(2)
    # Slab test along the segment parameter t in [0, 1].
    safe_directions = torch.where(directions_X.abs() < 1e-9, torch.full_like(directions_X, 1e-9), directions_X)
    t_low = (-half - origins_X) / safe_directions
    t_high = (half - origins_X) / safe_directions
    t_enter = torch.minimum(t_low, t_high).amax(dim=-1)
    t_exit = torch.maximum(t_low, t_high).amin(dim=-1)
    origin_inside = (origins_X.abs() <= half).all(dim=-1)
    hits = (t_exit >= t_enter) & (t_exit > 0.0) & (t_enter < 1.0 - end_margin) & ~origin_inside
    if box_enabled is not None:
        hits = hits & box_enabled.unsqueeze(-1)
    return hits.any(dim=1)
