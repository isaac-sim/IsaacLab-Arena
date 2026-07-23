# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for computing spatial relationships between objects in a scene.
This module provides functions for:
- Computing bounding boxes from USD assets
- Calculating placement poses based on semantic relationships (e.g., "on_top_of")
- Supporting randomized placement with specified constraints
- AxisAlignedBoundingBox with always-tensor storage supporting single (N=1) and batched (N>1) modes
"""

import torch
from collections.abc import Sequence

from isaaclab_arena.utils.pose import Pose


class AxisAlignedBoundingBox:
    """Axis-aligned bounding box storing local extents. Use get_corners_at(pos) for world-space corners.

    Stores min/max extents as (N, 3) float32 tensors where N is the number of environments.
    All properties consistently return tensors: (N, 3) for point values, (N,) for scalars.
    Constructor accepts tuples, 1D tensors, or (N, 3) tensors.
    """

    def __init__(
        self,
        min_point: tuple[float, float, float] | torch.Tensor,
        max_point: tuple[float, float, float] | torch.Tensor,
    ):
        self._min_point = self._to_batched_tensor(min_point)
        self._max_point = self._to_batched_tensor(max_point)
        assert self._min_point.shape == self._max_point.shape
        assert self._min_point.shape[-1] == 3

    def __repr__(self) -> str:
        return f"AxisAlignedBoundingBox(min_point={self._min_point}, max_point={self._max_point})"

    def __getitem__(self, idx: int) -> "AxisAlignedBoundingBox":
        """Select row idx (one env/candidate), returning a single-box (N=1) bbox."""
        assert 0 <= idx < self.num_envs, f"index {idx} out of range for bbox with num_envs={self.num_envs}."
        return AxisAlignedBoundingBox(
            min_point=self._min_point[idx : idx + 1], max_point=self._max_point[idx : idx + 1]
        )

    @staticmethod
    def union(boxes: Sequence["AxisAlignedBoundingBox"]) -> "AxisAlignedBoundingBox":
        """Return the tightest box enclosing every box in a common frame (component-wise min/max).

        Args:
            boxes: Non-empty sequence of boxes sharing num_envs, except N=1 boxes which broadcast.

        Returns:
            A box with num_envs = max over inputs, enclosing the union of all inputs.
        """
        assert len(boxes) > 0, "union requires at least one bounding box."
        num_envs = max(box.num_envs for box in boxes)
        for box in boxes:
            assert box.num_envs in (
                1,
                num_envs,
            ), f"union requires boxes with matching num_envs or N=1; got {box.num_envs} vs {num_envs}."
        mins = torch.stack([box._min_point.expand(num_envs, 3) for box in boxes], dim=0)
        maxs = torch.stack([box._max_point.expand(num_envs, 3) for box in boxes], dim=0)
        return AxisAlignedBoundingBox(min_point=mins.amin(dim=0), max_point=maxs.amax(dim=0))

    @staticmethod
    def _to_batched_tensor(value: tuple[float, float, float] | torch.Tensor) -> torch.Tensor:
        """Convert tuple, 1-D tensor, or (N, 3) tensor to (N, 3) float32 tensor."""
        if isinstance(value, tuple):
            return torch.tensor([value], dtype=torch.float32)
        if value.dim() == 1:
            return value.unsqueeze(0).float()
        return value.float()

    @property
    def min_point(self) -> torch.Tensor:
        """Local minimum extent (x, y, z) relative to object origin. Shape (N, 3)."""
        return self._min_point

    @property
    def max_point(self) -> torch.Tensor:
        """Local maximum extent (x, y, z) relative to object origin. Shape (N, 3)."""
        return self._max_point

    @property
    def num_envs(self) -> int:
        """Number of environments (leading dimension N)."""
        return self._min_point.shape[0]

    def is_batch_invariant(self) -> bool:
        """Return True when every env shares the same bbox extents."""
        return bool(
            torch.allclose(self._min_point, self._min_point[:1].expand_as(self._min_point))
            and torch.allclose(self._max_point, self._max_point[:1].expand_as(self._max_point))
        )

    @property
    def size(self) -> torch.Tensor:
        """Returns the size (width, depth, height) of the bounding box. Shape (N, 3)."""
        return self._max_point - self._min_point

    @property
    def center(self) -> torch.Tensor:
        """Returns the center point of the bounding box. Shape (N, 3)."""
        return (self._min_point + self._max_point) * 0.5

    @property
    def top_surface_z(self) -> torch.Tensor:
        """Returns the z-coordinate of the top surface. Shape (N,)."""
        return self._max_point[:, 2]

    @property
    def bottom_surface_z(self) -> torch.Tensor:
        """Returns the z-coordinate of the bottom surface. Shape (N,)."""
        return self._min_point[:, 2]

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Get 8 corners of this bounding box, optionally offset by position.

        Args:
            pos: If provided, world position (x, y, z) to offset corners by.
                 If None, returns corners in local/object frame.

        Returns:
            Tensor of shape (N, 8, 3) with corners ordered: bottom 4, then top 4.
        """
        min_pt, max_pt = self._min_point, self._max_point
        corners = torch.stack(
            [
                torch.stack([min_pt[:, 0], min_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-front-left
                torch.stack([max_pt[:, 0], min_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-front-right
                torch.stack([max_pt[:, 0], max_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-back-right
                torch.stack([min_pt[:, 0], max_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-back-left
                torch.stack([min_pt[:, 0], min_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-front-left
                torch.stack([max_pt[:, 0], min_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-front-right
                torch.stack([max_pt[:, 0], max_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-back-right
                torch.stack([min_pt[:, 0], max_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-back-left
            ],
            dim=1,
        )
        if pos is not None:
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            corners = corners + pos.unsqueeze(1)
        return corners

    def scaled(self, scale: tuple[float, float, float] | torch.Tensor) -> "AxisAlignedBoundingBox":
        """Return a new bounding box with scale applied.

        Args:
            scale: Scale factors (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with scaled dimensions.
        """
        scale = self._to_batched_tensor(scale)
        return AxisAlignedBoundingBox(min_point=self._min_point * scale, max_point=self._max_point * scale)

    def to(self, device: torch.device) -> "AxisAlignedBoundingBox":
        """Return a new bounding box with tensors on *device*."""
        return AxisAlignedBoundingBox(min_point=self._min_point.to(device), max_point=self._max_point.to(device))

    def translated(self, offset: tuple[float, float, float] | torch.Tensor) -> "AxisAlignedBoundingBox":
        """Return a new bounding box translated by an offset.

        Args:
            offset: Translation offset (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with translated position.
        """
        offset = self._to_batched_tensor(offset)
        return AxisAlignedBoundingBox(min_point=self._min_point + offset, max_point=self._max_point + offset)

    def centered(self) -> "AxisAlignedBoundingBox":
        """Return a new bounding box centered around the origin.

        The returned bbox has the same size but is shifted so that its
        center is at (0, 0, 0).

        Returns:
            New AxisAlignedBoundingBox centered at origin.
        """
        center = (self._min_point + self._max_point) * 0.5
        return AxisAlignedBoundingBox(min_point=self._min_point - center, max_point=self._max_point - center)

    def overlaps(self, other: "AxisAlignedBoundingBox", margin: float = 0.0) -> torch.Tensor:
        """Check if two AABBs overlap in 3D.

        Args:
            other: The other bounding box to test against.
            margin: Minimum required separation in meters. A positive value
                rejects placements where the gap is smaller than margin.

        Returns:
            Bool tensor of shape (N,). True where volumes overlap (or are closer than margin).
        """
        return (
            (self._max_point[:, 0] + margin > other._min_point[:, 0])
            & (other._max_point[:, 0] + margin > self._min_point[:, 0])
            & (self._max_point[:, 1] + margin > other._min_point[:, 1])
            & (other._max_point[:, 1] + margin > self._min_point[:, 1])
            & (self._max_point[:, 2] + margin > other._min_point[:, 2])
            & (other._max_point[:, 2] + margin > self._min_point[:, 2])
        )

    def rotated_90_around_z(self, quarters: int) -> "AxisAlignedBoundingBox":
        """Rotate AABB by quarters * 90° around Z axis.

        Only 90° increments are supported to preserve axis-alignment without size increase.

        Args:
            quarters: Number of 90° rotations (0=0°, 1=90°, 2=180°, 3=270°/-90°).

        Returns:
            New AxisAlignedBoundingBox rotated around Z axis.
        """
        quarters = quarters % 4
        min_x, min_y, min_z = self._min_point[:, 0], self._min_point[:, 1], self._min_point[:, 2]
        max_x, max_y, max_z = self._max_point[:, 0], self._max_point[:, 1], self._max_point[:, 2]
        if quarters == 0:
            return AxisAlignedBoundingBox(min_point=self._min_point.clone(), max_point=self._max_point.clone())
        elif quarters == 1:  # 90° CCW
            return AxisAlignedBoundingBox(
                min_point=torch.stack([-max_y, min_x, min_z], dim=1),
                max_point=torch.stack([-min_y, max_x, max_z], dim=1),
            )
        elif quarters == 2:  # 180°
            return AxisAlignedBoundingBox(
                min_point=torch.stack([-max_x, -max_y, min_z], dim=1),
                max_point=torch.stack([-min_x, -min_y, max_z], dim=1),
            )
        else:  # 270° CCW / -90° (quarters == 3)
            return AxisAlignedBoundingBox(
                min_point=torch.stack([min_y, -max_x, min_z], dim=1),
                max_point=torch.stack([max_y, -min_x, max_z], dim=1),
            )

    def enclosing_after_rotation(
        self, rotation_xyzw: tuple[float, float, float, float] | torch.Tensor
    ) -> "AxisAlignedBoundingBox":
        """Refit to the axis-aligned box enclosing this box under an arbitrary rotation.

        Rotates the eight corners about the object origin by ``rotation_xyzw`` and returns the
        tightest AABB containing them. Conservative (larger than the true rotated box) for any
        non-axis-aligned rotation. Unlike :meth:`rotated_around_z`, roll and pitch tilt the box
        out of plane, so the Z extent may grow as well.

        Args:
            rotation_xyzw: Quaternion ``(x, y, z, w)`` applied about the object origin. A single
                quaternion rotates every stacked box equally.

        Returns:
            New AxisAlignedBoundingBox enclosing the rotated corners.
        """
        device = self._min_point.device
        quat = torch.as_tensor(rotation_xyzw, dtype=torch.float32, device=device).reshape(-1)
        assert quat.shape == (
            4,
        ), f"enclosing_after_rotation expects a single (x, y, z, w) quaternion, got {tuple(quat.shape)}."
        qx, qy, qz, qw = quat.unbind(0)
        # Rotation matrix from the (x, y, z, w) quaternion.
        rot = torch.stack([
            torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)]),
            torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)]),
            torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]),
        ])  # (3, 3)
        corners = self.get_corners_at()  # (N, 8, 3)
        rotated = corners @ rot.transpose(0, 1)  # (N, 8, 3): each corner mapped by rot
        return AxisAlignedBoundingBox(
            min_point=rotated.min(dim=1).values,
            max_point=rotated.max(dim=1).values,
        )

    def rotated_around_z(self, angle_rad: float | torch.Tensor) -> "AxisAlignedBoundingBox":
        """Refit to the axis-aligned box enclosing this box rotated by angle_rad around Z.

        Conservative (larger than the true rotated box) except at 90° multiples; Z extents unchanged.

        Args:
            angle_rad: Yaw in radians. A scalar rotates every box equally; a 1-D tensor gives
                per-box angles (or, for a single box, one box per angle).

        Returns:
            New AxisAlignedBoundingBox enclosing the rotated box.
        """
        device = self._min_point.device
        angles = torch.as_tensor(angle_rad, dtype=torch.float32, device=device).reshape(-1)  # (M,)

        num_boxes, num_angles = self._min_point.shape[0], angles.shape[0]
        assert num_boxes == 1 or num_angles == 1 or num_boxes == num_angles, (
            "rotated_around_z requires one box, one angle, or equal counts; "
            f"got {num_boxes} boxes and {num_angles} angles."
        )

        cos = torch.cos(angles).unsqueeze(1)  # (M, 1)
        sin = torch.sin(angles).unsqueeze(1)  # (M, 1)

        # XY footprint corners relative to the object origin: (N, 4).
        min_x, min_y = self._min_point[:, 0], self._min_point[:, 1]
        max_x, max_y = self._max_point[:, 0], self._max_point[:, 1]
        corners_x = torch.stack([min_x, max_x, max_x, min_x], dim=1)  # (N, 4)
        corners_y = torch.stack([min_y, min_y, max_y, max_y], dim=1)  # (N, 4)

        # Rotate corners (broadcasts (N, 4) with (M, 1)).
        rot_x = corners_x * cos - corners_y * sin
        rot_y = corners_x * sin + corners_y * cos

        new_min_x = rot_x.min(dim=1).values  # (L,)
        new_max_x = rot_x.max(dim=1).values
        new_min_y = rot_y.min(dim=1).values
        new_max_y = rot_y.max(dim=1).values

        # Z extents are invariant under Z rotation; broadcast to the output leading dim.
        out_len = new_min_x.shape[0]
        min_z, max_z = self._min_point[:, 2], self._max_point[:, 2]
        if min_z.shape[0] == 1 and out_len > 1:
            min_z = min_z.expand(out_len)
            max_z = max_z.expand(out_len)

        return AxisAlignedBoundingBox(
            min_point=torch.stack([new_min_x, new_min_y, min_z], dim=1),
            max_point=torch.stack([new_max_x, new_max_y, max_z], dim=1),
        )

    def rotated_by_quat(
        self, rotation_xyzw: tuple[float, float, float, float] | torch.Tensor
    ) -> "AxisAlignedBoundingBox":
        """Refit to the axis-aligned box enclosing this box rotated about its origin by a quaternion.

        Args:
            rotation_xyzw: Rotation quaternion as (x, y, z, w). A single quaternion rotates every box
                equally; a (M, 4) tensor gives per-box quaternions (or, for a single box, one box per
                quaternion), matching rotated_around_z's batching.
        """
        corners = self.get_corners_at()  # (N, 8, 3)
        quats = torch.as_tensor(rotation_xyzw, dtype=corners.dtype, device=corners.device).reshape(-1, 4)  # (M, 4)

        num_boxes, num_quats = corners.shape[0], quats.shape[0]
        assert (
            num_boxes == 1 or num_quats == 1 or num_boxes == num_quats
        ), f"rotated_by_quat requires one box, one quat, or equal counts; got {num_boxes} boxes and {num_quats} quats."
        out_len = max(num_boxes, num_quats)
        if num_boxes == 1 and out_len > 1:
            corners = corners.expand(out_len, 8, 3)

        # Rotate the 8 object-frame corners by the quaternion (v + 2w(a×v) + 2a×(a×v)), then min/max.
        axis = quats[:, :3].view(num_quats, 1, 3).expand(out_len, 8, 3)  # (L, 8, 3)
        qw = quats[:, 3].view(num_quats, 1, 1).expand(out_len, 1, 1)  # (L, 1, 1)
        axis_cross = torch.linalg.cross(axis, corners, dim=-1)
        rotated = corners + 2.0 * qw * axis_cross + 2.0 * torch.linalg.cross(axis, axis_cross, dim=-1)
        return AxisAlignedBoundingBox(min_point=rotated.amin(dim=1), max_point=rotated.amax(dim=1))


def quaternion_to_90_deg_z_quarters(rotation_xyzw: tuple[float, float, float, float], tol_deg: float = 1.0) -> int:
    """Convert a quaternion to 90° rotation quarters around Z axis.

    Only supports rotations that are multiples of 90° around the Z axis.
    Raises AssertionError for any other rotation.

    Args:
        rotation_xyzw: Quaternion as (x, y, z, w).
        tol_deg: Tolerance in degrees for how close the angle must be to a 90° multiple.

    Returns:
        Number of 90° quarters (0, 1, 2, or 3).

    Raises:
        AssertionError: If the quaternion is not a pure Z rotation or not a 90° multiple.
    """
    import math

    x, y, z, w = rotation_xyzw

    # Must be a pure Z rotation (x and y components must be ~0)
    assert (
        abs(x) < 1e-3 and abs(y) < 1e-3
    ), f"Only rotations around Z axis are supported. Got quaternion (w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})."

    # Compute rotation angle around Z and normalize to [0°, 360°)
    angle_deg = math.degrees(2 * math.atan2(z, w)) % 360
    quarters = round(angle_deg / 90) % 4
    remainder_deg = min(angle_deg % 90, 90 - angle_deg % 90)

    assert remainder_deg < tol_deg, (
        "Only 90° rotation multiples around Z are supported. "
        f"Got {angle_deg:.1f}° (nearest 90° multiple: {quarters * 90}°)."
    )

    return quarters


def get_random_pose_within_bounding_box(bbox: AxisAlignedBoundingBox, seed: int | None = None) -> Pose:
    """Generate a random pose (position and identity rotation) with position uniformly
       sampled within a bounding box.

    Args:
        bbox: Bounding box defining the valid region for sampling
        seed: Optional random seed for reproducibility

    Returns:
        Pose with random position within bbox and identity rotation
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Get workspace bounds as (3,) tensors from the first (and typically only) environment
    min_point = bbox.min_point[0]
    max_point = bbox.max_point[0]

    # Sample random position uniformly within workspace bounds
    random_position = min_point + (max_point - min_point) * torch.rand(3)

    pose = Pose(position_xyz=tuple(random_position.tolist()), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    return pose
