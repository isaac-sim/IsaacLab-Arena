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
- AxisAlignedBoundingBox and OrientedBoundingBox with always-tensor storage supporting single (N=1) and batched (N>1) modes
"""

import math
import torch
from abc import ABC, abstractmethod

from isaaclab_arena.utils.pose import Pose


class BoundingBox(ABC):
    """Abstract batch of boxes with shared transform and overlap queries.

    Boxes live in a containing frame W, such as the world, and B is a box's local frame.
    Pairwise queries take boxes in the same frame and on the same device, pairing rows N/N
    or 1/N, and require finite, nonempty geometry.
    """

    def __init__(self):
        """Initialize the base; subclasses store the box geometry."""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of box rows N."""

    @abstractmethod
    def __getitem__(self, idx: int) -> "BoundingBox":
        """Select one row, preserving its leading dimension."""

    @abstractmethod
    def to(self, device: torch.device | str) -> "BoundingBox":
        """Return boxes with their tensors on device."""

    @abstractmethod
    def translated(self, offset: tuple[float, float, float] | torch.Tensor) -> "BoundingBox":
        """Return boxes translated by offset (3,) or (N, 3) in W."""

    @abstractmethod
    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Return corners in W (N, 8, 3), optionally translated by pos (3,) or (N, 3)."""

    @abstractmethod
    def get_axis_aligned_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return enclosing min/max points in W, each with shape (N, 3)."""

    @abstractmethod
    def _get_finite_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return center_W (N, 3), half-extents (N, 3) and xyzw rotations q_W_B (N, 4) for queries.

        Implementations must return finite centers, nonnegative half-extents and unit
        quaternions, asserting this when their storage can hold other values.
        """

    def transformed(
        self,
        t_A_W: tuple[float, float, float] | torch.Tensor,
        q_A_W: tuple[float, float, float, float] | torch.Tensor,
    ) -> "OrientedBoundingBox":
        """Map boxes from W into frame A, preserving their oriented shape.

        Args:
            t_A_W: Position of W's origin in A, shape (3,) or (N, 3).
            q_A_W: Unit xyzw rotation mapping W into A, shape (4,) or (N, 4).

        Returns:
            OrientedBoundingBox in frame A.
        """
        from isaaclab.utils.math import quat_apply, quat_mul

        center_W, half_extents, q_W_B = self._get_finite_geometry()
        t_A_W = _batched_tensor(t_A_W, 3, center_W.device)
        q_A_W = _unit_quaternion(_batched_tensor(q_A_W, 4, center_W.device))
        center_W, half_extents, q_W_B, t_A_W, q_A_W = _broadcast_rows(center_W, half_extents, q_W_B, t_A_W, q_A_W)
        center_A = quat_apply(q_A_W, center_W) + t_A_W
        q_A_B = quat_mul(q_A_W, q_W_B)
        return OrientedBoundingBox(center_A, half_extents, q_A_B)

    def to_aabb(self) -> "AxisAlignedBoundingBox":
        """Return the enclosing AxisAlignedBoundingBox in W."""
        return AxisAlignedBoundingBox(*self.get_axis_aligned_bounds())

    def overlaps(self, other: "BoundingBox", margin: float = 0.0) -> torch.Tensor:
        """Return strict separating-axis test (SAT) overlap or projected-clearance violation, shape (N,).

        At zero margin, touching on a separating axis is not overlap. Exactly parallel
        edge pairs contribute no axis; small nonzero cross axes remain in the test.
        Float32 roundoff can change the verdict at touching or margin boundaries.

        Args:
            other: Finite boxes in the same frame and device; rows pair N/N or 1/N.
            margin: Nonnegative required gap along SAT axes, in input length units;
                not Euclidean clearance.
        """
        depths, norms = self._sat_depths(other, margin)
        return ((depths > 0) | (norms == 0)).all(dim=-1)

    def overlap_volume_upper_bound(self, other: "BoundingBox", margin: float = 0.0) -> torch.Tensor:
        """Return a differentiable upper bound on the overlap volume, in cubed length units, shape (N,).

        It is the smaller interval-overlap volume along either box's face axes: exact for axis-aligned pairs at
        zero margin, but possibly much larger for rotated boxes, even positive for boxes separated only by an
        edge-cross-edge axis. It is zero when a face axis separates the boxes or this box has a zero-size dimension.

        Args:
            other: Finite boxes in the same frame and device; rows pair N/N or 1/N.
            margin: Finite nonnegative length added to other's projected radii.
                Positive margins make the result asymmetric.
        """
        from isaaclab.utils.math import matrix_from_quat

        # Here A and B are the local frames of this box and other.
        center_a_W, half_extents_a, q_W_A, center_b_W, half_extents_b, q_W_B = self._get_paired_geometry(other)
        overlap_bound = CachedOverlapVolumeUpperBound(
            half_extents_a, matrix_from_quat(q_W_A), half_extents_b, matrix_from_quat(q_W_B)
        )
        return overlap_bound.compute(center_a_W, center_b_W, margin)

    def _get_paired_geometry(self, other: "BoundingBox") -> tuple[torch.Tensor, ...]:
        """Return both boxes' finite geometry on a shared device, broadcast to shared rows.

        Returns:
            center_a_W, half_extents_a, q_W_A, center_b_W, half_extents_b, q_W_B, where A and B
            are the local frames of this box and other.
        """
        center_a_W, half_extents_a, q_W_A = self._get_finite_geometry()
        center_b_W, half_extents_b, q_W_B = other._get_finite_geometry()
        assert center_a_W.device == center_b_W.device, "Bounding boxes must share a device."
        return _broadcast_rows(center_a_W, half_extents_a, q_W_A, center_b_W, half_extents_b, q_W_B)

    def _sat_depths(self, other: "BoundingBox", margin: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected depths and axis norms (N, 15) over the separating-axis candidates."""
        from isaaclab.utils.math import matrix_from_quat

        _assert_valid_margin(margin)
        center_a_W, half_extents_a, q_W_A, center_b_W, half_extents_b, q_W_B = self._get_paired_geometry(other)
        # Here A and B are the local frames of this box and other. Rows are each box's local axes in W.
        axes_a_W = matrix_from_quat(q_W_A).transpose(1, 2)
        axes_b_W = matrix_from_quat(q_W_B).transpose(1, 2)
        cross_axes_W = torch.linalg.cross(axes_a_W.unsqueeze(2), axes_b_W.unsqueeze(1), dim=-1).flatten(1, 2)
        candidate_axes_W = torch.cat((axes_a_W, axes_b_W, cross_axes_W), dim=1)
        norms = torch.linalg.vector_norm(candidate_axes_W, dim=-1)
        # Absolute dot products of each candidate axis with each box's unit local axes, shape (N, 15, 3).
        alignment_a = torch.einsum("nkd,njd->nkj", candidate_axes_W, axes_a_W).abs()
        alignment_b = torch.einsum("nkd,njd->nkj", candidate_axes_W, axes_b_W).abs()
        radius_a = (alignment_a * half_extents_a.unsqueeze(1)).sum(dim=-1)
        radius_b = (alignment_b * half_extents_b.unsqueeze(1)).sum(dim=-1)
        distance = torch.einsum("nkd,nd->nk", candidate_axes_W, center_b_W - center_a_W).abs()
        return radius_a + radius_b + margin * norms - distance, norms


class OrientedBoundingBox(BoundingBox):
    """Finite boxes defined by a center, local half-extents and a rotation.

    The rotation q_W_B maps directions in the box frame B into the containing frame W, for example

        x_axis_W = q_W_B * x_axis_B

    Every box is finite and nonempty; zero-size dimensions are allowed. Inputs of shape (3,) or (4,)
    become single rows, stored as (N, 3) and (N, 4) float32 tensors and broadcast against N.
    Construction copies the inputs, and tensor properties return copies.
    """

    def __init__(
        self,
        center_W: tuple[float, float, float] | torch.Tensor,
        half_extents: tuple[float, float, float] | torch.Tensor,
        rotation_W_B_xyzw: tuple[float, float, float, float] | torch.Tensor = (0.0, 0.0, 0.0, 1.0),
    ):
        """Construct boxes from centers, local half-lengths and xyzw quaternions.

        Args:
            center_W: Box centers in W, shape (3,) or (N, 3).
            half_extents: Nonnegative lengths along box-local axes, shape (3,) or (N, 3).
            rotation_W_B_xyzw: Rotations q_W_B mapping box-local axes into W, shape (4,) or (N, 4).
                Norms within 1e-3 of one are normalized to allow rounded input values.
        """
        super().__init__()
        device = _get_device_from_maybe_tensors(center_W, half_extents, rotation_W_B_xyzw)
        center_W = _batched_tensor(center_W, 3, device)
        half_extents = _batched_tensor(half_extents, 3, device)
        q_W_B = _unit_quaternion(_batched_tensor(rotation_W_B_xyzw, 4, device))
        assert torch.isfinite(center_W).all() and torch.isfinite(half_extents).all(), "Box bounds must be finite."
        assert (half_extents >= 0).all(), "Box half-extents must be nonnegative."
        center_W, half_extents, q_W_B = _broadcast_rows(center_W, half_extents, q_W_B)
        self._center_W = center_W.clone()
        self._half_extents = half_extents.clone()
        self._rotation_W_B_xyzw = q_W_B.clone()

    @property
    def center_W(self) -> torch.Tensor:
        """Centers in W, shape (N, 3)."""
        return self._center_W.clone()

    @property
    def half_extents(self) -> torch.Tensor:
        """Nonnegative half-lengths along box-local axes, shape (N, 3)."""
        return self._half_extents.clone()

    @property
    def rotation_W_B_xyzw(self) -> torch.Tensor:
        """Unit quaternions q_W_B mapping box-local axes into W, shape (N, 4)."""
        return self._rotation_W_B_xyzw.clone()

    @property
    def num_envs(self) -> int:
        """Number of box rows N."""
        return self._center_W.shape[0]

    def __getitem__(self, idx: int) -> "OrientedBoundingBox":
        """Select one row as a copy, preserving its leading dimension."""
        assert 0 <= idx < self.num_envs, f"Index {idx} out of range for {self.num_envs} boxes."
        return OrientedBoundingBox(self._center_W[idx], self._half_extents[idx], self._rotation_W_B_xyzw[idx])

    def to(self, device: torch.device | str) -> "OrientedBoundingBox":
        """Return boxes with their tensors on device."""
        return OrientedBoundingBox(
            self._center_W.to(device), self._half_extents.to(device), self._rotation_W_B_xyzw.to(device)
        )

    def translated(self, offset: tuple[float, float, float] | torch.Tensor) -> "OrientedBoundingBox":
        """Return boxes translated by offset (3,) or (N, 3) in W."""
        center_W, half_extents, q_W_B = self._get_finite_geometry()
        offset_W = _batched_tensor(offset, 3, center_W.device)
        center_W, half_extents, q_W_B, offset_W = _broadcast_rows(center_W, half_extents, q_W_B, offset_W)
        return OrientedBoundingBox(center_W + offset_W, half_extents, q_W_B)

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Return corners in W (N, 8, 3), optionally translated by pos (3,) or (N, 3).

        The first four corners lie on the box's local -Z face and the last four on its
        +Z face, each in counterclockwise order about local +Z. The first four need not
        be lowest along W's Z axis.
        """
        from isaaclab.utils.math import matrix_from_quat

        center_W, half_extents, q_W_B = self._get_finite_geometry()
        if pos is not None:
            offset_W = _batched_tensor(pos, 3, center_W.device)
            center_W, half_extents, q_W_B, offset_W = _broadcast_rows(center_W, half_extents, q_W_B, offset_W)
            center_W = center_W + offset_W
        signs = center_W.new_tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ])
        corners_B = signs * half_extents.unsqueeze(1)
        R_W_B = matrix_from_quat(q_W_B)
        corners_W = corners_B @ R_W_B.transpose(1, 2) + center_W.unsqueeze(1)
        return corners_W

    def get_axis_aligned_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return enclosing min/max points in W, each with shape (N, 3)."""
        from isaaclab.utils.math import matrix_from_quat

        center_W, half_extents, q_W_B = self._get_finite_geometry()
        R_W_B = matrix_from_quat(q_W_B)
        radius_W = (R_W_B.abs() @ half_extents.unsqueeze(-1)).squeeze(-1)
        return center_W - radius_W, center_W + radius_W

    def _get_finite_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the stored geometry, which construction already asserted is finite."""
        return self._center_W, self._half_extents, self._rotation_W_B_xyzw


class AxisAlignedBoundingBox(BoundingBox):
    """Axis-aligned bounding box storing local extents. Use get_corners_at(pos) for world-space corners.

    Stores min/max extents as (N, 3) float32 tensors where N is the number of environments.
    All properties consistently return tensors: (N, 3) for point values, (N,) for scalars.
    Constructor accepts tuples, 1D tensors, or (N, 3) tensors.

    Min/max properties expose mutable storage. Empty and unbounded intervals are supported for range
    operations and AABB-pair overlaps(); other BoundingBox queries require finite, nonempty bounds.
    """

    def __init__(
        self,
        min_point: tuple[float, float, float] | torch.Tensor,
        max_point: tuple[float, float, float] | torch.Tensor,
    ):
        super().__init__()
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
    def _to_batched_tensor(value: tuple[float, float, float] | torch.Tensor) -> torch.Tensor:
        """Convert tuple, 1-D tensor, or (N, 3) tensor to (N, 3) float32 tensor."""
        if isinstance(value, tuple):
            return torch.tensor([value], dtype=torch.float32)
        if value.dim() == 1:
            return value.unsqueeze(0).float()
        return value.float()

    @property
    def min_point(self) -> torch.Tensor:
        """Minimum coordinates in the containing frame, shape (N, 3)."""
        return self._min_point

    @property
    def max_point(self) -> torch.Tensor:
        """Maximum coordinates in the containing frame, shape (N, 3)."""
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

    def _get_finite_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derive geometry from the current bounds, asserting they are finite and nonempty."""
        min_point, max_point = self._get_finite_bounds()
        center_W = (min_point + max_point) * 0.5
        half_extents = (max_point - min_point) * 0.5
        q_W_B = center_W.new_tensor((0.0, 0.0, 0.0, 1.0)).repeat(self.num_envs, 1)
        return center_W, half_extents, q_W_B

    def _get_finite_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current min/max points (N, 3), asserting they are finite and nonempty."""
        assert (
            torch.isfinite(self._min_point).all() and torch.isfinite(self._max_point).all()
        ), "Box bounds must be finite."
        assert (self._min_point <= self._max_point).all(), "Box bounds must be nonempty."
        return self._min_point, self._max_point

    def _get_paired_bounds(self, other: "AxisAlignedBoundingBox") -> tuple[torch.Tensor, ...]:
        """Return both boxes' finite min/max points on a shared device, broadcast to shared rows."""
        min_a, max_a = self._get_finite_bounds()
        min_b, max_b = other._get_finite_bounds()
        assert min_a.device == min_b.device, "Bounding boxes must share a device."
        return _broadcast_rows(min_a, max_a, min_b, max_b)

    def get_axis_aligned_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy min/max points (N, 3), preserving empty and unbounded intervals."""
        return self._min_point.clone(), self._max_point.clone()

    def to_obb(self) -> OrientedBoundingBox:
        """Convert finite, nonempty bounds to identity-oriented boxes in the same frame."""
        center_W, half_extents, _ = self._get_finite_geometry()
        return OrientedBoundingBox(center_W, half_extents)

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

    def to(self, device: torch.device | str) -> "AxisAlignedBoundingBox":
        """Return a new bounding box with tensors on *device*."""
        return AxisAlignedBoundingBox(min_point=self._min_point.to(device), max_point=self._max_point.to(device))

    def translated(self, offset: tuple[float, float, float] | torch.Tensor) -> "AxisAlignedBoundingBox":
        """Return a new bounding box translated by an offset.

        Args:
            offset: Translation offset (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with translated position.
        """
        offset = _batched_tensor(offset, 3, self._min_point.device)
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

    def intersected(self, other: "AxisAlignedBoundingBox") -> "AxisAlignedBoundingBox":
        """Return the box common to this one and other.

        Rows and axes where the two do not overlap come back with ``min_point > max_point``.
        Check ``is_empty`` before using the resulting AABB.
        """
        return AxisAlignedBoundingBox(
            min_point=torch.maximum(self._min_point, other._min_point),
            max_point=torch.minimum(self._max_point, other._max_point),
        )

    @property
    def is_empty(self) -> torch.Tensor:
        """Per-env flag for boxes with min > max on any axis."""
        return (self._min_point > self._max_point).any(dim=-1)

    def volume_fraction_within(self, other: "AxisAlignedBoundingBox") -> torch.Tensor:
        """Return the fraction of this box's volume inside another box.

        Args:
            other: Destination bounds in the same frame and on the same device.
                Batch sizes must match, or one must be one.

        Returns:
            Fractions in [0, 1], one per broadcast box pair. Zero-volume boxes
            return zero because they have no contained volume.
        """
        assert self.num_envs == other.num_envs or self.num_envs == 1 or other.num_envs == 1
        intersection_min = torch.maximum(self.min_point, other.min_point)
        intersection_max = torch.minimum(self.max_point, other.max_point)
        intersection_volume = (intersection_max - intersection_min).clamp_min(0).prod(dim=-1)
        volume = self.size.clamp_min(0).prod(dim=-1)
        safe_volume = torch.where(volume > 0, volume, torch.ones_like(volume))
        return (intersection_volume / safe_volume).clamp(0, 1)

    def overlaps(self, other: BoundingBox, margin: float = 0.0) -> torch.Tensor:
        """Check if this AABB overlaps other boxes in 3D, using the separating-axis test when other is oriented.

        Args:
            other: The other bounding box to test against, in the same frame and on the same device.
            margin: Minimum required separation in meters. A positive value
                rejects placements where the gap is smaller than margin. Against an oriented box,
                margin must be nonnegative and float32 roundoff can flip touching verdicts.

        Returns:
            Bool tensor of shape (N,). True where volumes overlap (or are closer than margin).
        """
        if not isinstance(other, AxisAlignedBoundingBox):
            return super().overlaps(other, margin)
        return (
            (self._max_point[:, 0] + margin > other._min_point[:, 0])
            & (other._max_point[:, 0] + margin > self._min_point[:, 0])
            & (self._max_point[:, 1] + margin > other._min_point[:, 1])
            & (other._max_point[:, 1] + margin > self._min_point[:, 1])
            & (self._max_point[:, 2] + margin > other._min_point[:, 2])
            & (other._max_point[:, 2] + margin > self._min_point[:, 2])
        )

    def overlap_volume_upper_bound(self, other: BoundingBox, margin: float = 0.0) -> torch.Tensor:
        """Return the overlap volume upper bound, read from min/max bounds when other is also axis-aligned.

        Reading the bounds directly keeps touching AABB pairs at exactly zero.

        Args:
            other: Finite, nonempty boxes in the same frame and device; rows pair N/N or 1/N.
            margin: Finite nonnegative length added to both sides of other's bounds.
        """
        if not isinstance(other, AxisAlignedBoundingBox):
            return super().overlap_volume_upper_bound(other, margin)
        _assert_valid_margin(margin)
        min_a, max_a, min_b, max_b = self._get_paired_bounds(other)
        upper = torch.minimum(max_a, max_b + margin)
        lower = torch.maximum(min_a, min_b - margin)
        return torch.relu(upper - lower).prod(dim=-1)

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


class CachedOverlapVolumeUpperBound:
    """Overlap-volume upper bound for box pairs whose half-extents and rotations stay fixed while centers move.

    For each box, the bound multiplies the interval overlaps along that box's 3 axes and keeps the smaller
    of the two products. Projections that depend only on half-extents and rotations are cached at construction.
    """

    def __init__(
        self,
        half_extents_a: torch.Tensor,
        R_W_A: torch.Tensor,
        half_extents_b: torch.Tensor,
        R_W_B: torch.Tensor,
    ):
        """Cache projections for boxes A and B, with broadcast leading dimensions.

        Args:
            half_extents_a: Box A local half-lengths, shape (..., 3).
            R_W_A: Rotations mapping box A's local frame A into W, shape (..., 3, 3).
            half_extents_b: Box B local half-lengths, shape (..., 3).
            R_W_B: Rotations mapping box B's local frame B into W, shape (..., 3, 3).
        """
        self._half_extents_a = half_extents_a
        self._half_extents_b = half_extents_b
        self._R_A_W = R_W_A.transpose(-1, -2)
        R_A_B = self._R_A_W @ R_W_B
        self._R_B_A = R_A_B.transpose(-1, -2)
        abs_R_A_B = R_A_B.abs()
        # Projected radii of each box along the other box's face axes.
        self._radius_b_A = (abs_R_A_B * half_extents_b.unsqueeze(-2)).sum(dim=-1)
        self._radius_a_B = (abs_R_A_B * half_extents_a.unsqueeze(-1)).sum(dim=-2)

    def compute(self, center_a_W: torch.Tensor, center_b_W: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """Return upper bounds in cubed length units with the broadcast leading shape, differentiable in the centers.

        Args:
            center_a_W: Box A centers in W, shape (..., 3).
            center_b_W: Box B centers in W, shape (..., 3).
            margin: Finite nonnegative length added to box B's projected radius on every face axis.
        """
        return torch.minimum(*self.compute_face_axis_volumes(center_a_W, center_b_W, margin))

    def compute_face_axis_volumes(
        self, center_a_W: torch.Tensor, center_b_W: torch.Tensor, margin: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the interval-overlap volumes along box A's and along box B's face axes.

        Each volume is an upper bound on the overlap volume, in cubed length units with the broadcast leading shape.

        Args:
            center_a_W: Box A centers in W, shape (..., 3).
            center_b_W: Box B centers in W, shape (..., 3).
            margin: Finite nonnegative length added to box B's projected radius on every face axis.
        """
        _assert_valid_margin(margin)
        # Offset from A's center to B's center, expressed in A and then in B.
        offset_W = center_b_W - center_a_W
        offset_A = (self._R_A_W @ offset_W.unsqueeze(-1)).squeeze(-1)
        offset_B = (self._R_B_A @ offset_A.unsqueeze(-1)).squeeze(-1)
        volume_A = _interval_overlap_volume(self._half_extents_a, self._radius_b_A, offset_A, margin)
        volume_B = _interval_overlap_volume(self._radius_a_B, self._half_extents_b, offset_B, margin)
        return volume_A, volume_B


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


def _get_device_from_maybe_tensors(*values: object) -> torch.device | None:
    """Return the device shared by the tensor values, or None when none is a tensor."""
    devices = {value.device for value in values if isinstance(value, torch.Tensor)}
    assert len(devices) <= 1, "Geometry tensors must share a device."
    return devices.pop() if devices else None


def _assert_valid_margin(margin: float) -> None:
    """Assert that a clearance margin is finite and nonnegative."""
    assert math.isfinite(margin) and margin >= 0, "Margin must be finite and nonnegative."


def _batched_tensor(
    value: tuple[float, ...] | torch.Tensor, width: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return float32 rows of width components on the requested device."""
    if isinstance(value, torch.Tensor):
        assert device is None or value.device == device, "Geometry tensors must share a device."
        tensor = value.to(dtype=torch.float32)
    else:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
    assert tensor.ndim in (1, 2) and tensor.shape[-1] == width, f"Expected ({width},) or (N, {width})."
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


def _broadcast_rows(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Expand singleton rows to a shared batch size."""
    count = max(value.shape[0] for value in values)
    assert count > 0 and all(
        value.shape[0] in (1, count) for value in values
    ), "Geometry batches must have equal row counts or a single row."
    return tuple(value.expand(count, -1) for value in values)


def _unit_quaternion(value: torch.Tensor) -> torch.Tensor:
    """Normalize finite xyzw rows whose norms differ from one by at most 1e-3."""
    assert torch.isfinite(value).all(), "Box rotations must be finite."
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-3, rtol=0), "Box rotations must be unit quaternions."
    return value / norm


def _interval_overlap_volume(
    radius_a: torch.Tensor, radius_b: torch.Tensor, offset: torch.Tensor, margin: float
) -> torch.Tensor:
    """Multiply interval intersections on three axes, inflating the second interval by margin.

    Args:
        radius_a: First intervals' half-lengths, centered at zero, shape (..., 3).
        radius_b: Second intervals' half-lengths, shape (..., 3).
        offset: Second intervals' centers relative to the first, shape (..., 3).
        margin: Length added to both ends of the second intervals.
    """
    upper = torch.minimum(radius_a, offset + radius_b + margin)
    lower = torch.maximum(-radius_a, offset - radius_b - margin)
    return torch.relu(upper - lower).prod(dim=-1)
