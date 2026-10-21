# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched oriented and axis-aligned bounding-box geometry."""

import math
import torch

from isaaclab_arena.utils.pose import Pose

Vector3 = tuple[float, float, float] | torch.Tensor
Quaternion = tuple[float, float, float, float] | torch.Tensor


def _batched_tensor(value: Vector3 | Quaternion, width: int, device: torch.device | None = None) -> torch.Tensor:
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


def _axis_aligned_order(rotation: torch.Tensor) -> torch.Tensor | None:
    """Return local-axis indices for an entirely axis-aligned batch of fixed rotations."""
    if rotation.requires_grad:
        return None
    magnitudes = rotation.abs()
    nonzero_count = (magnitudes != 0).sum(dim=-1)
    equal_components = ((magnitudes == 0) | (magnitudes == magnitudes.amax(dim=-1, keepdim=True))).all(dim=-1)
    # These quaternion patterns are exactly the 24 axis-aligned rotations; no angular tolerance is used.
    aligned = equal_components & ((nonzero_count == 1) | (nonzero_count == 2) | (nonzero_count == 4))
    if not aligned.all():
        return None
    from isaaclab.utils.math import matrix_from_quat

    return matrix_from_quat(rotation).abs().argmax(dim=-1)


class OrientedBoundingBox:
    """Oriented box geometry with float32 tensors, N rows and singleton broadcasting.

    Centers are expressed in a common containing frame; each quaternion maps box-local
    axes into that frame. Direct construction copies finite geometry, and its tensor
    properties return copies. AxisAlignedBoundingBox additionally supports mutable,
    empty or unbounded intervals. transformed(), get_bounds_along_axis() and SAT
    queries require finite, nonempty geometry from either class. Fixed axis-aligned
    batches use three-axis queries; other batches use the full separating-axis test.
    """

    def __init__(self, center: Vector3, half_extents: Vector3, rotation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)):
        """Construct boxes from centers, local half-lengths and xyzw quaternions.

        Args:
            center: Box centers, shape (3,) or (N, 3).
            half_extents: Nonnegative lengths along box-local axes, shape (3,) or (N, 3).
            rotation_xyzw: Box-local to containing-frame rotations, shape (4,) or (N, 4).
                Norms within 1e-3 of one are normalized to allow rounded input values.
        """
        device = None
        for value in (center, half_extents, rotation_xyzw):
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        center = _batched_tensor(center, 3, device)
        half_extents = _batched_tensor(half_extents, 3, device)
        rotation = _unit_quaternion(_batched_tensor(rotation_xyzw, 4, device))
        assert torch.isfinite(center).all() and torch.isfinite(half_extents).all(), "Box bounds must be finite."
        assert (half_extents >= 0).all(), "Box half-extents must be nonnegative."
        center, half_extents, rotation = _broadcast_rows(center, half_extents, rotation)
        self._center = center.clone()
        self._half_extents = half_extents.clone()
        self._rotation_xyzw = rotation.clone()
        self._axis_order = _axis_aligned_order(rotation)
        """Local-axis indices for containing-frame XYZ, shape (N, 3), or None when full SAT is required."""

    def _get_finite_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return finite centers, nonnegative half-extents and unit xyzw rotations."""
        return self._center, self._half_extents, self._rotation_xyzw

    def _get_axis_aligned_half_extents(self, half_extents: torch.Tensor) -> torch.Tensor | None:
        """Map local half-lengths to containing-frame axes when every rotation is axis-aligned."""
        if self._axis_order is None:
            return None
        return half_extents.gather(-1, self._axis_order)

    @property
    def center(self) -> torch.Tensor:
        """Centers in the containing frame, shape (N, 3)."""
        return self._center.clone()

    @property
    def half_extents(self) -> torch.Tensor:
        """Half-lengths along box-local axes, shape (N, 3); nonnegative for nonempty boxes."""
        return self._half_extents.clone()

    @property
    def rotation_xyzw(self) -> torch.Tensor:
        """Unit quaternions mapping box-local axes into the containing frame, shape (N, 4)."""
        return self._rotation_xyzw.clone()

    @property
    def num_envs(self) -> int:
        """Number of box rows N."""
        return self._center.shape[0]

    def __getitem__(self, idx: int) -> "OrientedBoundingBox":
        """Select one row, preserving its leading dimension."""
        assert 0 <= idx < self.num_envs, f"Index {idx} out of range for {self.num_envs} boxes."
        return OrientedBoundingBox(self._center[idx], self._half_extents[idx], self._rotation_xyzw[idx])

    def to(self, device: torch.device | str) -> "OrientedBoundingBox":
        """Return boxes with their tensors on device."""
        return OrientedBoundingBox(
            self._center.to(device), self._half_extents.to(device), self._rotation_xyzw.to(device)
        )

    def translated(self, offset: Vector3) -> "OrientedBoundingBox":
        """Return boxes translated by offset (3,) or (N, 3) in their containing frame."""
        center, half_extents, rotation = self._get_finite_geometry()
        offset = _batched_tensor(offset, 3, center.device)
        center, half_extents, rotation, offset = _broadcast_rows(center, half_extents, rotation, offset)
        return OrientedBoundingBox(center + offset, half_extents, rotation)

    def transformed(self, t_A_B: Vector3, q_A_B: Quaternion) -> "OrientedBoundingBox":
        """Map boxes from frame B into frame A, preserving their oriented shape.

        Args:
            t_A_B: Translation of frame B's origin in A, shape (3,) or (N, 3).
            q_A_B: Unit xyzw rotation mapping B into A, shape (4,) or (N, 4).

        Returns:
            OrientedBoundingBox in frame A.
        """
        from isaaclab.utils.math import quat_apply, quat_mul

        center, half_extents, rotation = self._get_finite_geometry()
        translation = _batched_tensor(t_A_B, 3, center.device)
        quaternion = _unit_quaternion(_batched_tensor(q_A_B, 4, center.device))
        center, half_extents, rotation, translation, quaternion = _broadcast_rows(
            center, half_extents, rotation, translation, quaternion
        )
        return OrientedBoundingBox(
            quat_apply(quaternion, center) + translation, half_extents, quat_mul(quaternion, rotation)
        )

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Return corners (N, 8, 3), optionally translated by pos (3,) or (N, 3).

        Corner ordering follows the box-local axes, matching AxisAlignedBoundingBox
        for identity rotation; the first four need not be lowest along the containing frame's Z axis.
        """
        from isaaclab.utils.math import matrix_from_quat

        center, half_extents, rotation = self._get_finite_geometry()
        if pos is not None:
            offset = _batched_tensor(pos, 3, center.device)
            center, half_extents, rotation, offset = _broadcast_rows(center, half_extents, rotation, offset)
            center = center + offset
        signs = center.new_tensor([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ])
        local_corners = signs * half_extents.unsqueeze(1)
        return local_corners @ matrix_from_quat(rotation).transpose(1, 2) + center.unsqueeze(1)

    def get_bounds_along_axis(self, axis: Vector3) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper projections (N,) onto axis, normalized internally.

        Args:
            axis: Nonzero direction in the containing frame, shape (3,) or (N, 3).
        """
        from isaaclab.utils.math import matrix_from_quat

        center, half_extents, rotation = self._get_finite_geometry()
        axis = _batched_tensor(axis, 3, center.device)
        center, half_extents, rotation, axis = _broadcast_rows(center, half_extents, rotation, axis)
        norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        assert (
            torch.isfinite(axis).all() and torch.isfinite(norm).all() and (norm > 0).all()
        ), "Projection axes must have finite, nonzero norms."
        axis = axis / norm
        box_axes = matrix_from_quat(rotation).transpose(1, 2)
        radius = (half_extents * torch.einsum("nij,nj->ni", box_axes, axis).abs()).sum(dim=-1)
        midpoint = (center * axis).sum(dim=-1)
        return midpoint - radius, midpoint + radius

    def get_axis_aligned_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return enclosing containing-frame min/max points, each with shape (N, 3)."""
        from isaaclab.utils.math import matrix_from_quat

        center, half_extents, rotation = self._get_finite_geometry()
        radius = (matrix_from_quat(rotation).abs() @ half_extents.unsqueeze(-1)).squeeze(-1)
        return center - radius, center + radius

    def to_aabb(self) -> "AxisAlignedBoundingBox":
        """Return the enclosing AxisAlignedBoundingBox in the same frame."""
        return AxisAlignedBoundingBox(*self.get_axis_aligned_bounds())

    def _sat_depths(self, other: "OrientedBoundingBox", margin: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected depths and axis norms (N, K), using K=3 for aligned pairs, else 15."""
        from isaaclab.utils.math import matrix_from_quat

        assert math.isfinite(margin) and margin >= 0, "Margin must be finite and nonnegative."
        center_a, extent_a, rotation_a = self._get_finite_geometry()
        center_b, extent_b, rotation_b = other._get_finite_geometry()
        assert center_a.device == center_b.device, "Bounding boxes must share a device."
        center_a, extent_a, rotation_a, center_b, extent_b, rotation_b = _broadcast_rows(
            center_a, extent_a, rotation_a, center_b, extent_b, rotation_b
        )
        aligned_extent_a = self._get_axis_aligned_half_extents(extent_a)
        aligned_extent_b = other._get_axis_aligned_half_extents(extent_b)
        if aligned_extent_a is not None and aligned_extent_b is not None:
            depths = aligned_extent_a + aligned_extent_b + margin - (center_b - center_a).abs()
            return depths, torch.ones_like(depths)
        axes_a = matrix_from_quat(rotation_a).transpose(1, 2)
        axes_b = matrix_from_quat(rotation_b).transpose(1, 2)
        cross_axes = torch.linalg.cross(axes_a.unsqueeze(2), axes_b.unsqueeze(1), dim=-1).flatten(1, 2)
        axes = torch.cat((axes_a, axes_b, cross_axes), dim=1)
        norms = torch.linalg.vector_norm(axes, dim=-1)
        radius_a = (torch.einsum("nkd,njd->nkj", axes, axes_a).abs() * extent_a.unsqueeze(1)).sum(dim=-1)
        radius_b = (torch.einsum("nkd,njd->nkj", axes, axes_b).abs() * extent_b.unsqueeze(1)).sum(dim=-1)
        distance = torch.einsum("nkd,nd->nk", axes, center_b - center_a).abs()
        return radius_a + radius_b + margin * norms - distance, norms

    def overlaps(self, other: "OrientedBoundingBox", margin: float = 0.0) -> torch.Tensor:
        """Return strict SAT overlap or projected-clearance violation, shape (N,).

        At zero margin, touching on a separating axis is not overlap. Exactly parallel
        edge pairs contribute no axis; small nonzero cross axes remain in the test.
        Float32 roundoff can change the verdict at touching or margin boundaries,
        including after converting min/max AABB bounds to center and half-extents.

        Args:
            other: Finite boxes in the same frame and device; rows pair N/N or 1/N.
            margin: Nonnegative required gap along SAT axes, in input length units;
                not Euclidean clearance.
        """
        depths, norms = self._sat_depths(other, margin)
        return ((depths > 0) | (norms == 0)).all(dim=-1)

    def penetration_depth(self, other: "OrientedBoundingBox", margin: float = 0.0) -> torch.Tensor:
        """Return minimum normalized SAT depth clamped at zero, shape (N,).

        At zero margin this measures translation to separate overlapping boxes at
        fixed orientation. A positive margin measures projected-clearance violation,
        not Euclidean distance. Neither is intersection volume. Gradients are
        piecewise-defined, with no special escape direction for concentric boxes.

        Args:
            other: Finite boxes in the same frame and device; rows pair N/N or 1/N.
            margin: Nonnegative projected-axis clearance added before clamping,
                in input length units.
        """
        depths, norms = self._sat_depths(other, margin)
        valid = norms > 0
        normalized = depths / torch.where(valid, norms, torch.ones_like(norms))
        normalized = torch.where(valid, normalized, torch.full_like(normalized, torch.inf))
        return normalized.amin(dim=-1).clamp_min(0)


class AxisAlignedBoundingBox(OrientedBoundingBox):
    """Axis-aligned bounds in a caller-chosen containing frame.

    Stores min/max coordinates as (N, 3) float32 tensors where N is the number of environments.
    Point properties have shape (N, 3), scalar tensors (N,), and rotation_xyzw (N, 4).
    Constructor accepts tuples, 1D tensors, or (N, 3) tensors. Min/max properties
    expose mutable storage. Empty and unbounded intervals remain supported for range
    operations; inherited transforms, projections and SAT require finite, nonempty bounds.
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

    @property
    def half_extents(self) -> torch.Tensor:
        """Half of each interval's signed length, shape (N, 3); negative for empty axes."""
        return self.size * 0.5

    @property
    def rotation_xyzw(self) -> torch.Tensor:
        """Identity box-local to containing-frame rotations, shape (N, 4)."""
        return self._min_point.new_tensor((0.0, 0.0, 0.0, 1.0)).repeat(self.num_envs, 1)

    def _get_finite_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derive finite geometry from the current mutable bounds."""
        center, half_extents = self.center, self.half_extents
        assert torch.isfinite(center).all() and torch.isfinite(half_extents).all(), "Box bounds must be finite."
        assert (half_extents >= 0).all(), "Box bounds must be nonempty."
        return center, half_extents, self.rotation_xyzw

    def _get_axis_aligned_half_extents(self, half_extents: torch.Tensor) -> torch.Tensor:
        """Keep the half-lengths derived from current axis-aligned bounds."""
        return half_extents

    def get_axis_aligned_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy min/max points (N, 3), preserving empty and unbounded intervals."""
        return self._min_point.clone(), self._max_point.clone()

    def to_obb(self) -> OrientedBoundingBox:
        """Copy finite, nonempty bounds to identity-oriented boxes in the same frame."""
        return OrientedBoundingBox(self.center, self.half_extents)

    @property
    def top_surface_z(self) -> torch.Tensor:
        """Returns the z-coordinate of the top surface. Shape (N,)."""
        return self._max_point[:, 2]

    @property
    def bottom_surface_z(self) -> torch.Tensor:
        """Returns the z-coordinate of the bottom surface. Shape (N,)."""
        return self._min_point[:, 2]

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Return eight corners, optionally offset by pos.

        Args:
            pos: Offset added to each corner, shape (3,) or (N, 3), such as the bounds'
                origin in a parent frame with parallel axes. None leaves coordinates unchanged.

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

    def overlaps(self, other: OrientedBoundingBox, margin: float = 0.0) -> torch.Tensor:
        """Check overlap in 3D, using SAT when the other box is oriented.

        Mixed AABB/OBB queries require both boxes to have finite, nonempty bounds.

        Args:
            other: Box in the same frame and on the same device.
            margin: Required separation along the tested axes, in input length units.
                Mixed AABB/OBB queries use nonnegative SAT clearance, not Euclidean distance,
                and have float32 uncertainty at touching or margin boundaries.

        Returns:
            Bool tensor of shape (N,) for strict projected overlap or clearance violation.
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
