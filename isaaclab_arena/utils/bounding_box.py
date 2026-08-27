# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Oriented bounding-box geometry utilities."""

import torch
from dataclasses import dataclass

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul

from isaaclab_arena.utils.pose import Pose

Vector3 = tuple[float, float, float] | torch.Tensor
Quaternion = tuple[float, float, float, float] | torch.Tensor


def _as_batched(value: tuple[float, ...] | torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    assert tensor.ndim in (1, 2), f"Expected a ({width},) or (N, {width}) value, got {tuple(tensor.shape)}."
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] == width, f"Expected trailing dimension {width}, got {tuple(tensor.shape)}."
    assert tensor.shape[0] > 0, "Bounding-box batches must not be empty."
    return tensor


def _common_device(*values: tuple[float, ...] | torch.Tensor) -> torch.device:
    devices = [value.device for value in values if not isinstance(value, tuple)]
    if not devices:
        return torch.device("cpu")
    assert all(device == devices[0] for device in devices), "Bounding-box inputs must be on the same device."
    return devices[0]


def _broadcast_rows(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    count = max(value.shape[0] for value in values)
    assert all(
        value.shape[0] in (1, count) for value in values
    ), f"Batched values must have equal leading dimensions or N=1; got {[value.shape[0] for value in values]}."
    return tuple(value.expand(count, *value.shape[1:]) if value.shape[0] == 1 else value for value in values)


@dataclass(init=False, slots=True, eq=False)
class OrientedBoundingBox:
    """A batch of oriented boxes, where N is the number of boxes."""

    center: torch.Tensor
    """Box centers. Shape (N, 3)."""

    half_extents: torch.Tensor
    """Non-negative half-lengths along each local box axis. Shape (N, 3)."""

    rotation_xyzw: torch.Tensor
    """Local-to-parent unit quaternions in xyzw order. Shape (N, 4)."""

    def __init__(self, center: Vector3, half_extents: Vector3, rotation_xyzw: Quaternion):
        device = _common_device(center, half_extents, rotation_xyzw)
        center_tensor = _as_batched(center, 3, device)
        half_extents_tensor = _as_batched(half_extents, 3, device)
        rotation_tensor = _as_batched(rotation_xyzw, 4, device)
        center_tensor, half_extents_tensor, rotation_tensor = _broadcast_rows(
            center_tensor, half_extents_tensor, rotation_tensor
        )
        center_tensor = center_tensor.clone()
        half_extents_tensor = half_extents_tensor.clone()
        rotation_tensor = rotation_tensor.clone()

        assert torch.isfinite(center_tensor).all(), "Box centers must be finite."
        assert torch.isfinite(half_extents_tensor).all(), "Box half-extents must be finite."
        assert (half_extents_tensor >= 0.0).all(), "Box half-extents must be non-negative."
        assert torch.isfinite(rotation_tensor).all(), "Box rotations must be finite."
        norms = torch.linalg.vector_norm(rotation_tensor, dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5
        ), "Box rotations must be unit quaternions."

        self.center = center_tensor
        self.half_extents = half_extents_tensor
        self.rotation_xyzw = rotation_tensor

    @classmethod
    def from_tensors_unchecked(
        cls,
        center: torch.Tensor,
        half_extents: torch.Tensor,
        rotation_xyzw: torch.Tensor,
    ) -> "OrientedBoundingBox":
        """Build from validated tensors without cloning or synchronizing the device."""
        bbox = cls.__new__(cls)
        bbox.center = center
        bbox.half_extents = half_extents
        bbox.rotation_xyzw = rotation_xyzw
        return bbox

    @classmethod
    def from_min_max(cls, min_point: Vector3, max_point: Vector3) -> "OrientedBoundingBox":
        """Construct axis-aligned boxes from minimum and maximum points."""
        device = _common_device(min_point, max_point)
        minimum = _as_batched(min_point, 3, device)
        maximum = _as_batched(max_point, 3, device)
        minimum, maximum = _broadcast_rows(minimum, maximum)
        assert (maximum >= minimum).all(), "Maximum points must not be below minimum points."
        identity = minimum.new_zeros((1, 4))
        identity[:, 3] = 1.0
        identity = identity.expand(minimum.shape[0], 4)
        return cls.from_tensors_unchecked((minimum + maximum) * 0.5, (maximum - minimum) * 0.5, identity)

    def __getitem__(self, idx: int) -> "OrientedBoundingBox":
        """Select one box while preserving its leading dimension."""
        assert 0 <= idx < self.num_envs, f"Index {idx} out of range for {self.num_envs} boxes."
        return self.from_tensors_unchecked(
            self.center[idx : idx + 1],
            self.half_extents[idx : idx + 1],
            self.rotation_xyzw[idx : idx + 1],
        )

    @property
    def num_envs(self) -> int:
        """Return the leading dimension N."""
        return self.center.shape[0]

    def is_batch_invariant(self) -> bool:
        """Return whether every row describes the same box."""
        return all(
            torch.allclose(value, value[:1].expand_as(value))
            for value in (self.center, self.half_extents, self.rotation_xyzw)
        )

    def to(self, device: torch.device | str) -> "OrientedBoundingBox":
        """Return the boxes on the requested device."""
        return self.from_tensors_unchecked(
            self.center.to(device),
            self.half_extents.to(device),
            self.rotation_xyzw.to(device),
        )

    def translated(self, offset: Vector3) -> "OrientedBoundingBox":
        """Return boxes translated in their parent frame."""
        offset_tensor = _as_batched(offset, 3, self.center.device)
        center, half_extents, rotation, offset_tensor = _broadcast_rows(
            self.center, self.half_extents, self.rotation_xyzw, offset_tensor
        )
        return self.from_tensors_unchecked(center + offset_tensor, half_extents, rotation)

    def rotated_by_quat(self, rotation_xyzw: Quaternion) -> "OrientedBoundingBox":
        """Rotate boxes about the parent-frame origin."""
        rotation = _as_batched(rotation_xyzw, 4, self.center.device)
        norms = torch.linalg.vector_norm(rotation, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5), "Rotation must be unit length."
        return self.rotated_by_quat_unchecked(rotation)

    def rotated_by_quat_unchecked(self, rotation: torch.Tensor) -> "OrientedBoundingBox":
        """Rotate by validated batched quaternions without synchronizing the device."""
        rotation = _as_batched(rotation, 4, self.center.device)
        center, half_extents, box_rotation, rotation = _broadcast_rows(
            self.center, self.half_extents, self.rotation_xyzw, rotation
        )
        return self.from_tensors_unchecked(
            quat_apply(rotation, center),
            half_extents,
            quat_mul(rotation, box_rotation),
        )

    def transformed(self, position_xyz: Vector3, rotation_xyzw: Quaternion) -> "OrientedBoundingBox":
        """Apply a parent-frame rigid transform to the boxes."""
        position = _as_batched(position_xyz, 3, self.center.device)
        rotation = _as_batched(rotation_xyzw, 4, self.center.device)
        norms = torch.linalg.vector_norm(rotation, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5), "Rotation must be unit length."
        return self.transformed_unchecked(position, rotation)

    def transformed_unchecked(self, position: torch.Tensor, rotation: torch.Tensor) -> "OrientedBoundingBox":
        """Apply validated batched transforms without synchronizing the device."""
        position = _as_batched(position, 3, self.center.device)
        rotated = self.rotated_by_quat_unchecked(rotation)
        center, half_extents, box_rotation, position = _broadcast_rows(
            rotated.center, rotated.half_extents, rotated.rotation_xyzw, position
        )
        return self.from_tensors_unchecked(center + position, half_extents, box_rotation)

    def get_corners(self) -> torch.Tensor:
        """Return parent-frame corners with shape (N, 8, 3)."""
        signs = self.center.new_tensor([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ])
        local_corners = signs.unsqueeze(0) * self.half_extents.unsqueeze(1)
        rotation = self.rotation_xyzw.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)
        rotated = quat_apply(rotation, local_corners.reshape(-1, 3)).reshape(-1, 8, 3)
        return self.center.unsqueeze(1) + rotated

    def get_bounds_along_axis(self, axis: Vector3) -> tuple[torch.Tensor, torch.Tensor]:
        """Return minimum and maximum projections onto an axis."""
        axis_tensor = _as_batched(axis, 3, self.center.device)
        center, half_extents, rotation, axis_tensor = _broadcast_rows(
            self.center, self.half_extents, self.rotation_xyzw, axis_tensor
        )
        axis_norm = torch.linalg.vector_norm(axis_tensor, dim=-1, keepdim=True)
        assert (axis_norm > 0.0).all(), "Projection axes must be non-zero."
        axis_tensor = axis_tensor / axis_norm
        local_axes = matrix_from_quat(rotation).transpose(1, 2)
        radius = (half_extents * torch.abs(torch.einsum("nid,nd->ni", local_axes, axis_tensor))).sum(dim=-1)
        projected_center = (center * axis_tensor).sum(dim=-1)
        return projected_center - radius, projected_center + radius

    def get_axis_aligned_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return enclosing axis-aligned minimum and maximum points."""
        rotation = matrix_from_quat(self.rotation_xyzw)
        radius = torch.bmm(rotation.abs(), self.half_extents.unsqueeze(-1)).squeeze(-1)
        return self.center - radius, self.center + radius

    def is_axis_aligned(self, atol: float = 1e-5) -> torch.Tensor:
        """Return whether each orientation is axis-aligned, with shape (N,)."""
        absolute = matrix_from_quat(self.rotation_xyzw).abs()
        near_zero_or_one = (absolute <= atol) | ((absolute - 1.0).abs() <= atol)
        ones = torch.ones_like(absolute[..., 0])
        return (
            near_zero_or_one.all(dim=(-1, -2))
            & torch.isclose(absolute.sum(dim=-1), ones, atol=atol, rtol=0).all(dim=-1)
            & torch.isclose(absolute.sum(dim=-2), ones, atol=atol, rtol=0).all(dim=-1)
        )

    def _paired(self, other: "OrientedBoundingBox") -> tuple[torch.Tensor, ...]:
        assert self.center.device == other.center.device, "Bounding boxes must be on the same device."
        return _broadcast_rows(
            self.center,
            self.half_extents,
            self.rotation_xyzw,
            other.center,
            other.half_extents,
            other.rotation_xyzw,
        )

    def penetration(
        self,
        other: "OrientedBoundingBox",
        clearance_m: float = 0.0,
        tie_break_sign: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Return positive SAT penetration or clearance violation with shape (N,).

        Args:
            other: Boxes to test, broadcastable over N.
            clearance_m: Additional required separation in metres.
            tie_break_sign: Directed escape sign for exactly concentric projections.

        Rows with separated enclosing AABBs are broadphase-culled to zero.
        """
        assert clearance_m >= 0.0, "Clearance must be non-negative."
        center_a, extent_a, quat_a, center_b, extent_b, quat_b = self._paired(other)
        count = center_a.shape[0]
        if isinstance(tie_break_sign, float):
            assert tie_break_sign in (-1.0, 1.0), "Tie-break sign must be -1 or 1."
        tie_sign = torch.as_tensor(tie_break_sign, dtype=center_a.dtype, device=center_a.device).reshape(-1, 1)
        assert tie_sign.shape[0] in (1, count), f"Expected one tie-break sign or N={count}, got {tie_sign.shape[0]}."
        rotation_a = matrix_from_quat(quat_a)
        rotation_b = matrix_from_quat(quat_b)
        aabb_radius_a = torch.bmm(rotation_a.abs(), extent_a.unsqueeze(-1)).squeeze(-1)
        aabb_radius_b = torch.bmm(rotation_b.abs(), extent_b.unsqueeze(-1)).squeeze(-1)
        min_a, max_a = center_a - aabb_radius_a, center_a + aabb_radius_a
        min_b, max_b = center_b - aabb_radius_b, center_b + aabb_radius_b
        broadphase = ((max_a + clearance_m > min_b) & (max_b + clearance_m > min_a)).all(dim=-1)

        axes_a = rotation_a.transpose(1, 2)
        axes_b = rotation_b.transpose(1, 2)
        cross_axes = torch.linalg.cross(axes_a.unsqueeze(2), axes_b.unsqueeze(1), dim=-1).reshape(count, 9, 3)
        axes = torch.cat([axes_a, axes_b, cross_axes], dim=1)
        axis_norm = torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
        valid = axis_norm.squeeze(-1) > 1e-6
        axes = axes / axis_norm.clamp_min(1e-6)
        dominant_index = torch.abs(axes).argmax(dim=-1, keepdim=True)
        dominant_component = torch.gather(axes, dim=-1, index=dominant_index)
        canonical_sign = torch.where(dominant_component < 0.0, -1.0, 1.0)
        axes = axes * canonical_sign

        radius_a = (torch.abs(torch.einsum("nkd,njd->nkj", axes, axes_a)) * extent_a.unsqueeze(1)).sum(dim=-1)
        radius_b = (torch.abs(torch.einsum("nkd,njd->nkj", axes, axes_b)) * extent_b.unsqueeze(1)).sum(dim=-1)
        projection = torch.einsum("nkd,nd->nk", axes, center_b - center_a)
        # abs() has zero derivative at zero. Preserve its value while selecting a directed
        # derivative so exactly concentric movable boxes receive an escape gradient.
        signed_zero_projection = tie_sign * projection
        distance = torch.where(projection == 0.0, signed_zero_projection, torch.abs(projection))
        depths = radius_a + radius_b + clearance_m - distance
        depths = torch.where(valid, depths, torch.full_like(depths, torch.inf))
        penetration = depths.amin(dim=-1).clamp_min(0.0)
        return torch.where(broadphase, penetration, torch.zeros_like(penetration))

    def overlaps(self, other: "OrientedBoundingBox", clearance_m: float = 0.0) -> torch.Tensor:
        """Return whether boxes overlap or violate the requested clearance."""
        return self.penetration(other, clearance_m) > 0.0


def get_random_pose_within_bounding_box(bbox: OrientedBoundingBox, seed: int | None = None) -> Pose:
    """Sample a position uniformly within the first box and use identity rotation."""
    if seed is not None:
        torch.manual_seed(seed)
    local_position = (2.0 * torch.rand(3, device=bbox.center.device) - 1.0) * bbox.half_extents[0]
    position = bbox.center[0] + quat_apply(bbox.rotation_xyzw[:1], local_position.unsqueeze(0))[0]
    return Pose(position_xyz=tuple(position.cpu().tolist()), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
