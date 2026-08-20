# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab_arena.scene.object_state import _to_torch, get_env, object_state
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def object_geometry(env, name: str) -> ObjectGeometry:
    """Return runtime geometry queries for a scene object."""
    return ObjectGeometry(get_env(env), name)


class ObjectGeometry:
    """Runtime geometry view for rigid and deformable scene objects."""

    def __init__(self, env, name: str):
        self.env = env
        self.name = name
        self.state = object_state(env, name)
        self.entity = self.state.entity

    @property
    def data(self):
        return self.state.data

    @property
    def has_nodal_state(self) -> bool:
        """Whether this object exposes deformable nodal position and velocity tensors."""
        data = self.data
        return data is not None and hasattr(data, "nodal_pos_w") and hasattr(data, "nodal_vel_w")

    @property
    def has_meaningful_orientation(self) -> bool:
        """Whether this object has a physical orientation, not an identity centroid placeholder."""
        return self.state.capabilities.has_orientation

    def centroid_w(self) -> torch.Tensor:
        """Return the object's centroid/root position in world frame."""
        return self.state.position_w()

    def linear_velocity_w(self) -> torch.Tensor:
        """Return the object's root/centroid linear velocity in world frame."""
        return self.state.linear_velocity_w()

    def nodal_pos_w(self, required: bool = True) -> torch.Tensor:
        """Return deformable nodal positions, or raise when unavailable and required."""
        if self.has_nodal_state:
            return _to_torch(self.data.nodal_pos_w)
        if required:
            raise AttributeError(f"Scene object '{self.name}' does not expose nodal positions.")
        return self.centroid_w().unsqueeze(1)

    def nodal_vel_w(self, required: bool = True) -> torch.Tensor:
        """Return deformable nodal velocities, or raise when unavailable and required."""
        if self.has_nodal_state:
            return _to_torch(self.data.nodal_vel_w)
        if required:
            raise AttributeError(f"Scene object '{self.name}' does not expose nodal velocities.")
        return self.linear_velocity_w().unsqueeze(1)

    def key_points_w(self) -> torch.Tensor:
        """Return points that represent the object's occupied runtime volume."""
        if self.has_nodal_state:
            return self.nodal_pos_w()
        return self.aabb_w().get_corners_at()

    def aabb_w(self) -> AxisAlignedBoundingBox:
        """Return an axis-aligned world-space box for this object's current geometry."""
        if self.has_nodal_state:
            nodal_pos = self.nodal_pos_w()
            return AxisAlignedBoundingBox(
                min_point=nodal_pos.amin(dim=1),
                max_point=nodal_pos.amax(dim=1),
            )

        static_bbox = self._static_local_bbox()
        pos_w = self.centroid_w()
        if static_bbox is None:
            return AxisAlignedBoundingBox(min_point=pos_w, max_point=pos_w)
        return _transform_local_bbox(static_bbox.to(pos_w.device), pos_w, self.state.quat_w(required=False))

    def nearest_point_w(self, query_w: torch.Tensor) -> torch.Tensor:
        """Return the closest representative point to ``query_w`` for each env."""
        query_w = _as_env_points(query_w, self.env.num_envs, self.env.device)
        if self.has_nodal_state:
            nodal_pos = self.nodal_pos_w()
            distances = torch.linalg.vector_norm(nodal_pos - query_w.unsqueeze(1), dim=-1)
            node_ids = distances.argmin(dim=1)
            return nodal_pos[torch.arange(self.env.num_envs, device=self.env.device), node_ids]

        bbox = self.aabb_w()
        return torch.minimum(torch.maximum(query_w, bbox.min_point), bbox.max_point)

    def max_point_speed(self) -> torch.Tensor:
        """Return max representative point speed per env."""
        if self.has_nodal_state:
            return torch.linalg.vector_norm(self.nodal_vel_w(), dim=-1).amax(dim=1)
        return torch.linalg.vector_norm(self.linear_velocity_w(), dim=-1)

    def fraction_inside_aabb(self, target_bbox: AxisAlignedBoundingBox, margin: float = 0.0) -> torch.Tensor:
        """Return the fraction of this object's representative points inside ``target_bbox``."""
        points = self.key_points_w()
        inside = points_inside_aabb(points, target_bbox, margin=margin)
        return inside.float().mean(dim=1)

    def _static_local_bbox(self) -> AxisAlignedBoundingBox | None:
        arena_assets = getattr(self.env, "arena_scene_assets", None)
        if not arena_assets or self.name not in arena_assets:
            return None
        asset = arena_assets[self.name]
        if hasattr(asset, "get_bounding_box_per_env") and getattr(asset, "variant_indices_by_env", None) is not None:
            return asset.get_bounding_box_per_env(self.env.num_envs)
        if getattr(asset, "usd_path", True) is None:
            return None
        if not hasattr(asset, "get_bounding_box"):
            return None
        return asset.get_bounding_box()


def points_inside_aabb(points_w: torch.Tensor, bbox_w: AxisAlignedBoundingBox, margin: float = 0.0) -> torch.Tensor:
    """Return mask of points inside ``bbox_w`` for each env."""
    min_point = bbox_w.min_point.unsqueeze(1) - margin
    max_point = bbox_w.max_point.unsqueeze(1) + margin
    return torch.all((points_w >= min_point) & (points_w <= max_point), dim=-1)


def _as_env_points(query_w: torch.Tensor, num_envs: int, device) -> torch.Tensor:
    query_w = torch.as_tensor(query_w, dtype=torch.float32, device=device)
    if query_w.dim() == 1:
        query_w = query_w.unsqueeze(0).expand(num_envs, -1)
    assert query_w.shape == (num_envs, 3), f"Expected query points with shape ({num_envs}, 3), got {query_w.shape}."
    return query_w


def _transform_local_bbox(
    local_bbox: AxisAlignedBoundingBox,
    pos_w: torch.Tensor,
    quat_xyzw_w: torch.Tensor,
) -> AxisAlignedBoundingBox:
    corners = local_bbox.get_corners_at().to(pos_w.device)
    if corners.shape[0] == 1:
        corners = corners.expand(pos_w.shape[0], -1, -1)
    assert corners.shape[0] == pos_w.shape[0], (
        "Static bounding box batch size must be one or match env count; "
        f"got {corners.shape[0]} boxes for {pos_w.shape[0]} envs."
    )
    rotated = _quat_rotate_xyzw(quat_xyzw_w, corners)
    corners_w = rotated + pos_w.unsqueeze(1)
    return AxisAlignedBoundingBox(min_point=corners_w.amin(dim=1), max_point=corners_w.amax(dim=1))


def _quat_rotate_xyzw(quat_xyzw: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    quat_xyzw = quat_xyzw / quat_xyzw.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    q_vec = quat_xyzw[:, :3].unsqueeze(1)
    q_w = quat_xyzw[:, 3:].unsqueeze(1)
    q_vec = q_vec.expand_as(points)
    t = 2.0 * torch.cross(q_vec, points, dim=-1)
    return points + q_w * t + torch.cross(q_vec, t, dim=-1)
