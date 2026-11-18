# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Analytic visibility of task-relevant surfaces from the egocentric camera.

A surface is visible when at least one of its sample points lies inside the camera frustum and the segment from the
camera to the point crosses no box: scene parts (fixtures and items) and a box model of the robot's hand.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass

import isaaclab.utils.math as math_utils

from isaaclab_arena_environments.spatial_memory.geometry import (
    compose_poses,
    inverse_transform_points,
    points_in_pinhole_frustum,
    sample_box_face_points,
    segments_blocked_by_boxes,
)


@dataclass(frozen=True)
class CameraModel:
    """Pinhole model of an egocentric camera rigidly mounted on a robot body."""

    mount_body: str
    """Robot body the camera is attached to."""

    T_body_cam: tuple[float, ...]
    """Pose of the camera's ROS optical frame (z forward, x right, y down) in the mount body frame."""

    focal_x_px: float
    """Horizontal focal length in pixels."""

    focal_y_px: float
    """Vertical focal length in pixels."""

    width_px: int
    """Image width in pixels."""

    height_px: int
    """Image height in pixels."""

    near_m: float = 0.02
    """Near clipping distance."""

    far_m: float = 6.0
    """Far clipping distance."""

    @property
    def horizontal_fov_deg(self) -> float:
        """Full horizontal field of view."""
        return math.degrees(2.0 * math.atan(0.5 * self.width_px / self.focal_x_px))

    @property
    def vertical_fov_deg(self) -> float:
        """Full vertical field of view."""
        return math.degrees(2.0 * math.atan(0.5 * self.height_px / self.focal_y_px))


def camera_model_from_cfg(camera_cfg, mount_body: str) -> CameraModel:
    """Build a ``CameraModel`` from an Isaac Lab ``CameraCfg`` with a pinhole spawn and a ROS or OpenGL offset.

    The renderer fits the horizontal aperture to the image width and keeps pixels square, so the vertical focal
    length equals the horizontal one (this is checked against rendered segmentation in the suite's camera test).
    """
    convention = camera_cfg.offset.convention
    assert convention in ("ros", "opengl"), f"Unsupported camera offset convention {convention!r}."
    spawn = camera_cfg.spawn
    focal_x_px = camera_cfg.width * spawn.focal_length / spawn.horizontal_aperture
    q_body_cam = tuple(float(v) for v in camera_cfg.offset.rot)
    if convention == "opengl":
        q_opengl = torch.tensor(q_body_cam).unsqueeze(0)
        q_body_cam = tuple(
            math_utils.convert_camera_frame_orientation_convention(q_opengl, origin="opengl", target="ros")[0].tolist()
        )
    T_body_cam = tuple(float(v) for v in camera_cfg.offset.pos) + q_body_cam
    near_m, far_m = spawn.clipping_range
    return CameraModel(
        mount_body=mount_body,
        T_body_cam=T_body_cam,
        focal_x_px=focal_x_px,
        focal_y_px=focal_x_px,
        width_px=camera_cfg.width,
        height_px=camera_cfg.height,
        near_m=max(near_m, 0.02),
        far_m=min(far_m, 6.0),
    )


@dataclass(frozen=True)
class VisibilityTarget:
    """A task-relevant surface whose visibility is logged every step."""

    name: str
    """Name reported in logs and tests."""

    part_name: str
    """Scene part whose live pose carries the sample points."""

    faces: tuple[int, ...]
    """Box faces of the part to sample (see ``geometry.FACE_*``)."""

    spacing_m: float = 0.03
    """Sample spacing on each face."""


@dataclass(frozen=True)
class RobotOccluder:
    """Box model of a robot body that can block the camera's view (for example the hand of a wrist camera)."""

    body_name: str
    """Robot body carrying the box."""

    T_body_box: tuple[float, ...]
    """Pose of the box centre in the body frame."""

    half_extents: tuple[float, float, float]
    """Half extents of the box."""


class VisibilityModel:
    """Batched visibility of a fixed list of targets from one camera.

    Args:
        camera: Egocentric camera model.
        targets: Surfaces whose visibility is reported.
        part_sizes: Full extents of every scene part, keyed by part name.
        part_names: Scene parts used as occluders, in the order their poses are passed to ``compute``.
        robot_occluders: Boxes attached to robot bodies that also occlude.
        min_visible_points: Number of unblocked in-frustum points needed to call a target visible.
        device: Torch device.
    """

    def __init__(
        self,
        camera: CameraModel,
        targets: list[VisibilityTarget],
        part_sizes: dict[str, tuple[float, float, float]],
        part_names: list[str],
        robot_occluders: list[RobotOccluder],
        min_visible_points: int = 1,
        device: str = "cpu",
    ):
        self.camera = camera
        self.targets = targets
        self.target_names = [target.name for target in targets]
        self.part_names = part_names
        self.robot_occluders = robot_occluders
        self.min_visible_points = min_visible_points
        self.device = device
        point_chunks = []
        target_index_chunks = []
        part_index_chunks = []
        for target_index, target in enumerate(targets):
            points = sample_box_face_points(part_sizes[target.part_name], target.faces, spacing_m=target.spacing_m)
            point_chunks.append(points)
            target_index_chunks.append(torch.full((points.shape[0],), target_index, dtype=torch.long))
            part_index_chunks.append(torch.full((points.shape[0],), part_names.index(target.part_name)))
        self.points_local = torch.cat(point_chunks).to(device) if point_chunks else torch.zeros(0, 3, device=device)
        self.point_target_index = (
            torch.cat(target_index_chunks).to(device) if target_index_chunks else torch.zeros(0, dtype=torch.long)
        )
        self.point_part_index = (
            torch.cat(part_index_chunks).to(device) if part_index_chunks else torch.zeros(0, dtype=torch.long)
        )
        part_half = [[0.5 * s for s in part_sizes[name]] for name in part_names]
        occluder_half = [list(occluder.half_extents) for occluder in robot_occluders]
        self.box_half_extents = torch.tensor(part_half + occluder_half, device=device, dtype=torch.float32)
        self.T_body_box = torch.tensor(
            [list(occluder.T_body_box) for occluder in robot_occluders], device=device, dtype=torch.float32
        ).reshape(-1, 7)
        self.T_body_cam = torch.tensor(camera.T_body_cam, device=device, dtype=torch.float32)

    def camera_pose_w(self, T_W_mount: torch.Tensor) -> torch.Tensor:
        """Return the camera's ROS optical-frame pose (N, 7) from the mount body pose (N, 7)."""
        return compose_poses(T_W_mount, self.T_body_cam.expand(T_W_mount.shape[0], 7))

    def compute(
        self,
        T_W_mount: torch.Tensor,
        T_W_parts: torch.Tensor,
        T_W_occluder_bodies: torch.Tensor,
        part_enabled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the number of visible sample points per target.

        Args:
            T_W_mount: Camera mount body poses (N, 7).
            T_W_parts: Poses of the scene parts (N, K, 7), ordered like ``part_names``.
            T_W_occluder_bodies: Poses of the robot occluder bodies (N, R, 7), ordered like ``robot_occluders``.
            part_enabled: Optional (N, K) mask of parts that take part in occlusion.

        Returns:
            Visible point counts with shape (N, T).
        """
        num_envs = T_W_mount.shape[0]
        num_targets = len(self.targets)
        if num_targets == 0:
            return torch.zeros(num_envs, 0, device=T_W_mount.device, dtype=torch.long)
        T_W_cam = self.camera_pose_w(T_W_mount)
        # Target points in the world: each point rides on its part's live pose.
        T_W_point_parts = T_W_parts[:, self.point_part_index]
        points_W = (
            math_utils.quat_apply(T_W_point_parts[..., 3:7], self.points_local.unsqueeze(0)) + T_W_point_parts[..., :3]
        )
        points_C = inverse_transform_points(T_W_cam, points_W)
        in_frustum = points_in_pinhole_frustum(
            points_C,
            self.camera.focal_x_px,
            self.camera.focal_y_px,
            self.camera.width_px,
            self.camera.height_px,
            self.camera.near_m,
            self.camera.far_m,
        )
        T_W_boxes = T_W_parts
        enabled = part_enabled
        if len(self.robot_occluders) > 0:
            T_W_robot_boxes = compose_poses(T_W_occluder_bodies, self.T_body_box.unsqueeze(0).expand(num_envs, -1, 7))
            T_W_boxes = torch.cat([T_W_parts, T_W_robot_boxes], dim=1)
            if enabled is not None:
                robot_enabled = torch.ones(num_envs, len(self.robot_occluders), dtype=torch.bool, device=enabled.device)
                enabled = torch.cat([enabled, robot_enabled], dim=1)
        blocked = segments_blocked_by_boxes(T_W_cam[:, :3], points_W, T_W_boxes, self.box_half_extents, enabled)
        visible_points = (in_frustum & ~blocked).long()
        counts = torch.zeros(num_envs, num_targets, dtype=torch.long, device=T_W_mount.device)
        counts.index_add_(1, self.point_target_index, visible_points)
        return counts
