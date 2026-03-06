# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_from_matrix

ALL_DATA_TYPES = [
    "rgb",
    "distance_to_image_plane",
    "normals",
    "motion_vectors",
    "semantic_segmentation",
    "instance_id_segmentation_fast",
]


class TrackType(enum.IntEnum):
    """Tracking category for each pixel in the scene-flow output."""

    STATIC = 0
    RIGID = 1
    ARTICULATION = 2
    UNSUPPORTED = 255


@dataclass
class SceneFlowResult:
    """Bundle returned by the exact adjacent-frame scene-flow computation."""

    scene_flow_3d: torch.Tensor
    """(H, W, 3) float32 — exact world-space displacement from frame t to t+1."""
    scene_flow_valid_mask: torch.Tensor
    """(H, W) bool — True where the flow value is trustworthy ground truth."""
    scene_flow_track_type: torch.Tensor
    """(H, W) uint8 — per-pixel :class:`TrackType` enum value."""


@dataclass
class FirstFrameFlowResult:
    """Bundle returned by :meth:`IsaacLabArenaCameraHandler.compute_first_frame_flow`.

    All tensors are defined over the frame-0 pixel grid ``(H, W)``.
    """

    flow3d_from_first: torch.Tensor
    """(H, W, 3) float32 — world-space displacement from frame-0 anchor to
    frame-k reconstructed position: ``p_k - p_0``."""
    trackable_mask: torch.Tensor
    """(H, W) bool — True for frame-0 pixels that are STATIC, RIGID, or
    ARTICULATION (constant across the sequence)."""
    in_frame_mask: torch.Tensor
    """(H, W) bool — True when the projected ``p_k`` falls inside the current
    image bounds **and** has ``z_cam > 0``."""
    visible_now_mask: torch.Tensor
    """(H, W) bool — True when the point is in-frame **and** depth-consistent
    with the current depth map (not occluded)."""
    points_world_k: torch.Tensor
    """(H, W, 3) float32 — reconstructed world positions ``p_k`` for every
    trackable anchor (debug / trajectory visualisation)."""


class IsaacLabArenaCameraHandler:
    """Handles a static camera sensor in an IsaacLab Arena environment.

    Wraps an IsaacLab Camera sensor and provides methods to extract
    RGB, depth, normals, optical flow, semantic segmentation,
    intrinsic matrix, and camera-to-world extrinsic matrix.

    Args:
        camera: An isaaclab.sensors.Camera instance.
        camera_name: Identifier used for file naming when writing data.
    """

    def __init__(self, camera: Any, camera_name: str = "static_camera"):
        self._camera = camera
        self._camera_name = camera_name
        # Persistent RGBA → object_id mapping for temporal consistency across frames
        self._color_to_object_id: Dict[tuple, int] = {}
        self._next_object_id: int = 0

        # Cached data for one-frame-lag exact scene flow
        self._prev_points_world: Optional[torch.Tensor] = None  # (H, W, 3)
        self._prev_track_type: Optional[torch.Tensor] = None  # (H, W) uint8
        self._prev_valid_mask: Optional[torch.Tensor] = None  # (H, W) bool
        self._prev_local_points: Optional[torch.Tensor] = None  # (H, W, 3)
        self._prev_rigid_keys: Optional[torch.Tensor] = None  # (H, W) int64
        self._prev_artic_keys: Optional[torch.Tensor] = None  # (H, W) int64
        self._prev_artic_body_idx: Optional[torch.Tensor] = None  # (H, W) int64

        # Mapping tables filled per cache_frame call
        self._rigid_key_to_name: Dict[int, str] = {}
        self._artic_key_to_name: Dict[int, str] = {}

        # First-frame anchors (set once by init_first_frame_anchors)
        self._ff_initialised: bool = False
        self._ff_p0_world: Optional[torch.Tensor] = None  # (H, W, 3)
        self._ff_trackable_mask: Optional[torch.Tensor] = None  # (H, W) bool
        self._ff_track_type: Optional[torch.Tensor] = None  # (H, W) uint8
        self._ff_local_points: Optional[torch.Tensor] = None  # (H, W, 3)
        self._ff_rigid_keys: Optional[torch.Tensor] = None  # (H, W) int64
        self._ff_artic_keys: Optional[torch.Tensor] = None  # (H, W) int64
        self._ff_artic_body_idx: Optional[torch.Tensor] = None  # (H, W) int64
        self._ff_rigid_key_to_name: Dict[int, str] = {}
        self._ff_artic_key_to_name: Dict[int, str] = {}

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def _get_camera_output(self) -> Dict[str, torch.Tensor]:
        from isaaclab.utils import convert_dict_to_backend

        return convert_dict_to_backend(self._camera.data.output, backend="torch")

    def update(self, dt: float) -> None:
        """Read the latest render data from the camera sensor."""
        self._camera.update(dt)

    # ------------------------------------------------------------------
    # Core outputs
    # ------------------------------------------------------------------

    def get_rgb(self) -> torch.Tensor:
        """Returns RGB image as a (H, W, 3) uint8 tensor."""
        rgb = self._get_camera_output()["rgb"].clone()
        assert rgb.shape[0] == 1, "Expected single environment"
        assert rgb.shape[-1] == 3, "Expected 3-channel RGB"
        return rgb.squeeze(0)

    def get_depth(self) -> torch.Tensor:
        """Returns depth map as a (H, W) float32 tensor (distance to image plane)."""
        depth = self._get_camera_output()["distance_to_image_plane"].clone()
        assert depth.shape[0] == 1, "Expected single environment"
        return depth.squeeze()

    def get_intrinsics(self) -> torch.Tensor:
        """Returns the 3x3 camera intrinsic matrix."""
        assert self._camera.data.intrinsic_matrices.shape[0] == 1, "Expected single environment"
        return self._camera.data.intrinsic_matrices.data[0].clone()

    def get_extrinsics(self) -> torch.Tensor:
        """Returns the 4x4 camera-to-world homogeneous transformation matrix."""
        assert self._camera.data.pos_w.shape[0] == 1, "Expected single environment"
        translation = self._camera.data.pos_w.data[0].clone()
        rotation_quat = self._camera.data.quat_w_ros.data[0].clone()

        T = torch.eye(4, dtype=torch.float32, device=rotation_quat.device)
        quat_flat = rotation_quat.reshape(4)
        T[:3, :3] = matrix_from_quat(quat_flat)
        T[:3, 3] = translation
        return T

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def get_normals(self) -> torch.Tensor:
        """Returns surface normals as a (H, W, 3) float32 tensor (x, y, z)."""
        normals = self._get_camera_output()["normals"].clone()
        assert normals.shape[0] == 1, "Expected single environment"
        assert normals.shape[-1] == 3, "Expected 3-channel normals"
        return normals.squeeze(0)

    # ------------------------------------------------------------------
    # Optical flow (motion vectors)
    # ------------------------------------------------------------------

    def get_optical_flow(self) -> torch.Tensor:
        """Returns dense optical flow as a (H, W, 2) float32 tensor (dx, dy in pixels)."""
        mv = self._get_camera_output()["motion_vectors"].clone()
        assert mv.shape[0] == 1, "Expected single environment"
        assert mv.shape[-1] == 2, "Expected 2-channel motion vectors"
        return mv.squeeze(0)

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    def get_semantic_segmentation(self) -> Tuple[torch.Tensor, Dict]:
        """Returns semantic segmentation data.

        Returns:
            Tuple of:
                - (H, W, 4) uint8 tensor with RGBA segmentation colors.
                - dict mapping RGBA id-strings to label dicts (``{"class": ...}``).
        """
        seg = self._get_camera_output()["semantic_segmentation"].clone()
        id_to_labels = self._camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        assert seg.shape[0] == 1, "Expected single environment"
        assert seg.shape[-1] == 4, "Expected 4-channel segmentation (RGBA)"
        return seg.squeeze(0).to(torch.uint8), id_to_labels

    # ------------------------------------------------------------------
    # Instance ID segmentation
    # ------------------------------------------------------------------

    def get_instance_id_segmentation(self) -> Tuple[torch.Tensor, Dict]:
        """Returns per-pixel integer instance IDs and the ID-to-prim-path mapping.

        Requires ``instance_id_segmentation_fast`` in the camera's data types
        with ``colorize_instance_id_segmentation=False``.

        Returns:
            Tuple of:
                - (H, W) int32 tensor of instance IDs.
                - dict mapping ID strings to label dicts (``{"class": <prim_path>}``).
        """
        seg = self._get_camera_output()["instance_id_segmentation_fast"].clone()
        id_to_labels = self._camera.data.info[0]["instance_id_segmentation_fast"]["idToLabels"]
        assert seg.shape[0] == 1, "Expected single environment"
        return seg.squeeze(0).squeeze(-1).to(torch.int32), id_to_labels

    # ------------------------------------------------------------------
    # 3D scene flow — exact adjacent-frame ground truth (SE(3) based)
    # ------------------------------------------------------------------

    def cache_scene_flow_frame(self, env: Any) -> None:
        """Cache per-pixel world points and tracking anchors for this frame.

        Must be called once per simulation step *after* :meth:`update`.
        The cached data is consumed by :meth:`compute_exact_scene_flow`
        on the *next* frame to produce one-frame-lag flow.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
        """
        from isaaclab.utils.math import unproject_depth

        depth = self.get_depth()
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()

        H, W = depth.shape
        device = depth.device

        # Unproject to world (preserving existing column-major → row-major fix)
        points_cam = unproject_depth(depth, intrinsics)  # (H*W, 3)
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (points_cam @ R.T) + t  # (H*W, 3)
        points_world = points_world.reshape(W, H, 3).permute(1, 0, 2).contiguous()

        valid_mask = torch.isfinite(depth)

        inst_ids, id_to_labels = self.get_instance_id_segmentation()

        scene = env.unwrapped.scene

        track_type = torch.full((H, W), TrackType.UNSUPPORTED, dtype=torch.uint8, device=device)
        local_points = torch.zeros(H, W, 3, dtype=torch.float32, device=device)

        rigid_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_body_idx = torch.full((H, W), -1, dtype=torch.int64, device=device)

        self._rigid_key_to_name = {}
        self._artic_key_to_name = {}

        rigid_pose_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        artic_pose_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        for uid in inst_ids.unique():
            uid_key = uid.item()
            if uid_key not in id_to_labels:
                continue

            label = id_to_labels[uid_key]
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                pixel_mask = (inst_ids == uid) & valid_mask
                if pixel_mask.any():
                    track_type[pixel_mask] = TrackType.STATIC
                continue

            binding = self._resolve_tracking_binding(prim_path, scene)

            pixel_mask = (inst_ids == uid) & valid_mask
            if not pixel_mask.any():
                continue

            pts = points_world[pixel_mask]  # (N, 3)

            if binding[0] == "STATIC":
                track_type[pixel_mask] = TrackType.STATIC

            elif binding[0] == "RIGID":
                asset_name = binding[1]
                track_type[pixel_mask] = TrackType.RIGID

                key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                self._rigid_key_to_name[key] = asset_name
                rigid_keys[pixel_mask] = key

                if asset_name not in rigid_pose_cache:
                    obj = scene[asset_name]
                    pos = obj.data.root_link_pos_w[0].clone()
                    quat = obj.data.root_link_quat_w[0].clone()  # (w, x, y, z)
                    rigid_pose_cache[asset_name] = (pos, quat)

                pos_t, quat_t = rigid_pose_cache[asset_name]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

            elif binding[0] == "ARTICULATION":
                asset_name, body_idx = binding[1], binding[2]
                track_type[pixel_mask] = TrackType.ARTICULATION

                artic_key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                self._artic_key_to_name[artic_key] = asset_name
                artic_keys[pixel_mask] = artic_key
                artic_body_idx[pixel_mask] = body_idx

                cache_key = (asset_name, body_idx)
                if cache_key not in artic_pose_cache:
                    artic_obj = scene[asset_name]
                    link_pos = artic_obj.data.body_link_pos_w[0, body_idx].clone()
                    link_quat = artic_obj.data.body_link_quat_w[0, body_idx].clone()
                    artic_pose_cache[cache_key] = (link_pos, link_quat)

                pos_t, quat_t = artic_pose_cache[cache_key]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

            else:
                pass  # UNSUPPORTED — already default

        self._prev_points_world = points_world
        self._prev_track_type = track_type
        self._prev_valid_mask = valid_mask
        self._prev_local_points = local_points
        self._prev_rigid_keys = rigid_keys
        self._prev_artic_keys = artic_keys
        self._prev_artic_body_idx = artic_body_idx

    def compute_exact_scene_flow(self, env: Any) -> Optional[SceneFlowResult]:
        """Compute exact adjacent-frame 3D scene flow for the *previous* frame.

        Uses the cached anchors from the previous call to
        :meth:`cache_scene_flow_frame` together with the current frame's
        SE(3) poses to produce ground-truth displacement.

        Returns ``None`` if no previous frame has been cached yet (first frame).

        Args:
            env: The gymnasium-wrapped IsaacLab environment.

        Returns:
            A :class:`SceneFlowResult` for the *previous* frame, or ``None``.
        """
        if self._prev_points_world is None:
            return None

        H, W = self._prev_points_world.shape[:2]
        device = self._prev_points_world.device

        scene = env.unwrapped.scene

        flow = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        valid = self._prev_valid_mask.clone()

        # --- STATIC pixels: displacement = 0 (already zero in flow) ---

        # --- RIGID pixels ---
        rigid_mask = self._prev_track_type == TrackType.RIGID
        if rigid_mask.any():
            rigid_pose_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for key, name in self._rigid_key_to_name.items():
                obj = scene[name]
                rigid_pose_cache[name] = (
                    obj.data.root_link_pos_w[0].clone(),
                    obj.data.root_link_quat_w[0].clone(),
                )

            for key, name in self._rigid_key_to_name.items():
                sel = rigid_mask & (self._prev_rigid_keys == key)
                if not sel.any():
                    continue
                pos_tp1, quat_tp1 = rigid_pose_cache[name]
                q_local = self._prev_local_points[sel]
                p_tp1 = quat_apply(
                    quat_tp1.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                ) + pos_tp1.unsqueeze(0)
                flow[sel] = p_tp1 - self._prev_points_world[sel]

        # --- ARTICULATION pixels ---
        artic_mask = self._prev_track_type == TrackType.ARTICULATION
        if artic_mask.any():
            artic_pose_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
            for artic_key, name in self._artic_key_to_name.items():
                artic_obj = scene[name]
                num_bodies = artic_obj.data.body_link_pos_w.shape[1]
                for bi in range(num_bodies):
                    cache_key = (name, bi)
                    if cache_key not in artic_pose_cache:
                        artic_pose_cache[cache_key] = (
                            artic_obj.data.body_link_pos_w[0, bi].clone(),
                            artic_obj.data.body_link_quat_w[0, bi].clone(),
                        )

            for artic_key, name in self._artic_key_to_name.items():
                artic_obj = scene[name]
                num_bodies = artic_obj.data.body_link_pos_w.shape[1]
                for bi in range(num_bodies):
                    sel = artic_mask & (self._prev_artic_keys == artic_key) & (self._prev_artic_body_idx == bi)
                    if not sel.any():
                        continue
                    pos_tp1, quat_tp1 = artic_pose_cache[(name, bi)]
                    q_local = self._prev_local_points[sel]
                    p_tp1 = quat_apply(
                        quat_tp1.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                    ) + pos_tp1.unsqueeze(0)
                    flow[sel] = p_tp1 - self._prev_points_world[sel]

        # --- UNSUPPORTED pixels: mark invalid ---
        unsupported = self._prev_track_type == TrackType.UNSUPPORTED
        valid[unsupported] = False

        return SceneFlowResult(
            scene_flow_3d=flow,
            scene_flow_valid_mask=valid,
            scene_flow_track_type=self._prev_track_type.clone(),
        )

    # ------------------------------------------------------------------
    # First-frame-anchored 3D trajectory flow
    # ------------------------------------------------------------------

    def init_first_frame_anchors(self, env: Any) -> None:
        """Store per-pixel anchor state from the current (frame-0) render.

        Must be called exactly once, after the first valid :meth:`update`.
        Subsequent calls raise :class:`RuntimeError`.

        For each pixel with finite depth the method records:

        * ``p0_world`` — the 3-D world position.
        * ``track_type`` — STATIC / RIGID / ARTICULATION / UNSUPPORTED.
        * ``trackable_mask`` — True for every type except UNSUPPORTED.
        * For RIGID / ARTICULATION pixels: the body-local anchor
          ``q0 = T_0^{-1} * p0`` so that later frames can reconstruct
          ``p_k = T_k * q0``.
        * For STATIC pixels: the world point itself (immovable).

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
        """
        if self._ff_initialised:
            raise RuntimeError("init_first_frame_anchors must only be called once")

        from isaaclab.utils.math import unproject_depth

        depth = self.get_depth()
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()

        H, W = depth.shape
        device = depth.device

        points_cam = unproject_depth(depth, intrinsics)
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (points_cam @ R.T) + t
        points_world = points_world.reshape(W, H, 3).permute(1, 0, 2).contiguous()

        valid_depth = torch.isfinite(depth)
        inst_ids, id_to_labels = self.get_instance_id_segmentation()
        scene = env.unwrapped.scene

        track_type = torch.full((H, W), TrackType.UNSUPPORTED, dtype=torch.uint8, device=device)
        local_points = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        rigid_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_keys = torch.full((H, W), -1, dtype=torch.int64, device=device)
        artic_body_idx = torch.full((H, W), -1, dtype=torch.int64, device=device)

        ff_rigid_key_to_name: Dict[int, str] = {}
        ff_artic_key_to_name: Dict[int, str] = {}

        rigid_pose_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        artic_pose_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        for uid in inst_ids.unique():
            uid_key = uid.item()
            if uid_key not in id_to_labels:
                continue

            label = id_to_labels[uid_key]
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                pixel_mask = (inst_ids == uid) & valid_depth
                if pixel_mask.any():
                    track_type[pixel_mask] = TrackType.STATIC
                continue

            binding = self._resolve_tracking_binding(prim_path, scene)
            pixel_mask = (inst_ids == uid) & valid_depth
            if not pixel_mask.any():
                continue

            pts = points_world[pixel_mask]

            if binding[0] == "STATIC":
                track_type[pixel_mask] = TrackType.STATIC
                local_points[pixel_mask] = pts

            elif binding[0] == "RIGID":
                asset_name = binding[1]
                track_type[pixel_mask] = TrackType.RIGID
                key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                ff_rigid_key_to_name[key] = asset_name
                rigid_keys[pixel_mask] = key

                if asset_name not in rigid_pose_cache:
                    obj = scene[asset_name]
                    rigid_pose_cache[asset_name] = (
                        obj.data.root_link_pos_w[0].clone(),
                        obj.data.root_link_quat_w[0].clone(),
                    )
                pos_t, quat_t = rigid_pose_cache[asset_name]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

            elif binding[0] == "ARTICULATION":
                asset_name, body_idx = binding[1], binding[2]
                track_type[pixel_mask] = TrackType.ARTICULATION
                artic_key = hash(asset_name) & 0x7FFFFFFFFFFFFFFF
                ff_artic_key_to_name[artic_key] = asset_name
                artic_keys[pixel_mask] = artic_key
                artic_body_idx[pixel_mask] = body_idx

                cache_key = (asset_name, body_idx)
                if cache_key not in artic_pose_cache:
                    artic_obj = scene[asset_name]
                    artic_pose_cache[cache_key] = (
                        artic_obj.data.body_link_pos_w[0, body_idx].clone(),
                        artic_obj.data.body_link_quat_w[0, body_idx].clone(),
                    )
                pos_t, quat_t = artic_pose_cache[cache_key]
                q_local = quat_apply_inverse(
                    quat_t.unsqueeze(0).expand(pts.shape[0], -1), pts - pos_t.unsqueeze(0)
                )
                local_points[pixel_mask] = q_local

        self._ff_p0_world = points_world
        self._ff_track_type = track_type
        self._ff_trackable_mask = (track_type != TrackType.UNSUPPORTED) & valid_depth
        self._ff_local_points = local_points
        self._ff_rigid_keys = rigid_keys
        self._ff_artic_keys = artic_keys
        self._ff_artic_body_idx = artic_body_idx
        self._ff_rigid_key_to_name = ff_rigid_key_to_name
        self._ff_artic_key_to_name = ff_artic_key_to_name
        self._ff_initialised = True

    def compute_first_frame_flow(
        self,
        env: Any,
        *,
        occlusion_tol: float = 0.05,
    ) -> FirstFrameFlowResult:
        """Compute flow from frame-0 anchors to the current frame k.

        For every trackable frame-0 pixel:

        * **STATIC**: ``p_k = p_0`` (no displacement).
        * **RIGID**: ``p_k = T_k * q_0`` where ``q_0 = T_0^{-1} * p_0``.
        * **ARTICULATION**: same as RIGID but using the per-body-link pose.

        Additionally computes visibility masks by projecting each ``p_k``
        into the current camera and comparing against the current depth map.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
            occlusion_tol: Depth tolerance (metres) for the occlusion test.

        Returns:
            A :class:`FirstFrameFlowResult`.

        Raises:
            RuntimeError: If :meth:`init_first_frame_anchors` was not called.
        """
        if not self._ff_initialised:
            raise RuntimeError("Call init_first_frame_anchors before compute_first_frame_flow")

        assert self._ff_p0_world is not None
        H, W = self._ff_p0_world.shape[:2]
        device = self._ff_p0_world.device

        scene = env.unwrapped.scene
        trackable = self._ff_trackable_mask.clone()
        assert trackable is not None

        p_k = self._ff_p0_world.clone()  # start from p0; STATIC stays here

        # --- RIGID anchors ---
        rigid_mask = (self._ff_track_type == TrackType.RIGID) & trackable
        if rigid_mask.any():
            for key, name in self._ff_rigid_key_to_name.items():
                sel = rigid_mask & (self._ff_rigid_keys == key)
                if not sel.any():
                    continue
                obj = scene[name]
                pos_k = obj.data.root_link_pos_w[0].clone()
                quat_k = obj.data.root_link_quat_w[0].clone()
                q_local = self._ff_local_points[sel]
                p_k[sel] = quat_apply(
                    quat_k.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                ) + pos_k.unsqueeze(0)

        # --- ARTICULATION anchors ---
        artic_mask = (self._ff_track_type == TrackType.ARTICULATION) & trackable
        if artic_mask.any():
            for artic_key, name in self._ff_artic_key_to_name.items():
                artic_obj = scene[name]
                num_bodies = artic_obj.data.body_link_pos_w.shape[1]
                for bi in range(num_bodies):
                    sel = artic_mask & (self._ff_artic_keys == artic_key) & (self._ff_artic_body_idx == bi)
                    if not sel.any():
                        continue
                    pos_k = artic_obj.data.body_link_pos_w[0, bi].clone()
                    quat_k = artic_obj.data.body_link_quat_w[0, bi].clone()
                    q_local = self._ff_local_points[sel]
                    p_k[sel] = quat_apply(
                        quat_k.unsqueeze(0).expand(q_local.shape[0], -1), q_local
                    ) + pos_k.unsqueeze(0)

        flow_0k = p_k - self._ff_p0_world

        # --- Visibility / projection masks ---
        intrinsics = self.get_intrinsics()  # (3, 3)
        extrinsics = self.get_extrinsics()  # (4, 4) cam-to-world
        depth_k = self.get_depth()  # (H, W)

        R_inv = extrinsics[:3, :3].T
        t_inv = -(R_inv @ extrinsics[:3, 3])

        p_k_flat = p_k.reshape(-1, 3)
        p_cam = (p_k_flat @ R_inv.T) + t_inv  # (H*W, 3)
        p_cam = p_cam.reshape(H, W, 3)

        z_cam = p_cam[..., 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        u = (p_cam[..., 0] / z_cam) * fx + cx
        v = (p_cam[..., 1] / z_cam) * fy + cy

        # Half-pixel tolerance: the unproject→reproject round-trip can
        # shift pixel centres by up to ~1e-4 px due to floating-point
        # precision, causing edge pixels to land just outside [0, dim).
        _PIX_EPS = 0.5
        in_frame = (
            (z_cam > 0)
            & (u >= -_PIX_EPS) & (u < W - 1 + _PIX_EPS)
            & (v >= -_PIX_EPS) & (v < H - 1 + _PIX_EPS)
        )

        u_idx = u.round().clamp(0, W - 1).long()
        v_idx = v.round().clamp(0, H - 1).long()
        sampled_depth = depth_k[v_idx, u_idx]

        visible_now = in_frame & (torch.abs(z_cam - sampled_depth) < occlusion_tol)

        return FirstFrameFlowResult(
            flow3d_from_first=flow_0k,
            trackable_mask=trackable,
            in_frame_mask=in_frame & trackable,
            visible_now_mask=visible_now & trackable,
            points_world_k=p_k,
        )

    # ------------------------------------------------------------------
    # 3D scene flow — legacy v*dt approximation (kept for compatibility)
    # ------------------------------------------------------------------

    def compute_scene_flow_3d(self, env: Any, dt: float) -> torch.Tensor:
        """Compute per-pixel 3D scene flow using ground-truth physics velocities.

        For each pixel belonging to a rigid body, computes the 3D displacement
        over one timestep:

        .. math::
            \\Delta p = (v_{\\text{lin}} + \\omega \\times (p - p_{\\text{com}})) \\cdot dt

        Pixels belonging to static/background geometry or unsupported asset
        types receive zero displacement.

        Args:
            env: The gymnasium-wrapped IsaacLab environment.
            dt: Simulation timestep in seconds.

        Returns:
            (H, W, 3) float32 tensor — per-pixel 3D displacement in world
            frame (metres).
        """
        from isaaclab.utils.math import unproject_depth

        depth = self.get_depth()                   # (H, W)
        intrinsics = self.get_intrinsics()         # (3, 3)
        extrinsics = self.get_extrinsics()         # (4, 4) cam-to-world

        H, W = depth.shape
        device = depth.device

        # Unproject depth to camera-space 3D points, then transform to world.
        # unproject_depth returns (H*W, 3) in column-major order (meshgrid
        # with indexing="ij" iterates width first), so we reshape via (W, H)
        # and transpose back to (H, W, 3).
        points_cam = unproject_depth(depth, intrinsics)      # (H*W, 3)
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (points_cam @ R.T) + t               # (H*W, 3)
        points_world = points_world.reshape(W, H, 3).permute(1, 0, 2).contiguous()

        valid_mask = torch.isfinite(depth)                   # (H, W)

        inst_ids, id_to_labels = self.get_instance_id_segmentation()

        scene_flow = torch.zeros(H, W, 3, dtype=torch.float32, device=device)

        scene = env.unwrapped.scene
        vel_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for uid in inst_ids.unique():
            uid_key = uid.item()
            if uid_key not in id_to_labels:
                continue

            label = id_to_labels[uid_key]
            prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
            if not prim_path or prim_path.upper() in ("BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"):
                continue

            asset_name = self._find_rigid_object_for_prim(prim_path, scene)
            if asset_name is None:
                continue

            if asset_name not in vel_cache:
                obj = scene[asset_name]
                vel_cache[asset_name] = (
                    obj.data.root_lin_vel_w[0].clone(),
                    obj.data.root_ang_vel_w[0].clone(),
                    obj.data.root_pos_w[0].clone(),
                )

            lin_vel, ang_vel, com_pos = vel_cache[asset_name]

            pixel_mask = (inst_ids == uid) & valid_mask
            if not pixel_mask.any():
                continue

            pts = points_world[pixel_mask]                          # (N, 3)
            r = pts - com_pos.unsqueeze(0)
            vel = lin_vel.unsqueeze(0) + torch.cross(
                ang_vel.unsqueeze(0).expand_as(r), r, dim=-1
            )
            scene_flow[pixel_mask] = vel * dt

        return scene_flow

    # ------------------------------------------------------------------
    # Prim-path → tracking binding resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tracking_binding(
        prim_path: str, scene: Any
    ) -> Tuple[str, ...]:
        """Classify a USD prim path into a tracking category.

        Returns one of:
            ("STATIC",)
            ("RIGID", asset_name)
            ("ARTICULATION", asset_name, body_idx)
            ("UNSUPPORTED",)
        """
        path_parts = prim_path.strip("/").split("/")

        for name in scene.rigid_objects.keys():
            if name in path_parts:
                return ("RIGID", name)

        for name, artic in scene.articulations.items():
            if name in path_parts:
                body_names = artic.data.body_names
                body_idx = _find_body_index_for_prim(prim_path, body_names)
                if body_idx is not None:
                    return ("ARTICULATION", name, body_idx)
                return ("ARTICULATION", name, 0)

        # Prim exists but doesn't belong to any tracked dynamic asset —
        # treat it as static world geometry (zero displacement).
        return ("STATIC",)

    @staticmethod
    def _find_rigid_object_for_prim(prim_path: str, scene: Any) -> str | None:
        """Match a USD prim path to a rigid-object name registered in the scene."""
        path_parts = prim_path.strip("/").split("/")
        for name in scene.rigid_objects.keys():
            if name in path_parts:
                return name
        return None

    # ------------------------------------------------------------------
    # Per-object semantic info
    # ------------------------------------------------------------------

    def _get_object_id(self, rgba: tuple) -> int:
        """Return a stable integer object ID for a given RGBA colour.

        The same colour always maps to the same ID across frames, so objects
        can be tracked temporally even when no semantic label is available.
        """
        if rgba not in self._color_to_object_id:
            self._color_to_object_id[rgba] = self._next_object_id
            self._next_object_id += 1
        return self._color_to_object_id[rgba]

    def get_semantic_info(self) -> List[Dict[str, Any]]:
        """Returns per-object metadata for every semantic class visible in this frame.

        For each distinct RGBA colour present in the segmentation image, reports:
        - ``object_id``: temporally consistent integer ID (same object → same ID
          across frames).
        - ``object_name``: stable name, e.g. ``"obj_0"``, ``"obj_1"``, ...
          If the USD prim has a ``semanticLabel`` the label is appended:
          ``"obj_0_cracker_box"``.
        - ``rgba``: the RGBA colour tuple.
        - ``class_name``: raw semantic class from USD (may be ``""`` or
          ``"BACKGROUND"``).
        - ``pixel_count``: number of pixels belonging to this object.

        Returns:
            List of dicts, one per visible object, sorted by descending pixel count.
        """
        seg_data, id_to_labels = self.get_semantic_segmentation()

        rgba_to_class: Dict[tuple, str] = {}
        for key, val in id_to_labels.items():
            rgba_tuple = tuple(eval(key)) if isinstance(key, str) else key
            rgba_to_class[rgba_tuple] = val.get("class", "")

        seg_np = seg_data.cpu().numpy()
        flat_seg = seg_np.reshape(-1, 4)
        unique_colors = np.unique(flat_seg, axis=0)

        results: List[Dict[str, Any]] = []
        for color in unique_colors:
            color_tuple = tuple(int(c) for c in color)
            mask = (seg_np == color.reshape(1, 1, 4)).all(axis=-1)
            pixel_count = int(mask.sum())
            if pixel_count == 0:
                continue

            object_id = self._get_object_id(color_tuple)
            class_name = rgba_to_class.get(color_tuple, "")

            # Build a human-readable, temporally consistent name
            if class_name and class_name.upper() not in ("UNKNOWN", "UNLABELLED", "BACKGROUND"):
                object_name = f"obj_{object_id}_{class_name}"
            else:
                object_name = f"obj_{object_id}"

            results.append({
                "object_id": object_id,
                "object_name": object_name,
                "rgba": color_tuple,
                "class_name": class_name,
                "pixel_count": pixel_count,
            })

        results.sort(key=lambda x: x["pixel_count"], reverse=True)
        return results


# ------------------------------------------------------------------
# Helper: find body index for articulation prim path
# ------------------------------------------------------------------


def _find_body_index_for_prim(prim_path: str, body_names: list[str]) -> int | None:
    """Find which articulation body a prim path belongs to.

    Matches the last body name that appears as a path component.
    """
    path_parts = prim_path.strip("/").split("/")
    best_idx: int | None = None
    best_depth = -1
    for idx, bname in enumerate(body_names):
        for depth, part in enumerate(path_parts):
            if part == bname and depth > best_depth:
                best_idx = idx
                best_depth = depth
    return best_idx


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def create_static_camera(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float],
    width: int = 640,
    height: int = 480,
    focal_length: float = 24.0,
    prim_path: str = "/World/StaticCamera",
) -> IsaacLabArenaCameraHandler:
    """Create and initialise a static camera sensor looking from *position* at *target*.

    Must be called **after** the simulation app has been launched and the
    environment has been reset so that the USD stage is ready.

    The camera is configured with all sensor data types needed for 3D
    reconstruction data generation: RGB, depth, normals, motion vectors
    (optical flow), and semantic segmentation.

    Args:
        position: Camera position in world frame (x, y, z).
        target: Look-at point in world frame (x, y, z).
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Focal length in mm.
        prim_path: USD prim path where the camera will be spawned.

    Returns:
        An initialised :class:`IsaacLabArenaCameraHandler`.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    quat_wxyz = _look_at_quaternion_ros(position, target)

    cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=ALL_DATA_TYPES,
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=quat_wxyz,
            convention="ros",
        ),
    )

    camera = Camera(cfg)
    camera._initialize_callback(None)
    camera.reset([0])
    return IsaacLabArenaCameraHandler(camera)


def _look_at_quaternion_ros(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[float, float, float, float]:
    """Compute a quaternion (w, x, y, z) for a ROS-convention camera at *eye* looking at *target*.

    ROS camera convention: x = right, y = down, z = forward (optical axis).
    """
    eye_arr = np.array(eye, dtype=np.float64)
    target_arr = np.array(target, dtype=np.float64)
    up_arr = np.array(up, dtype=np.float64)

    forward = target_arr - eye_arr
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_arr)
    right /= np.linalg.norm(right)

    down = np.cross(forward, right)

    R_cam_to_world = np.column_stack([right, down, forward])
    quat_tensor = quat_from_matrix(torch.from_numpy(R_cam_to_world).float())
    return tuple(quat_tensor.tolist())
