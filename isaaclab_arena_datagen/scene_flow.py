# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Scene-flow data types and computation.

:class:`SceneFlowComputer` is owned by
:class:`~isaaclab_arena_camera_handler.IsaacLabArenaCameraHandler` which
passes camera data (depth, intrinsics, T_W_from_C, segmentation) as
explicit method arguments.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3
from isaaclab_arena_datagen.object_registry import ObjectType
from isaaclab_arena_datagen.utils.isaac_data import to_torch

BACKGROUND_LABELS = frozenset({"BACKGROUND", "UNLABELLED", "UNKNOWN", "INVALID"})


@dataclass
class SceneFlowResult:
    """Bundle returned by the exact adjacent-frame scene-flow computation."""

    scene_flow_W_hw3: torch.Tensor
    """(H, W, 3) float32 -- world-space displacement from frame t to t+1.
    Use :meth:`~IsaacLabArenaCameraHandler.convert_scene_flow_W_to_C` to
    obtain camera-relative flow that also captures camera ego-motion."""
    scene_flow_valid_mask_hw: torch.Tensor
    """(H, W) bool -- True where the flow value is trustworthy ground truth."""
    scene_flow_track_type_hw: torch.Tensor
    """(H, W) uint8 -- per-pixel :class:`ObjectType` enum value."""


@dataclass
class _PixelTrackingResult:
    """Result of per-pixel tracking classification."""

    points_W_hw3: torch.Tensor
    """(H, W, 3) float32 -- unprojected world positions."""
    valid_mask_hw: torch.Tensor
    """(H, W) bool -- True where depth is finite."""
    track_type_hw: torch.Tensor
    """(H, W) uint8 -- per-pixel ObjectType."""
    points_localbody_hw3: torch.Tensor
    """(H, W, 3) float32 -- body-local coordinates for RIGID/ARTICULATION."""
    rigid_keys_hw: torch.Tensor
    """(H, W) int64 -- hash key identifying the rigid object."""
    articulation_keys_hw: torch.Tensor
    """(H, W) int64 -- hash key identifying the articulation."""
    articulation_body_idx_hw: torch.Tensor
    """(H, W) int64 -- body-link index within the articulation."""
    rigid_key_to_name: dict[int, str]
    """Mapping from rigid object integer key to scene asset name."""
    articulation_key_to_name: dict[int, str]
    """Mapping from articulation integer key to scene asset name."""


def _classify_pixels_by_tracking(  # pylint: disable=too-many-statements  # per-object-type classification pipeline
    depth_hw: torch.Tensor,
    intrinsics_33: torch.Tensor,
    T_W_from_C: TransformSE3,
    instance_ids_hw: torch.Tensor,
    id_to_labels: dict,
    scene: Any,
    resolve_binding_fn: Callable[[str, Any], tuple[Any, ...]],
) -> _PixelTrackingResult:
    """Unproject depth and classify each pixel by tracking type.

    Shared by adjacent-frame caching and anchor-frame initialisation.

    Args:
        depth_hw: (H, W) float32 depth map.
        intrinsics_33: (3, 3) camera intrinsic matrix.
        T_W_from_C: Camera-to-world transform (TransformSE3).
        instance_ids_hw: (H, W) int32 instance IDs.
        id_to_labels: Mapping from instance ID to label dict.
        scene: The IsaacLab scene object.
        resolve_binding_fn: Callable that classifies a prim path.

    Returns:
        A :class:`_PixelTrackingResult` with per-pixel tracking data.
    """
    from isaaclab.utils.math import quat_apply_inverse, unproject_depth

    H, W = depth_hw.shape
    device = depth_hw.device

    points_C_n3 = unproject_depth(depth_hw, intrinsics_33)
    rotation_W_from_C = T_W_from_C.rotation.R.reshape(3, 3)
    translation_W_from_C = T_W_from_C.translation.t.reshape(3)
    points_W_hw3 = (points_C_n3 @ rotation_W_from_C.T) + translation_W_from_C
    points_W_hw3 = points_W_hw3.reshape(W, H, 3).permute(1, 0, 2).contiguous()

    valid_mask_hw = torch.isfinite(depth_hw)

    track_type_hw = torch.full((H, W), ObjectType.UNSUPPORTED, dtype=torch.uint8, device=device)
    points_localbody_hw3 = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
    rigid_keys_hw = torch.full((H, W), -1, dtype=torch.int64, device=device)
    articulation_keys_hw = torch.full((H, W), -1, dtype=torch.int64, device=device)
    articulation_body_idx_hw = torch.full((H, W), -1, dtype=torch.int64, device=device)

    rigid_key_to_name: dict[int, str] = {}
    articulation_key_to_name: dict[int, str] = {}

    # Deterministic name-to-key mappings (avoids hash non-determinism)
    _name_to_rigid_key: dict[str, int] = {}
    _name_to_articulation_key: dict[str, int] = {}
    _next_key: int = 1

    rigid_pose_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    articulation_pose_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    for instance_id in instance_ids_hw.unique():
        uid_key = instance_id.item()
        if uid_key not in id_to_labels:
            continue

        label = id_to_labels[uid_key]
        prim_path = label.get("class", "") if isinstance(label, dict) else str(label)
        if not prim_path or prim_path.upper() in BACKGROUND_LABELS:
            mask_hw = (instance_ids_hw == instance_id) & valid_mask_hw
            if mask_hw.any():
                track_type_hw[mask_hw] = ObjectType.STATIC
            continue

        binding = resolve_binding_fn(prim_path, scene)

        mask_hw = (instance_ids_hw == instance_id) & valid_mask_hw
        if not mask_hw.any():
            continue

        selected_W_n3 = points_W_hw3[mask_hw]

        if binding[0] == "STATIC":
            track_type_hw[mask_hw] = ObjectType.STATIC
            points_localbody_hw3[mask_hw] = selected_W_n3

        elif binding[0] == "RIGID":
            asset_name = binding[1]
            track_type_hw[mask_hw] = ObjectType.RIGID

            if asset_name not in _name_to_rigid_key:
                _name_to_rigid_key[asset_name] = _next_key
                _next_key += 1
            key = _name_to_rigid_key[asset_name]
            rigid_key_to_name[key] = asset_name
            rigid_keys_hw[mask_hw] = key

            if asset_name not in rigid_pose_cache:
                obj = scene[asset_name]
                translation_W = to_torch(obj.data.root_link_pos_w)[0].clone()
                quat_W = to_torch(obj.data.root_link_quat_w)[0].clone()
                rigid_pose_cache[asset_name] = (translation_W, quat_W)

            # Transform world-space points into the rigid body's local frame
            translation_W, quat_W = rigid_pose_cache[asset_name]
            points_W_n3 = selected_W_n3 - translation_W.unsqueeze(0)
            quat_W_n4 = quat_W.unsqueeze(0).expand(selected_W_n3.shape[0], -1)
            points_localbody_n3 = quat_apply_inverse(quat_W_n4, points_W_n3)
            points_localbody_hw3[mask_hw] = points_localbody_n3

        elif binding[0] == "ARTICULATION":
            asset_name, body_idx = binding[1], binding[2]
            track_type_hw[mask_hw] = ObjectType.ARTICULATION

            if asset_name not in _name_to_articulation_key:
                _name_to_articulation_key[asset_name] = _next_key
                _next_key += 1
            articulation_key = _name_to_articulation_key[asset_name]
            articulation_key_to_name[articulation_key] = asset_name
            articulation_keys_hw[mask_hw] = articulation_key
            articulation_body_idx_hw[mask_hw] = body_idx

            cache_key = (asset_name, body_idx)
            if cache_key not in articulation_pose_cache:
                articulation_obj = scene[asset_name]
                link_translation_W = to_torch(articulation_obj.data.body_link_pos_w)[0, body_idx].clone()
                link_quat_W = to_torch(articulation_obj.data.body_link_quat_w)[0, body_idx].clone()
                articulation_pose_cache[cache_key] = (link_translation_W, link_quat_W)

            # Transform world-space points into the articulation link's local frame
            translation_W, quat_W = articulation_pose_cache[cache_key]
            points_W_n3 = selected_W_n3 - translation_W.unsqueeze(0)
            quat_W_n4 = quat_W.unsqueeze(0).expand(selected_W_n3.shape[0], -1)
            points_localbody_n3 = quat_apply_inverse(quat_W_n4, points_W_n3)
            points_localbody_hw3[mask_hw] = points_localbody_n3

    return _PixelTrackingResult(
        points_W_hw3=points_W_hw3,
        valid_mask_hw=valid_mask_hw,
        track_type_hw=track_type_hw,
        points_localbody_hw3=points_localbody_hw3,
        rigid_keys_hw=rigid_keys_hw,
        articulation_keys_hw=articulation_keys_hw,
        articulation_body_idx_hw=articulation_body_idx_hw,
        rigid_key_to_name=rigid_key_to_name,
        articulation_key_to_name=articulation_key_to_name,
    )


@dataclass
class _CachedFrame:
    """Snapshot of per-pixel tracking state for a single frame."""

    points_W_hw3: torch.Tensor
    T_W_from_C: TransformSE3
    track_type_hw: torch.Tensor
    valid_mask_hw: torch.Tensor
    points_localbody_hw3: torch.Tensor
    rigid_keys_hw: torch.Tensor
    articulation_keys_hw: torch.Tensor
    articulation_body_idx_hw: torch.Tensor


class SceneFlowComputer:
    """Computes adjacent-frame scene flow from camera data.

    Owns all per-frame tracking state.  Camera data (depth, intrinsics,
    T_W_from_C, segmentation) is passed as explicit method arguments by
    the owning camera handler.
    """

    def __init__(self) -> None:
        """Initialize with empty tracking state."""
        self._prev: _CachedFrame | None = None

        self._rigid_key_to_name: dict[int, str] = {}
        self._articulation_key_to_name: dict[int, str] = {}

    def cache_frame(
        self,
        depth_hw: torch.Tensor,
        intrinsics_33: torch.Tensor,
        T_W_from_C: TransformSE3,
        instance_ids_hw: torch.Tensor,
        id_to_labels: dict,
        scene: Any,
        resolve_binding_fn: Callable[[str, Any], tuple[Any, ...]],
    ) -> None:
        """Cache per-pixel world points and tracking anchors for this frame.

        Must be called once per simulation step.  The cached data is
        consumed by :meth:`compute_exact_flow` on the *next* frame.

        Args:
            depth_hw: (H, W) float32 depth map.
            intrinsics_33: (3, 3) camera intrinsic matrix.
            T_W_from_C: Camera-to-world transform (TransformSE3).
            instance_ids_hw: (H, W) int32 instance IDs.
            id_to_labels: Mapping from instance ID to label dict.
            scene: The IsaacLab scene object.
            resolve_binding_fn: Callable that classifies a prim path
                into a tracking category tuple.
        """
        result = _classify_pixels_by_tracking(
            depth_hw,
            intrinsics_33,
            T_W_from_C,
            instance_ids_hw,
            id_to_labels,
            scene,
            resolve_binding_fn,
        )

        self._prev = _CachedFrame(
            points_W_hw3=result.points_W_hw3,
            T_W_from_C=T_W_from_C,
            track_type_hw=result.track_type_hw,
            valid_mask_hw=result.valid_mask_hw,
            points_localbody_hw3=result.points_localbody_hw3,
            rigid_keys_hw=result.rigid_keys_hw,
            articulation_keys_hw=result.articulation_keys_hw,
            articulation_body_idx_hw=result.articulation_body_idx_hw,
        )
        self._rigid_key_to_name = result.rigid_key_to_name
        self._articulation_key_to_name = result.articulation_key_to_name

    def compute_exact_flow(self, scene: Any) -> SceneFlowResult | None:
        """Compute exact adjacent-frame 3D scene flow for the *previous* frame.

        Uses the cached anchors from :meth:`cache_frame` together with
        the current SE(3) poses to produce ground-truth displacement.

        Returns ``None`` if no previous frame has been cached yet.

        Args:
            scene: The IsaacLab scene object.
        """
        if self._prev is None:
            return None

        from isaaclab.utils.math import quat_apply

        prev = self._prev

        H, W = prev.points_W_hw3.shape[:2]
        device = prev.points_W_hw3.device

        flow_W_hw3 = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        valid_mask_hw = prev.valid_mask_hw.clone()

        rigid_mask_hw = prev.track_type_hw == ObjectType.RIGID
        if rigid_mask_hw.any():
            rigid_pose_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for key, name in self._rigid_key_to_name.items():
                obj = scene[name]
                rigid_pose_cache[name] = (
                    to_torch(obj.data.root_link_pos_w)[0].clone(),
                    to_torch(obj.data.root_link_quat_w)[0].clone(),
                )

            for key, name in self._rigid_key_to_name.items():
                selected_mask_hw = rigid_mask_hw & (prev.rigid_keys_hw == key)
                if not selected_mask_hw.any():
                    continue
                translation_next_W, quat_next_W = rigid_pose_cache[name]
                points_localbody_n3 = prev.points_localbody_hw3[selected_mask_hw]
                points_next_W_n3 = quat_apply(
                    quat_next_W.unsqueeze(0).expand(points_localbody_n3.shape[0], -1),
                    points_localbody_n3,
                ) + translation_next_W.unsqueeze(0)
                flow_W_hw3[selected_mask_hw] = points_next_W_n3 - prev.points_W_hw3[selected_mask_hw]

        articulation_mask_hw = prev.track_type_hw == ObjectType.ARTICULATION
        if articulation_mask_hw.any():
            articulation_pose_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
            for articulation_key, name in self._articulation_key_to_name.items():
                articulation_obj = scene[name]
                num_bodies = articulation_obj.data.body_link_pos_w.shape[1]
                for body_index in range(num_bodies):
                    cache_key = (name, body_index)
                    if cache_key not in articulation_pose_cache:
                        articulation_pose_cache[cache_key] = (
                            to_torch(articulation_obj.data.body_link_pos_w)[0, body_index].clone(),
                            to_torch(articulation_obj.data.body_link_quat_w)[0, body_index].clone(),
                        )

            for articulation_key, name in self._articulation_key_to_name.items():
                articulation_obj = scene[name]
                num_bodies = articulation_obj.data.body_link_pos_w.shape[1]
                for body_index in range(num_bodies):
                    selected_mask_hw = (
                        articulation_mask_hw
                        & (prev.articulation_keys_hw == articulation_key)
                        & (prev.articulation_body_idx_hw == body_index)
                    )
                    if not selected_mask_hw.any():
                        continue
                    translation_next_W, quat_next_W = articulation_pose_cache[(name, body_index)]
                    points_localbody_n3 = prev.points_localbody_hw3[selected_mask_hw]
                    points_next_W_n3 = quat_apply(
                        quat_next_W.unsqueeze(0).expand(points_localbody_n3.shape[0], -1),
                        points_localbody_n3,
                    ) + translation_next_W.unsqueeze(0)
                    flow_W_hw3[selected_mask_hw] = points_next_W_n3 - prev.points_W_hw3[selected_mask_hw]

        unsupported_mask_hw = prev.track_type_hw == ObjectType.UNSUPPORTED
        valid_mask_hw[unsupported_mask_hw] = False

        return SceneFlowResult(
            scene_flow_W_hw3=flow_W_hw3,
            scene_flow_valid_mask_hw=valid_mask_hw,
            scene_flow_track_type_hw=prev.track_type_hw.clone(),
        )

    @staticmethod
    def _convert_flow_W_to_C(
        points_src_W_hw3: torch.Tensor,
        points_dst_W_hw3: torch.Tensor,
        T_W_from_C_src: TransformSE3,
        T_W_from_C_dst: TransformSE3,
    ) -> torch.Tensor:
        """Convert world-space point clouds to camera-relative 3D flow.

        Args:
            points_src_W_hw3: (H, W, 3) world positions at the source frame.
            points_dst_W_hw3: (H, W, 3) world positions at the destination frame.
            T_W_from_C_src: Camera-to-world transform of the source frame.
            T_W_from_C_dst: Camera-to-world transform of the destination frame.

        Returns:
            (H, W, 3) float32 camera-relative 3D displacement.
        """
        H, W = points_src_W_hw3.shape[:2]

        rotation_W_from_C_src = T_W_from_C_src.rotation.R.reshape(3, 3)
        translation_W_from_C_src = T_W_from_C_src.translation.t.reshape(3)
        points_src_C_n3 = (points_src_W_hw3.reshape(-1, 3) - translation_W_from_C_src) @ rotation_W_from_C_src

        rotation_W_from_C_dst = T_W_from_C_dst.rotation.R.reshape(3, 3)
        translation_W_from_C_dst = T_W_from_C_dst.translation.t.reshape(3)
        points_dst_C_n3 = (points_dst_W_hw3.reshape(-1, 3) - translation_W_from_C_dst) @ rotation_W_from_C_dst

        flow_C_hw3 = (points_dst_C_n3 - points_src_C_n3).reshape(H, W, 3)
        flow_C_hw3[~torch.isfinite(flow_C_hw3)] = 0.0
        return flow_C_hw3

    def convert_scene_flow_W_to_C(self, flow_W_hw3: torch.Tensor, T_W_from_C: TransformSE3) -> torch.Tensor | None:
        """Convert adjacent-frame world-space scene flow to camera-relative.

        Args:
            flow_W_hw3: (H, W, 3) world-space 3D displacement.
            T_W_from_C: Camera-to-world transform of the current frame.

        Returns:
            (H, W, 3) camera-relative 3D displacement, or ``None`` if
            no previous frame data is cached.
        """
        if self._prev is None:
            return None
        points_dst_W_hw3 = self._prev.points_W_hw3 + flow_W_hw3
        return self._convert_flow_W_to_C(self._prev.points_W_hw3, points_dst_W_hw3, self._prev.T_W_from_C, T_W_from_C)

    def compute_true_optical_flow(
        self,
        intrinsics_33: torch.Tensor,
        T_W_from_C: TransformSE3,
        scene_flow_W_hw3: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute true 2D optical flow including camera ego-motion.

        Args:
            intrinsics_33: (3, 3) current camera intrinsic matrix.
            T_W_from_C: Current camera-to-world transform (TransformSE3).
            scene_flow_W_hw3: (H, W, 3) world-space displacement.
                If ``None``, only camera ego-motion contributes.

        Returns:
            (H, W, 2) float32 pixel displacement, or ``None`` if no
            previous frame has been cached.
        """
        if self._prev is None:
            return None

        H, W = self._prev.points_W_hw3.shape[:2]
        device = self._prev.points_W_hw3.device

        if scene_flow_W_hw3 is not None:
            points_W_hw3 = self._prev.points_W_hw3 + scene_flow_W_hw3
        else:
            points_W_hw3 = self._prev.points_W_hw3

        rotation_W_from_C = T_W_from_C.rotation.R.reshape(3, 3)
        translation_W_from_C = T_W_from_C.translation.t.reshape(3)
        points_C_n3 = (points_W_hw3.reshape(-1, 3) - translation_W_from_C.unsqueeze(0)) @ rotation_W_from_C

        fx, fy = intrinsics_33[0, 0], intrinsics_33[1, 1]
        cx, cy = intrinsics_33[0, 2], intrinsics_33[1, 2]
        z_C_n = points_C_n3[:, 2]
        u_proj_hw = (fx * points_C_n3[:, 0] / z_C_n + cx).reshape(H, W)
        v_proj_hw = (fy * points_C_n3[:, 1] / z_C_n + cy).reshape(H, W)

        grid_v_hw, grid_u_hw = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        optical_flow_hw2 = torch.stack([u_proj_hw - grid_u_hw, v_proj_hw - grid_v_hw], dim=-1)

        valid_mask_hw = self._prev.valid_mask_hw
        behind_mask_hw = z_C_n.reshape(H, W) <= 0
        optical_flow_hw2[~valid_mask_hw | behind_mask_hw] = 0.0

        return optical_flow_hw2
