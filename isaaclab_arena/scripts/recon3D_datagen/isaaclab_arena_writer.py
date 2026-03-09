# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Subfolder names under output_dir (must match datagen_visualizer)
SUBFOLDER_COLOR = "color"
SUBFOLDER_DEPTH = "depth"
SUBFOLDER_FLOW2D = "flow2d"
SUBFOLDER_NORMAL = "normal"
SUBFOLDER_EXTRINSIC = "extrinsic"
SUBFOLDER_INTRINSIC = "intrinsic"
SUBFOLDER_SEMANTIC = "semantic"
SUBFOLDER_FLOW3D = "flow3d"
SUBFOLDER_FLOW3D_TRACK_TYPE = "flow3d_track_type"
SUBFOLDER_FLOW3D_FROM_FIRST = "flow3d_from_first"
SUBFOLDER_TRACKABLE_MASK = "trackable_mask"
SUBFOLDER_IN_FRAME_MASK = "in_frame_mask"
SUBFOLDER_VISIBLE_NOW_MASK = "visible_now_mask"


def camera_id_from_index(index: int) -> str:
    """Return the camera folder name for a given index (e.g. 0 -> \"cam0\")."""
    return f"cam{index}"


class IsaacLabArenaWriter:
    """Writes per-frame camera data to disk.

    Data is stored under *output_dir*/{camera_id}/ with subfolders color, depth,
    flow2d, normal, extrinsic, intrinsic, semantic (e.g. cam0/color, cam1/depth).
    Files use numeric names only::

        {camera_id}/{subfolder}/{frame_index:010d}.{ext}

    Frame indices are 0-based (e.g. 0000000000.png). Use camera_id=\"cam0\",
    \"cam1\", etc. for multiple views.

    Args:
        output_dir: Root directory; camera folders (cam0, cam1, ...) are created under it.
    """

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._subfolders = (
            SUBFOLDER_COLOR,
            SUBFOLDER_DEPTH,
            SUBFOLDER_FLOW2D,
            SUBFOLDER_FLOW3D,
            SUBFOLDER_FLOW3D_TRACK_TYPE,
            SUBFOLDER_FLOW3D_FROM_FIRST,
            SUBFOLDER_TRACKABLE_MASK,
            SUBFOLDER_IN_FRAME_MASK,
            SUBFOLDER_VISIBLE_NOW_MASK,
            SUBFOLDER_NORMAL,
            SUBFOLDER_EXTRINSIC,
            SUBFOLDER_INTRINSIC,
            SUBFOLDER_SEMANTIC,
        )

    def _ensure_camera_dirs(self, camera_id: str) -> None:
        """Create output_dir/camera_id/{color,depth,...} if needed."""
        for sub in self._subfolders:
            os.makedirs(os.path.join(self._output_dir, camera_id, sub), exist_ok=True)

    def _path(self, camera_id: str, subfolder: str, filename: str) -> str:
        return os.path.join(self._output_dir, camera_id, subfolder, filename)

    # ------------------------------------------------------------------
    # Core outputs
    # ------------------------------------------------------------------

    def write_rgb(
        self, rgb: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save an RGB image as a PNG file.

        Args:
            rgb: (H, W, 3) uint8 tensor.
            camera_id: Folder name for this camera (e.g. \"cam0\", \"cam1\").
        """
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_COLOR, f"{frame_index:010d}.png")
        Image.fromarray(rgb.cpu().numpy()).save(path)

    def write_depth(
        self, depth: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save a depth map as a float32 ``.npy`` file.

        Args:
            depth: (H, W) float32 tensor (metres, distance-to-image-plane).
            camera_id: Folder name for this camera (e.g. \"cam0\").
        """
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_DEPTH, f"{frame_index:010d}.npy")
        np.save(path, depth.cpu().numpy().astype(np.float32))

    def write_intrinsics(
        self, intrinsics: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save the 3x3 intrinsic matrix as a ``.npy`` file."""
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_INTRINSIC, f"{frame_index:010d}.npy")
        np.save(path, intrinsics.cpu().numpy())

    def write_extrinsics(
        self, extrinsics: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save the 4x4 camera-to-world matrix as a ``.npy`` file."""
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_EXTRINSIC, f"{frame_index:010d}.npy")
        np.save(path, extrinsics.cpu().numpy())

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def write_normals(
        self, normals: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save surface normals as a float32 ``.npy`` file.

        Args:
            normals: (H, W, 3) float32 tensor (x, y, z world-space normals).
            camera_id: Folder name for this camera (e.g. \"cam0\").
        """
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_NORMAL, f"{frame_index:010d}.npy")
        np.save(path, normals.cpu().numpy().astype(np.float32))

    # ------------------------------------------------------------------
    # Optical flow (motion vectors)
    # ------------------------------------------------------------------

    def write_optical_flow(
        self, flow: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save dense optical flow as a float32 ``.npy`` file.

        Args:
            flow: (H, W, 2) float32 tensor (dx, dy in pixels).
            camera_id: Folder name for this camera (e.g. \"cam0\").
        """
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_FLOW2D, f"{frame_index:010d}.npy")
        np.save(path, flow.cpu().numpy().astype(np.float32))

    # ------------------------------------------------------------------
    # 3D scene flow
    # ------------------------------------------------------------------

    def write_scene_flow_3d(
        self, flow3d: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save per-pixel 3D scene flow as a float32 ``.npy`` file.

        Args:
            flow3d: (H, W, 3) float32 tensor — 3D displacement in world frame (metres).
            camera_id: Folder name for this camera (e.g. ``"cam0"``).
        """
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_FLOW3D, f"{frame_index:010d}.npy")
        np.save(path, flow3d.cpu().numpy().astype(np.float32))

    # Per-category colours for track-type visualisation (RGB)
    _TRACK_TYPE_COLORS = {
        0: (128, 128, 128),  # STATIC  — grey
        1: (0, 160, 255),    # RIGID   — blue
        2: (255, 160, 0),    # ARTICULATION — orange
        255: (255, 0, 0),    # UNSUPPORTED  — red
    }

    def write_scene_flow_track_type(
        self,
        track_type: torch.Tensor,
        camera_id: str,
        frame_index: int,
        *,
        camera_name: str = "",
    ) -> None:
        """Save per-pixel tracking type as both ``.npy`` and a colorised ``.png``.

        Values follow :class:`~isaaclab_arena_camera_handler.TrackType`:
        0 = STATIC (grey), 1 = RIGID (blue), 2 = ARTICULATION (orange),
        255 = UNSUPPORTED (red).

        Args:
            track_type: (H, W) uint8 tensor.
            camera_id: Folder name for this camera.
        """
        self._ensure_camera_dirs(camera_id)
        tt_np = track_type.cpu().numpy()
        tag = f"{frame_index:010d}"

        np.save(self._path(camera_id, SUBFOLDER_FLOW3D_TRACK_TYPE, f"{tag}.npy"), tt_np)

        H, W = tt_np.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for val, color in self._TRACK_TYPE_COLORS.items():
            rgb[tt_np == val] = color
        Image.fromarray(rgb).save(
            self._path(camera_id, SUBFOLDER_FLOW3D_TRACK_TYPE, f"{tag}.png")
        )

    # ------------------------------------------------------------------
    # First-frame-anchored trajectory flow
    # ------------------------------------------------------------------

    def write_flow3d_from_first(
        self,
        flow: torch.Tensor,
        camera_id: str,
        frame_index: int,
        *,
        camera_name: str = "",
    ) -> None:
        """Save per-pixel 3-D flow from frame 0 as float32 ``.npy``.

        Args:
            flow: (H, W, 3) float32 tensor — ``p_k - p_0`` in metres.
        """
        self._ensure_camera_dirs(camera_id)
        path = self._path(camera_id, SUBFOLDER_FLOW3D_FROM_FIRST, f"{frame_index:010d}.npy")
        np.save(path, flow.cpu().numpy().astype(np.float32))

    def _write_bool_mask_png(
        self, mask: torch.Tensor, camera_id: str, subfolder: str, frame_index: int
    ) -> None:
        """Save a boolean mask as a single-channel PNG (white = True)."""
        self._ensure_camera_dirs(camera_id)
        arr = (mask.cpu().numpy().astype(np.uint8)) * 255
        path = self._path(camera_id, subfolder, f"{frame_index:010d}.png")
        Image.fromarray(arr, mode="L").save(path)

    def write_trackable_mask(
        self, mask: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save the trackable mask (constant from frame 0) as PNG."""
        self._write_bool_mask_png(mask, camera_id, SUBFOLDER_TRACKABLE_MASK, frame_index)

    def write_in_frame_mask(
        self, mask: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save the in-frame projection mask as PNG."""
        self._write_bool_mask_png(mask, camera_id, SUBFOLDER_IN_FRAME_MASK, frame_index)

    def write_visible_now_mask(
        self, mask: torch.Tensor, camera_id: str, frame_index: int, *, camera_name: str = ""
    ) -> None:
        """Save the visible-now (depth-consistent) mask as PNG."""
        self._write_bool_mask_png(mask, camera_id, SUBFOLDER_VISIBLE_NOW_MASK, frame_index)

    # ------------------------------------------------------------------
    # Semantic segmentation + metadata
    # ------------------------------------------------------------------

    def write_semantic_segmentation(
        self,
        seg_data: torch.Tensor,
        semantic_info: List[Dict[str, Any]],
        camera_id: str,
        frame_index: int,
        *,
        camera_name: str = "",
    ) -> None:
        """Save semantic segmentation as a 4-channel RGBA PNG plus a JSON metadata file.

        The JSON file lists every object visible in this frame with its RGBA
        colour, class name, and pixel count.

        Args:
            seg_data: (H, W, 4) uint8 tensor.
            semantic_info: Per-object metadata from
                :meth:`IsaacLabArenaCameraHandler.get_semantic_info`.
            camera_id: Folder name for this camera (e.g. \"cam0\").
        """
        self._ensure_camera_dirs(camera_id)
        png_path = self._path(camera_id, SUBFOLDER_SEMANTIC, f"{frame_index:010d}.png")
        Image.fromarray(seg_data.cpu().numpy()).save(png_path)
        json_path = self._path(camera_id, SUBFOLDER_SEMANTIC, f"{frame_index:010d}.json")
        with open(json_path, "w") as f:
            json.dump({"objects": semantic_info}, f, indent=2)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def write_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        normals: torch.Tensor,
        semantic_seg: torch.Tensor,
        semantic_info: List[Dict[str, Any]],
        camera_id: str,
        frame_index: int,
        *,
        camera_name: str = "",
        optical_flow: Optional[torch.Tensor] = None,
        scene_flow_3d: Optional[torch.Tensor] = None,
        scene_flow_track_type: Optional[torch.Tensor] = None,
    ) -> None:
        """Write all data types for a single frame.

        Static modalities (rgb, depth, intrinsics, extrinsics, normals,
        semantics) are always written.  Flow modalities (optical_flow,
        scene_flow_3d, scene_flow_track_type) are only written when provided.
        This allows the last frame in a sequence to be saved without flow.

        Validity of scene flow can be derived from scene_flow_track_type:
        ``valid = (track_type != 255)`` (UNSUPPORTED).

        Args:
            camera_id: Folder name for this camera (e.g. ``"cam0"``, ``"cam1"``).
            optical_flow: Optional (H, W, 2) float32 forward optical flow.
            scene_flow_3d: Optional (H, W, 3) float32 3-D scene flow.
            scene_flow_track_type: Optional (H, W) uint8 per-pixel track type.
        """
        self.write_rgb(rgb, camera_id, frame_index, camera_name=camera_name)
        self.write_depth(depth, camera_id, frame_index, camera_name=camera_name)
        self.write_intrinsics(intrinsics, camera_id, frame_index, camera_name=camera_name)
        self.write_extrinsics(extrinsics, camera_id, frame_index, camera_name=camera_name)
        self.write_normals(normals, camera_id, frame_index, camera_name=camera_name)
        self.write_semantic_segmentation(
            semantic_seg, semantic_info, camera_id, frame_index, camera_name=camera_name
        )
        if optical_flow is not None:
            self.write_optical_flow(optical_flow, camera_id, frame_index, camera_name=camera_name)
        if scene_flow_3d is not None:
            self.write_scene_flow_3d(scene_flow_3d, camera_id, frame_index, camera_name=camera_name)
        if scene_flow_track_type is not None:
            self.write_scene_flow_track_type(
                scene_flow_track_type, camera_id, frame_index, camera_name=camera_name
            )
