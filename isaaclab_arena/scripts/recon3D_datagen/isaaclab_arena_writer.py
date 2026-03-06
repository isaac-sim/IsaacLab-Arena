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

    def write_output_readme(self) -> None:
        """Write a README.md in the output directory describing all data types and folder layout."""
        path = os.path.join(self._output_dir, "README.md")
        with open(path, "w") as f:
            f.write(_OUTPUT_README_CONTENT)
        print(f"[IsaacLabArenaWriter] Wrote {path}")


# ---------------------------------------------------------------------------
# Output folder documentation (written as README.md)
# ---------------------------------------------------------------------------

_OUTPUT_README_CONTENT = """# Recon3D data generation output

This folder contains per-frame camera data produced by the Isaac Lab Arena datagen pipeline. Each camera has its own subfolder (e.g. `cam0/`, `cam1/`). Within a camera folder, data is organised by modality in subfolders. Files use 10-digit zero-padded frame indices (e.g. `0000000000.png`, `0000000001.npy`).

## Folder layout

```
(output root)/
├── README.md          (this file)
├── cam0/
│   ├── color/         RGB images
│   ├── depth/         Depth maps
│   ├── flow2d/        Optical flow (2D)
│   ├── flow3d/        Adjacent-frame 3D scene flow
│   ├── flow3d_track_type/   Track type for adjacent flow
│   ├── flow3d_from_first/   First-frame-anchored 3D flow
│   ├── trackable_mask/      Trackable mask (frame-0 anchors)
│   ├── in_frame_mask/       In-frame projection mask
│   ├── visible_now_mask/    Visible-now (occlusion) mask
│   ├── normal/         Surface normals
│   ├── intrinsic/      Camera intrinsic matrices
│   ├── extrinsic/      Camera-to-world matrices
│   ├── semantic/       Semantic segmentation + metadata
│   └── visualizations/ Pre-rendered visualisations (if generated)
├── cam1/
│   └── ...
```

## Data types

### Static modalities (one file per frame, indices 0 .. N-1)

| Subfolder   | Format | Description |
|------------|--------|-------------|
| **color**  | PNG    | RGB image (H×W×3), uint8. |
| **depth**  | .npy   | Depth map (H×W), float32, metres, distance to image plane. |
| **intrinsic** | .npy | 3×3 camera intrinsic matrix (e.g. focal length, principal point). |
| **extrinsic** | .npy | 4×4 camera-to-world homogeneous transform. |
| **normal** | .npy   | Surface normals (H×W×3), float32, world-space x,y,z. |
| **semantic** | .png + .json | 4-channel RGBA segmentation image; JSON lists visible objects with class names and pixel counts. |

### Adjacent-frame flow (one file per transition, indices 0 .. N-2; last frame has no forward flow)

| Subfolder   | Format | Description |
|------------|--------|-------------|
| **flow2d** | .npy   | Dense optical flow (H×W×2), float32, (dx, dy) in pixels. Flow at index i: frame i → frame i+1. |
| **flow3d** | .npy   | 3D scene flow (H×W×3), float32, world-space displacement in metres. Same indexing as flow2d. |
| **flow3d_track_type** | .npy + .png | Per-pixel track type (H×W), uint8. Values: 0=Static, 1=Rigid, 2=Articulation, 255=Unsupported. Validity: use pixels where track_type ≠ 255. The PNG is a colorised view (grey/blue/orange/red). |

### First-frame-anchored trajectory flow (one file per frame, indices 0 .. N-1)

Flow from frame-0 anchor positions to the current frame: `flow_0k = p_k - p_0`, defined for every trackable frame-0 pixel even when occluded or out of view.

| Subfolder   | Format | Description |
|------------|--------|-------------|
| **flow3d_from_first** | .npy | 3D displacement from frame-0 to current frame (H×W×3), float32, metres. |
| **trackable_mask** | PNG | Boolean mask (white=valid). True where the frame-0 pixel has a ground-truth tracking model (Static, Rigid, or Articulation). Constant over the sequence. Use to filter pixels where flow is exact GT. |
| **in_frame_mask** | PNG | Boolean mask (white=valid). True where the reconstructed 3D point at the current frame projects inside the current image (and z_cam > 0). Tracks whether the point left the field of view. |
| **visible_now_mask** | PNG | Boolean mask (white=valid). True where the point is in-frame and its depth matches the current depth map (not occluded). Strictest visibility. |

Relationship: `trackable_mask ⊇ in_frame_mask ⊇ visible_now_mask`. Flow values remain valid for all trackable pixels; the masks indicate visibility for downstream use (e.g. loss weighting or rendering comparison).

## Frame index convention

- Static modalities and first-frame flow: frame index = simulation step index (0, 1, …, N-1).
- Adjacent flow (flow2d, flow3d, flow3d_track_type): stored at the *source* frame index. File at index i describes motion from frame i to frame i+1; the last frame (N-1) has no adjacent-flow file.

## Visualisations

If the datagen script runs the visualisation step, each camera folder will contain a `visualizations/` subfolder with:

- `data_vis.png` — Grid of sampled frames: color, depth, flow2d, flow3d, track type, flow-from-first, in-frame mask, visible-now mask, normals, semantics.
- `camera_trajectory_3d.png` — 3D camera path with coordinate frames.
- `scene_flow_3d_frame*.html` — Interactive 3D view of adjacent scene flow (plotly).
- `first_frame_flow_3d_frame*.html` — Interactive 3D view of first-frame-anchored flow (plotly).
"""

