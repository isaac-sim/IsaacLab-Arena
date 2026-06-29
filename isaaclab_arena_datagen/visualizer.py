# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Visualization utilities for recon3D data generation outputs.

Provides functions to:
- Combined grid: color, depth, normals, track type, semantics, flow2d, flow3d
  in one figure.
- Plot 3D camera trajectory with coordinate frames and optional frustums.
- Interactive 3D scene flow on a point cloud (plotly, rotatable).
- Individual grids for RGB, depth, normals, optical flow, semantic segmentation.
"""

from __future__ import annotations

# pylint: disable=too-many-lines  # visualization module covers many output modalities
import colorsys
import h5py
import json
import numpy as np
import os
from scipy.spatial import Delaunay, QhullError

import imageio.v3 as iio
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from isaaclab_arena_datagen.dynamic_object_tracker import MeshSamplesResult
from isaaclab_arena_datagen.io import hdf5_keys as Keys
from isaaclab_arena_datagen.utils.mesh_utils import reconstruct_mesh_points_at_step

# ---------------------------------------------------------------------------
# Colormaps (pure-numpy, replaces matplotlib colormaps)
# ---------------------------------------------------------------------------

# Key colour stops for Spectral_r (Spectral reversed): position, R, G, B in [0,1].
# Derived from matplotlib's Spectral colormap (Brewer palette).
_COLORMAP_SPECTRAL_R = np.array(
    [
        [0.00, 0.369, 0.310, 0.635],
        [0.10, 0.243, 0.471, 0.714],
        [0.20, 0.455, 0.678, 0.820],
        [0.30, 0.671, 0.851, 0.914],
        [0.40, 0.878, 0.953, 0.732],
        [0.50, 1.000, 1.000, 0.749],
        [0.60, 0.996, 0.878, 0.545],
        [0.70, 0.992, 0.682, 0.380],
        [0.80, 0.957, 0.427, 0.263],
        [0.90, 0.836, 0.188, 0.153],
        [1.00, 0.620, 0.004, 0.259],
    ],
    dtype=np.float64,
)

# Key colour stops for Inferno.
# Derived from matplotlib's Inferno colormap (Stef van der Walt, Nathaniel Smith).
_COLORMAP_INFERNO = np.array(
    [
        [0.00, 0.001, 0.000, 0.014],
        [0.10, 0.073, 0.042, 0.253],
        [0.20, 0.233, 0.060, 0.432],
        [0.30, 0.419, 0.056, 0.456],
        [0.40, 0.578, 0.148, 0.389],
        [0.50, 0.735, 0.258, 0.260],
        [0.60, 0.865, 0.396, 0.142],
        [0.70, 0.954, 0.559, 0.039],
        [0.80, 0.976, 0.756, 0.153],
        [0.90, 0.955, 0.937, 0.432],
        [1.00, 0.988, 1.000, 0.644],
    ],
    dtype=np.float64,
)

_COLORMAPS: dict[str, np.ndarray] = {
    "Spectral_r": _COLORMAP_SPECTRAL_R,
    "inferno": _COLORMAP_INFERNO,
}


def _apply_colormap(values: np.ndarray, cmap_name: str) -> np.ndarray:
    """Apply a named colormap to normalised [0, 1] values.

    Returns:
        ``(*values.shape, 3)`` uint8 array.
    """
    stops = _COLORMAPS[cmap_name]
    positions = stops[:, 0]
    colors = stops[:, 1:4]

    shape = values.shape
    values_flat_n = values.ravel().astype(np.float64).clip(0.0, 1.0)

    idx = np.searchsorted(positions, values_flat_n, side="right") - 1
    idx = np.clip(idx, 0, len(positions) - 2)

    seg_len = positions[idx + 1] - positions[idx]
    t = np.where(seg_len > 1e-12, (values_flat_n - positions[idx]) / seg_len, 0.0)
    t = t.clip(0.0, 1.0)

    rgb = colors[idx] * (1.0 - t[:, None]) + colors[idx + 1] * t[:, None]
    return (rgb.reshape(*shape, 3) * 255).astype(np.uint8)


def _hsv_to_rgb(hsv_hw3: np.ndarray) -> np.ndarray:
    """Convert an HSV array (H, S, V in [0, 1]) to RGB ([0, 1]).

    Args:
        hsv_hw3: ``(..., 3)`` float array.

    Returns:
        ``(..., 3)`` float array with RGB values in [0, 1].
    """
    h = hsv_hw3[..., 0]
    s = hsv_hw3[..., 1]
    v = hsv_hw3[..., 2]

    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t_val = v * (1.0 - s * (1.0 - f))

    rgb_hw3 = np.zeros_like(hsv_hw3)
    for mask_val, r_src, g_src, b_src in [
        (0, v, t_val, p),
        (1, q, v, p),
        (2, p, v, t_val),
        (3, p, q, v),
        (4, t_val, p, v),
        (5, v, p, q),
    ]:
        m = i == mask_val
        rgb_hw3[m, 0] = r_src[m]
        rgb_hw3[m, 1] = g_src[m]
        rgb_hw3[m, 2] = b_src[m]

    return rgb_hw3


# ---------------------------------------------------------------------------
# PIL-based image composition helpers
# ---------------------------------------------------------------------------

_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a TrueType font at *size*, falling back to PIL's built-in default."""
    for path in _FONT_SEARCH_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure an image is ``(H, W, 3)`` uint8."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1).astype(np.uint8)
    return img[..., :3].astype(np.uint8)


def _pad_to_size(img: np.ndarray, h: int, w: int, fill: int = 255) -> np.ndarray:
    """Pad ``img`` to ``(h, w, 3)`` with a constant fill value."""
    out = np.full((h, w, 3), fill, dtype=np.uint8)
    ih = min(img.shape[0], h)
    iw = min(img.shape[1], w)
    src = _ensure_rgb(img)
    out[:ih, :iw] = src[:ih, :iw]
    return out


def _tile_images(cells: list[list[np.ndarray]], gap: int = 2, bg: int = 255) -> np.ndarray:
    """Tile a 2-D list of images into a single image (handles variable row heights)."""
    nrows = len(cells)
    ncols = max(len(row) for row in cells) if cells else 0
    if nrows == 0 or ncols == 0:
        return np.full((10, 10, 3), bg, dtype=np.uint8)

    col_width = max(img.shape[1] for row in cells for img in row)
    row_heights = [max(img.shape[0] for img in row) for row in cells]

    total_h = sum(row_heights) + max(nrows - 1, 0) * gap
    total_w = ncols * col_width + max(ncols - 1, 0) * gap
    canvas_hw3 = np.full((total_h, total_w, 3), bg, dtype=np.uint8)

    y = 0
    for r, row in enumerate(cells):
        for c, img in enumerate(row):
            x0 = c * (col_width + gap)
            canvas_hw3[y : y + row_heights[r], x0 : x0 + col_width] = _pad_to_size(
                img, row_heights[r], col_width, fill=bg
            )
        y += row_heights[r] + gap

    return canvas_hw3


def _add_title_bar(img: np.ndarray, title: str) -> np.ndarray:
    """Return a copy of *img* with a centred title bar above it."""
    if not title:
        return img
    h, w = img.shape[:2]
    font_size = max(8, min(h, w) // 6)
    bar_h = font_size + 6

    title_bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    pil_bar = Image.fromarray(title_bar)
    draw = ImageDraw.Draw(pil_bar)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((max(0, (w - tw) // 2), 2), title, fill=(0, 0, 0), font=font)

    return np.concatenate([np.array(pil_bar), _ensure_rgb(img)], axis=0)


def _add_suptitle_bar(grid_hw3: np.ndarray, suptitle: str) -> np.ndarray:
    """Return a copy of *grid_hw3* with a centred super-title bar above it."""
    if not suptitle:
        return grid_hw3
    w = grid_hw3.shape[1]
    font_size = max(11, w // 60)
    bar_h = font_size + 10

    title_bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    pil_bar = Image.fromarray(title_bar)
    draw = ImageDraw.Draw(pil_bar)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), suptitle, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((max(0, (w - tw) // 2), 3), suptitle, fill=(0, 0, 0), font=font)

    return np.concatenate([np.array(pil_bar), grid_hw3], axis=0)


def _blank_with_na(h: int, w: int) -> np.ndarray:
    """Create a dark ``(h, w, 3)`` uint8 image with centred 'N/A' text."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    text = "N/A"
    font = _get_font(max(8, min(h, w) // 4))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(128, 128, 128), font=font)
    return np.array(pil_img)


def _overlay_frame_label(img: np.ndarray, label: str) -> np.ndarray:
    """Return a copy of *img* with a frame label in the top-left corner."""
    result = _ensure_rgb(img).copy()
    pil_img = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(max(8, img.shape[0] // 8))
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(2, img.shape[0] // 40)
    draw.rectangle([pad, pad, pad + tw + 2 * pad, pad + th + 2 * pad], fill=(0, 0, 0))
    draw.text((2 * pad, 2 * pad), label, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def _overlay_legend(img: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Return a copy of *img* with coloured legend entries in the top-left corner.

    Each entry is a dict with ``color`` (tuple of 0-1 floats) and ``label`` (str).
    """
    if not legends:
        return img
    result = _ensure_rgb(img).copy()
    pil_img = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(max(6, img.shape[0] // 16))

    line_h = max(10, img.shape[0] // 12)
    swatch = max(6, line_h - 4)
    pad = max(2, img.shape[0] // 40)

    max_label_w = 0
    for entry in legends:
        bbox = draw.textbbox((0, 0), entry["label"], font=font)
        max_label_w = max(max_label_w, bbox[2] - bbox[0])

    bg_w = swatch + pad + max_label_w + 3 * pad
    bg_h = len(legends) * line_h + 2 * pad
    draw.rectangle([pad, pad, pad + bg_w, pad + bg_h], fill=(255, 255, 255))

    y = 2 * pad
    for entry in legends:
        fill_color = tuple(int(v * 255) for v in entry["color"][:3])
        draw.rectangle([2 * pad, y, 2 * pad + swatch, y + swatch], fill=fill_color, outline=(0, 0, 0))
        draw.text((2 * pad + swatch + pad, y - 1), entry["label"], fill=(0, 0, 0), font=font)
        y += line_h

    return np.array(pil_img)


def _save_or_show(
    image: np.ndarray,
    save_path: str | None,
    title: str = "",
    func_name: str = "",
) -> None:
    """Save a composed image as PNG or display interactively with plotly."""
    if save_path:
        Image.fromarray(image).save(save_path)
        print(f"[{func_name}] Saved to {save_path}")
    else:
        fig = go.Figure(go.Image(z=image))
        fig.update_layout(
            title_text=title,
            margin={"l": 0, "r": 0, "t": 40 if title else 0, "b": 0},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.show()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_TRACK_TYPE_COLORS_VIS = {
    0: (128, 128, 128),  # STATIC  -> grey
    1: (0, 160, 255),  # RIGID   -> blue
    2: (255, 160, 0),  # ARTICULATION -> orange
    255: (255, 0, 0),  # UNSUPPORTED  -> red
}


def _track_type_to_rgb(track_type_hw: np.ndarray) -> np.ndarray:
    """Convert a uint8 track-type map to an ``(H, W, 3)`` RGB image."""
    H, W = track_type_hw.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for val, color in _TRACK_TYPE_COLORS_VIS.items():
        rgb[track_type_hw == val] = color
    return rgb


def _semantic_ids_to_rgba(semantic_hw: np.ndarray, semantic_info: list[dict] | None) -> np.ndarray:
    """Convert an int32 ID map back to ``(H, W, 4)`` RGBA using colour metadata."""
    H, W = semantic_hw.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if not semantic_info:
        return rgba
    for obj_idx, obj in enumerate(semantic_info, start=1):
        color = obj.get("rgba") or obj.get("color") or [128, 128, 128, 255]
        rgba[semantic_hw == obj_idx] = np.array(color, dtype=np.uint8)
    return rgba


def _load_frames(hdf5_path: str, camera_id: str) -> dict:
    """Load all frames from an HDF5 dataset file.

    Returns:
        Dictionary with keys: rgbs, depths, intrinsics, extrinsics, normals,
        optical_flows, scene_flows_3d, flow3d_track_types, semantics,
        semantic_infos, frame_indices.
    """
    result: dict = {
        "rgbs": [],
        "depths": [],
        "intrinsics": [],
        "extrinsics": [],
        "normals": [],
        "optical_flows": [],
        "scene_flows_3d": [],
        "flow3d_track_types": [],
        "semantics": [],
        "semantic_infos": [],
        "frame_indices": [],
    }

    with h5py.File(hdf5_path, "r") as f:
        seq_names = sorted(k for k in f.keys() if k.startswith(Keys.SEQUENCE_GROUP_PREFIX))
        if not seq_names:
            return result
        seq_group = f[seq_names[0]]
        if camera_id not in seq_group:
            return result
        cam_group = seq_group[camera_id]

        num_frames = int(seq_group.attrs[Keys.ATTR_NUM_FRAMES])
        has_flow2d = Keys.FLOW2D in cam_group
        has_flow3d = Keys.FLOW3D in cam_group
        has_track_type = Keys.FLOW3D_TRACK_TYPE in cam_group
        num_flow = num_frames - 1

        for frame_idx in range(num_frames):
            result["frame_indices"].append(frame_idx)
            result["rgbs"].append(cam_group[Keys.COLOR][frame_idx])
            result["depths"].append(cam_group[Keys.DEPTH][frame_idx])
            result["intrinsics"].append(cam_group[Keys.INTRINSIC][frame_idx].reshape(3, 3))

            ext_34 = cam_group[Keys.EXTRINSIC][frame_idx]
            ext_44 = np.eye(4, dtype=np.float64)
            ext_44[:3, :] = ext_34
            result["extrinsics"].append(ext_44)

            result["normals"].append(cam_group[Keys.NORMAL][frame_idx])

            if has_flow2d and frame_idx < num_flow:
                result["optical_flows"].append(cam_group[Keys.FLOW2D][frame_idx])
            else:
                result["optical_flows"].append(None)

            if has_flow3d and frame_idx < num_flow:
                result["scene_flows_3d"].append(cam_group[Keys.FLOW3D][frame_idx])
            else:
                result["scene_flows_3d"].append(None)

            if has_track_type and frame_idx < num_flow:
                raw_tt = cam_group[Keys.FLOW3D_TRACK_TYPE][frame_idx]
                result["flow3d_track_types"].append(_track_type_to_rgb(raw_tt))
            else:
                result["flow3d_track_types"].append(None)

            semantic_hw = cam_group[Keys.SEMANTIC][frame_idx]
            json_str = cam_group[Keys.SEMANTIC_JSON][frame_idx]
            if json_str:
                info = json.loads(json_str)
                objects_list = info.get("objects", [])
                result["semantic_infos"].append(info)
                result["semantics"].append(_semantic_ids_to_rgba(semantic_hw, objects_list))
            else:
                result["semantic_infos"].append(None)
                result["semantics"].append(None)

    return result


def _load_dynamic_objects_from_hdf5(
    hdf5_path: str,
) -> tuple[int | None, dict[str, np.ndarray], MeshSamplesResult]:
    """Load dynamic object poses and mesh samples from an HDF5 file.

    Returns:
        ``(num_steps, T_W_from_localbody_arrays, mesh_samples)`` or
        ``(None, {}, MeshSamplesResult({}))`` when no dynamic data is present.
    """
    empty: tuple[None, dict, MeshSamplesResult] = (
        None,
        {},
        MeshSamplesResult(se3_localbody_from_point_arrays={}),
    )
    with h5py.File(hdf5_path, "r") as f:
        seq_names = sorted(k for k in f.keys() if k.startswith(Keys.SEQUENCE_GROUP_PREFIX))
        if not seq_names:
            return empty
        seq_group = f[seq_names[0]]

        dyn_path = Keys.DYNAMIC_OBJECTS
        if dyn_path not in seq_group:
            return empty

        dyn_group = seq_group[dyn_path]
        metadata_str = dyn_group.attrs.get(Keys.ATTR_METADATA_JSON, "")
        if not metadata_str:
            return empty
        meta = json.loads(metadata_str)
        num_steps = meta.get("metadata", {}).get("num_steps")

        poses_group = dyn_group.get(Keys.POSES)
        T_W_from_localbody_arrays: dict[str, np.ndarray] = {}
        if poses_group is not None:
            poses_group.visititems(
                lambda name, obj: (
                    T_W_from_localbody_arrays.__setitem__(name, obj[:]) if isinstance(obj, h5py.Dataset) else None
                )
            )

        mesh_group = dyn_group.get(Keys.MESH_SAMPLES)
        se3_arrays: dict[str, np.ndarray] = {}
        if mesh_group is not None:
            mesh_group.visititems(
                lambda name, obj: (se3_arrays.__setitem__(name, obj[:]) if isinstance(obj, h5py.Dataset) else None)
            )

    return (
        num_steps,
        T_W_from_localbody_arrays,
        MeshSamplesResult(
            se3_localbody_from_point_arrays=se3_arrays,
        ),
    )


def _sample_indices(num_total: int, num_samples: int) -> list[int]:
    """Return *num_samples* evenly-spaced indices from ``[0, num_total)``."""
    if num_total <= num_samples or num_samples <= 1:
        return list(range(min(num_total, max(num_samples, 1))))
    return [int(round(i * (num_total - 1) / (num_samples - 1))) for i in range(num_samples)]


# ---------------------------------------------------------------------------
# Colourisation helpers
# ---------------------------------------------------------------------------


def _colorize_depth(depth_hw: np.ndarray, cmap_name: str = "Spectral_r") -> np.ndarray:
    """Normalize a depth map to [0, 1] and apply a colormap.

    Returns:
        (H, W, 3) uint8 array.
    """
    depth_norm_hw = depth_hw.copy().astype(np.float64)
    finite_mask = np.isfinite(depth_norm_hw)
    if finite_mask.any():
        d_min = depth_norm_hw[finite_mask].min()
        d_max = depth_norm_hw[finite_mask].max()
        depth_norm_hw[~finite_mask] = d_max
    else:
        d_min, d_max = 0.0, 1.0

    if d_max - d_min < 1e-8:
        depth_norm_hw = np.zeros_like(depth_norm_hw)
    else:
        depth_norm_hw = (depth_norm_hw - d_min) / (d_max - d_min)

    return _apply_colormap(depth_norm_hw, cmap_name)


def _colorize_normals(normals_hw3: np.ndarray) -> np.ndarray:
    """Map surface normals from [-1, 1] to [0, 255] for visualization.

    Returns:
        (H, W, 3) uint8 array.
    """
    n = normals_hw3.copy().astype(np.float64)
    n = np.nan_to_num(n, nan=0.0)
    return ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def _colorize_flow_fast(flow_hw2: np.ndarray) -> np.ndarray:
    """Vectorised HSV-based optical flow visualization.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    flow_x_hw = flow_hw2[..., 0].astype(np.float64)
    flow_y_hw = flow_hw2[..., 1].astype(np.float64)
    magnitude_hw = np.sqrt(flow_x_hw**2 + flow_y_hw**2)
    angle_rad_hw = np.arctan2(flow_y_hw, flow_x_hw)

    magnitude_max = magnitude_hw.max() if magnitude_hw.max() > 0 else 1.0

    hsv_hw3 = np.zeros((*flow_hw2.shape[:2], 3), dtype=np.float64)
    hsv_hw3[..., 0] = (angle_rad_hw + np.pi) / (2 * np.pi)
    hsv_hw3[..., 1] = 1.0
    hsv_hw3[..., 2] = magnitude_hw / magnitude_max

    rgb_hw3 = (_hsv_to_rgb(hsv_hw3) * 255).clip(0, 255).astype(np.uint8)
    return rgb_hw3


def _colorize_flow3d(flow3d_hw3: np.ndarray, min_scale_m: float = 1e-3) -> np.ndarray:
    """Colorize 3D scene flow by displacement magnitude.

    Args:
        flow3d_hw3: (H, W, 3) float32 array of 3D displacement vectors (metres).
        min_scale_m: Minimum normalisation denominator (metres).  Prevents
            sub-millimetre floating-point noise from being amplified into
            visible colours by the colormap.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    flow3d_clean_hw3 = np.nan_to_num(flow3d_hw3, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude_hw = np.linalg.norm(flow3d_clean_hw3, axis=-1)
    magnitude_max = max(np.nanmax(magnitude_hw) if magnitude_hw.size else 0.0, min_scale_m)
    magnitude_norm_hw = magnitude_hw / magnitude_max

    return _apply_colormap(magnitude_norm_hw, "inferno")


# Flow 3D track type legend (match writer's _TRACK_TYPE_COLORS)
_FLOW3D_TRACK_TYPE_LEGEND = [
    (tuple(c / 255.0 for c in (128, 128, 128)), "Static"),
    (tuple(c / 255.0 for c in (0, 160, 255)), "Rigid"),
    (tuple(c / 255.0 for c in (255, 160, 0)), "Articulation"),
    (tuple(c / 255.0 for c in (255, 0, 0)), "Unsupported"),
]


def _build_flow3d_track_type_legend() -> list[dict]:
    """Build legend entries for Flow 3D track types."""
    return [{"color": color, "label": label} for color, label in _FLOW3D_TRACK_TYPE_LEGEND]


def _build_semantic_legend(semantic_info: dict | None) -> list[dict]:
    """Build legend entries from the semantic JSON metadata."""
    if semantic_info is None:
        return []
    objects = semantic_info.get("objects", [])
    legends: list[dict] = []
    for obj in objects:
        rgba = obj.get("rgba", (128, 128, 128, 255))
        color = tuple(c / 255.0 for c in rgba[:3])
        name = obj.get("object_name", obj.get("class_name", "unknown"))
        label = f"{name} ({obj.get('pixel_count', '?')} px)"
        legends.append({"color": color, "label": label})
    return legends


# ---------------------------------------------------------------------------
# Public API --image grids
# ---------------------------------------------------------------------------


def visualize_rgb_grid(
    hdf5_path: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    r"""Show a grid of sampled RGB frames (e.g. camera_id=\"cam0\")."""
    data = _load_frames(hdf5_path, camera_id)
    rgbs, frame_indices = data["rgbs"], data["frame_indices"]
    if not rgbs:
        print("[visualize_rgb_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    num_items = len(sample_ids)
    ncols = min(num_items, 4)
    nrows = (num_items + ncols - 1) // ncols

    cells: list[list[np.ndarray]] = []
    for r in range(nrows):
        row: list[np.ndarray] = []
        for c in range(ncols):
            i = r * ncols + c
            if i < num_items:
                idx = sample_ids[i]
                row.append(_add_title_bar(_ensure_rgb(rgbs[idx]), f"Frame {frame_indices[idx]}"))
            else:
                h, w = rgbs[0].shape[:2]
                row.append(np.full((h, w, 3), 255, dtype=np.uint8))
        cells.append(row)

    grid_hw3 = _add_suptitle_bar(_tile_images(cells), "Sampled RGB Frames")
    _save_or_show(grid_hw3, save_path, "Sampled RGB Frames", "visualize_rgb_grid")


def visualize_depth_grid(
    hdf5_path: str,
    camera_id: str,
    num_samples: int = 8,
    cmap: str = "Spectral_r",
    save_path: str | None = None,
) -> None:
    """Show colorized depth images alongside their RGB counterparts."""
    data = _load_frames(hdf5_path, camera_id)
    rgbs, depths, frame_indices = data["rgbs"], data["depths"], data["frame_indices"]
    if not rgbs:
        print("[visualize_depth_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    cells: list[list[np.ndarray]] = []
    for idx in sample_ids:
        rgb_cell = _add_title_bar(_ensure_rgb(rgbs[idx]), f"RGB -Frame {frame_indices[idx]}")
        depth_cell = _add_title_bar(
            _colorize_depth(depths[idx], cmap_name=cmap),
            f"Depth -Frame {frame_indices[idx]}",
        )
        cells.append([rgb_cell, depth_cell])

    grid_hw3 = _add_suptitle_bar(_tile_images(cells), "RGB & Colorized Depth")
    _save_or_show(grid_hw3, save_path, "RGB & Colorized Depth", "visualize_depth_grid")


def visualize_normals_grid(
    hdf5_path: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show RGB alongside colorized surface normals."""
    data = _load_frames(hdf5_path, camera_id)
    rgbs, normals_list, frame_indices = data["rgbs"], data["normals"], data["frame_indices"]
    if not rgbs:
        print("[visualize_normals_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    valid = [i for i in sample_ids if normals_list[i] is not None]
    if not valid:
        print("[visualize_normals_grid] No normal maps found.")
        return

    cells: list[list[np.ndarray]] = []
    for idx in valid:
        rgb_cell = _add_title_bar(_ensure_rgb(rgbs[idx]), f"RGB -Frame {frame_indices[idx]}")
        nrm_cell = _add_title_bar(
            _colorize_normals(normals_list[idx]),
            f"Normals -Frame {frame_indices[idx]}",
        )
        cells.append([rgb_cell, nrm_cell])

    grid_hw3 = _add_suptitle_bar(_tile_images(cells), "RGB & Surface Normals")
    _save_or_show(grid_hw3, save_path, "RGB & Surface Normals", "visualize_normals_grid")


def visualize_optical_flow_grid(
    hdf5_path: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show RGB alongside HSV-colorized optical flow."""
    data = _load_frames(hdf5_path, camera_id)
    rgbs, flows, frame_indices = data["rgbs"], data["optical_flows"], data["frame_indices"]
    if not rgbs:
        print("[visualize_optical_flow_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    valid = [i for i in sample_ids if flows[i] is not None]
    if not valid:
        print("[visualize_optical_flow_grid] No optical flow data found.")
        return

    cells: list[list[np.ndarray]] = []
    for idx in valid:
        rgb_cell = _add_title_bar(_ensure_rgb(rgbs[idx]), f"RGB -Frame {frame_indices[idx]}")
        flow_cell = _add_title_bar(
            _colorize_flow_fast(flows[idx]),
            f"Optical Flow -Frame {frame_indices[idx]}",
        )
        cells.append([rgb_cell, flow_cell])

    grid_hw3 = _add_suptitle_bar(_tile_images(cells), "RGB & Dense Optical Flow")
    _save_or_show(grid_hw3, save_path, "RGB & Dense Optical Flow", "visualize_optical_flow_grid")


def visualize_semantic_segmentation_grid(
    hdf5_path: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show RGB alongside semantic segmentation with a per-object legend.

    The legend is built from the JSON metadata stored in the HDF5 file.
    Each entry shows the object colour, class name, and pixel count.
    """
    data = _load_frames(hdf5_path, camera_id)
    rgbs = data["rgbs"]
    semantics = data["semantics"]
    semantics_infos = data["semantic_infos"]
    frame_indices = data["frame_indices"]
    if not rgbs:
        print("[visualize_semantic_segmentation_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    valid = [i for i in sample_ids if semantics[i] is not None]
    if not valid:
        print("[visualize_semantic_segmentation_grid] No semantic segmentation data found.")
        return

    cells: list[list[np.ndarray]] = []
    for idx in valid:
        rgb_cell = _add_title_bar(_ensure_rgb(rgbs[idx]), f"RGB -Frame {frame_indices[idx]}")

        semantics_rgba_hw4 = semantics[idx]
        semantics_rgb_hw3 = _ensure_rgb(semantics_rgba_hw4)
        legend = _build_semantic_legend(semantics_infos[idx] if idx < len(semantics_infos) else None)
        if legend:
            semantics_rgb_hw3 = _overlay_legend(semantics_rgb_hw3, legend)
        semantics_cell = _add_title_bar(semantics_rgb_hw3, f"Semantic -Frame {frame_indices[idx]}")

        cells.append([rgb_cell, semantics_cell])

    grid_hw3 = _add_suptitle_bar(_tile_images(cells), "RGB & Semantic Segmentation")
    _save_or_show(
        grid_hw3,
        save_path,
        "RGB & Semantic Segmentation",
        "visualize_semantic_segmentation_grid",
    )


# ---------------------------------------------------------------------------
# 3-D drawing & sampling helpers
# ---------------------------------------------------------------------------


def _draw_frustum(
    intrinsics_33: np.ndarray,
    T_W_from_C: np.ndarray,
    depth_m: float,
    _color: str = "royalblue",
    _linewidth: float = 0.6,
) -> tuple[list[float], list[float], list[float]]:
    """Build wireframe camera frustum as NaN-separated line segments.

    Returns:
        ``(xs, ys, zs)`` coordinate lists for a single Scatter3d trace.
    """
    fx, fy = intrinsics_33[0, 0], intrinsics_33[1, 1]
    cx, cy = intrinsics_33[0, 2], intrinsics_33[1, 2]

    corners_px = np.array(
        [
            [0, 0],
            [2 * cx, 0],
            [2 * cx, 2 * cy],
            [0, 2 * cy],
        ],
        dtype=np.float64,
    )

    corners_C_n3 = np.zeros((4, 3))
    for i, (u, v) in enumerate(corners_px):
        corners_C_n3[i] = [(u - cx) / fx * depth_m, (v - cy) / fy * depth_m, depth_m]

    rotation_W_from_C = T_W_from_C[:3, :3]
    translation_W_from_C = T_W_from_C[:3, 3]
    corners_W_n3 = (rotation_W_from_C @ corners_C_n3.T).T + translation_W_from_C
    apex_W_3 = translation_W_from_C

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for corner_world in corners_W_n3:
        xs.extend([apex_W_3[0], corner_world[0], float("nan")])
        ys.extend([apex_W_3[1], corner_world[1], float("nan")])
        zs.extend([apex_W_3[2], corner_world[2], float("nan")])

    for i in range(4):
        j = (i + 1) % 4
        xs.extend([corners_W_n3[i, 0], corners_W_n3[j, 0], float("nan")])
        ys.extend([corners_W_n3[i, 1], corners_W_n3[j, 1], float("nan")])
        zs.extend([corners_W_n3[i, 2], corners_W_n3[j, 2], float("nan")])

    return xs, ys, zs


def _set_equal_aspect_3d(points_n3: np.ndarray) -> dict:
    """Compute equal-aspect 3-D bounds.

    Returns:
        Dict with ``xlim``, ``ylim``, ``zlim`` tuples for axis ranges.
    """
    mid = points_n3.mean(axis=0)
    span = (points_n3.max(axis=0) - points_n3.min(axis=0)).max() / 2
    span = max(span, 0.1)
    return {
        "xlim": (mid[0] - span, mid[0] + span),
        "ylim": (mid[1] - span, mid[1] + span),
        "zlim": (mid[2] - span, mid[2] + span),
    }


def _fps_subsample(points_n3: np.ndarray, num_samples: int) -> np.ndarray:
    """Return *num_samples* indices into *points_n3* chosen by farthest-point sampling."""
    num_points = points_n3.shape[0]
    if num_samples >= num_points:
        return np.arange(num_points)
    selected = np.empty(num_samples, dtype=np.int64)
    selected[0] = 0
    min_dist_sq_n = np.full(num_points, np.inf)
    for i in range(1, num_samples):
        diff_n3 = points_n3 - points_n3[selected[i - 1]]
        dist_sq_n = np.einsum("ij,ij->i", diff_n3, diff_n3)
        np.minimum(min_dist_sq_n, dist_sq_n, out=min_dist_sq_n)
        selected[i] = np.argmax(min_dist_sq_n)
    return selected


# ---------------------------------------------------------------------------
# DatagenVisualizer
# ---------------------------------------------------------------------------


class DatagenVisualizer:  # pragma: no cover
    """Orchestrates all data generation visualizations.

    Stores shared configuration (output directory, camera IDs, etc.) and
    exposes one method per visualization type.  Call :meth:`generate_all`
    to produce every visualization, or invoke individual methods for
    selective rendering.
    """

    def __init__(
        self,
        output_dir: str,
        camera_ids: list[str],
        num_visualization_samples: int = 8,
        scene_flow_visualization_frame: int = 0,
        num_steps: int = 30,
        *,
        hdf5_path: str,
    ) -> None:
        """Initialize the visualizer with output directory and camera settings.

        Args:
            output_dir: Root output directory (used for saving visualizations).
            camera_ids: List of camera identifiers.
            num_visualization_samples: Number of sampled frames in grid views.
            scene_flow_visualization_frame: Frame index for the 3D scene-flow plot.
            num_steps: Total number of simulation steps.
            hdf5_path: Path to the HDF5 dataset file to load data from.
        """
        self._output_dir = output_dir
        self._camera_ids = camera_ids
        self._num_visualization_samples = num_visualization_samples
        self._scene_flow_visualization_frame = scene_flow_visualization_frame
        self._num_steps = num_steps
        self._hdf5_path = hdf5_path

    def _load_camera_frames(self, camera_id: str) -> dict:
        """Load frames from the HDF5 dataset."""
        return _load_frames(self._hdf5_path, camera_id)

    def _cam_visualization_dir(self, cam_id: str) -> str:
        """Return (and create) the visualization directory for a camera."""
        visualization_dir = os.path.join(self._output_dir, "visualizations", cam_id)
        os.makedirs(visualization_dir, exist_ok=True)
        return visualization_dir

    def all_modalities_grid(
        self,
        camera_id: str,
        num_samples: int = 8,
        depth_cmap: str = "Spectral_r",
        save_path: str | None = None,
    ) -> None:
        """Single figure: for each sampled frame show all modalities in one row.

        Column order (left to right): Color | Depth | Normals | Track Type |
        Semantics | Flow 2D | Flow 3D.
        Missing data is shown as a dark placeholder with 'N/A'.
        """
        data = self._load_camera_frames(camera_id)
        rgbs = data["rgbs"]
        depths = data["depths"]
        flows = data["optical_flows"]
        flows_3d = data["scene_flows_3d"]
        flow3d_track_types = data["flow3d_track_types"]
        normals_list = data["normals"]
        semantics = data["semantics"]
        semantics_infos = data["semantic_infos"]
        frame_indices = data["frame_indices"]

        if not rgbs:
            print("[visualize_all_modalities_grid] No frames found.")
            return

        sample_ids = _sample_indices(len(rgbs), num_samples)

        col_titles = [
            "Color",
            "Depth",
            "Normals",
            "Track Type",
            "Semantics",
            "Flow 2D",
            "Flow 3D",
        ]

        cells: list[list[np.ndarray]] = []
        for i, idx in enumerate(sample_ids):
            frame_idx = frame_indices[idx]
            h, w = rgbs[idx].shape[:2]
            blank_hw3 = _blank_with_na(h, w)
            row: list[np.ndarray] = []

            rgb_cell = _overlay_frame_label(_ensure_rgb(rgbs[idx]), f"Frame {frame_idx}")
            row.append(rgb_cell)

            row.append(
                _colorize_depth(depths[idx], cmap_name=depth_cmap) if depths[idx] is not None else blank_hw3.copy()
            )

            row.append(_colorize_normals(normals_list[idx]) if normals_list[idx] is not None else blank_hw3.copy())

            track_type_hw = flow3d_track_types[idx] if idx < len(flow3d_track_types) else None
            if track_type_hw is not None:
                track_type_cell = _ensure_rgb(track_type_hw)
                track_type_cell = _overlay_legend(track_type_cell, _build_flow3d_track_type_legend())
                row.append(track_type_cell)
            else:
                row.append(blank_hw3.copy())

            if semantics[idx] is not None:
                semantics_cell = _ensure_rgb(semantics[idx])
                semantics_info_i = semantics_infos[idx] if idx < len(semantics_infos) else None
                legend = _build_semantic_legend(semantics_info_i)
                if legend:
                    semantics_cell = _overlay_legend(semantics_cell, legend)
                row.append(semantics_cell)
            else:
                row.append(blank_hw3.copy())

            row.append(_colorize_flow_fast(flows[idx]) if flows[idx] is not None else blank_hw3.copy())

            row.append(_colorize_flow3d(flows_3d[idx]) if flows_3d[idx] is not None else blank_hw3.copy())

            if i == 0:
                row = [_add_title_bar(cell, title) for cell, title in zip(row, col_titles)]

            cells.append(row)

        grid_hw3 = _tile_images(cells)
        _save_or_show(grid_hw3, save_path, "", "visualize_all_modalities_grid")

    def all_modalities_video(  # pylint: disable=too-many-statements
        self,
        camera_id: str,
        fps: int = 5,
        depth_cmap: str = "Spectral_r",
        save_path: str | None = None,
    ) -> None:
        """Render one video frame per timestep with all modalities in a grid.

        Layout (4 columns):
            Row 0: Color | Depth | Normals | Track Type
            Row 1: Semantics | Flow 2D | Flow 3D

        The video runs at *fps* frames per second and is saved as an mp4.

        Args:
            output_dir: Root output directory containing camera sub-directories.
            camera_id: Camera folder name (e.g. ``"cam0"``).
            fps: Playback frame rate (default 5).
            depth_cmap: Colormap name used for depth (``"Spectral_r"``).
            save_path: Destination ``.mp4`` path.  If ``None``, defaults to
                ``<output_dir>/visualizations/<camera_id>/data_vis.mp4``.
        """
        data = self._load_camera_frames(camera_id)
        rgbs = data["rgbs"]
        if not rgbs:
            print("[visualize_all_modalities_video] No frames found.")
            return

        depths = data["depths"]
        flows = data["optical_flows"]
        flows_3d = data["scene_flows_3d"]
        flow3d_track_types = data["flow3d_track_types"]
        normals_list = data["normals"]
        semantics = data["semantics"]
        frame_indices = data["frame_indices"]

        row0_titles = ["Color", "Depth", "Normals", "Track Type"]
        row1_titles = ["Semantics", "Flow 2D", "Flow 3D"]

        if save_path is None:
            visualization_dir = os.path.join(self._output_dir, "visualizations", camera_id)
            os.makedirs(visualization_dir, exist_ok=True)
            save_path = os.path.join(visualization_dir, "data_vis.mp4")

        frames_list_hw3: list[np.ndarray] = []
        for idx in range(len(rgbs)):  # pylint: disable=consider-using-enumerate  # indexing parallel lists
            frame_idx = frame_indices[idx]
            h, w = rgbs[idx].shape[:2]
            blank_hw3 = np.full((h, w, 3), 40, dtype=np.uint8)

            top_row_images: list[np.ndarray] = []
            top_row_images.append(_overlay_frame_label(_ensure_rgb(rgbs[idx]), f"Frame {frame_idx}"))
            top_row_images.append(
                _colorize_depth(depths[idx], cmap_name=depth_cmap) if depths[idx] is not None else blank_hw3
            )
            top_row_images.append(_colorize_normals(normals_list[idx]) if normals_list[idx] is not None else blank_hw3)
            track_type_hw = flow3d_track_types[idx] if idx < len(flow3d_track_types) else None
            top_row_images.append(_ensure_rgb(track_type_hw) if track_type_hw is not None else blank_hw3)

            bottom_row_images: list[np.ndarray] = []
            semantics_hw4 = semantics[idx] if idx < len(semantics) else None
            bottom_row_images.append(_ensure_rgb(semantics_hw4) if semantics_hw4 is not None else blank_hw3)
            bottom_row_images.append(_colorize_flow_fast(flows[idx]) if flows[idx] is not None else blank_hw3)
            bottom_row_images.append(_colorize_flow3d(flows_3d[idx]) if flows_3d[idx] is not None else blank_hw3)

            top_row_images = [_add_title_bar(c, t) for c, t in zip(top_row_images, row0_titles)]
            bottom_row_images = [_add_title_bar(c, t) for c, t in zip(bottom_row_images, row1_titles)]

            frame_img_hw3 = _tile_images([top_row_images, bottom_row_images], gap=2, bg=255)
            if frame_img_hw3.shape[-1] == 4:
                frame_img_hw3 = frame_img_hw3[..., :3]
            frames_list_hw3.append(frame_img_hw3)

        h_max = max(f.shape[0] for f in frames_list_hw3)
        w_max = max(f.shape[1] for f in frames_list_hw3)
        h_max += h_max % 2
        w_max += w_max % 2
        uniform_hw3: list[np.ndarray] = []
        for f in frames_list_hw3:
            if f.shape[0] != h_max or f.shape[1] != w_max:
                padded_hw3 = np.full((h_max, w_max, 3), 255, dtype=np.uint8)
                padded_hw3[: f.shape[0], : f.shape[1]] = f
                uniform_hw3.append(padded_hw3)
            else:
                uniform_hw3.append(f)

        iio.imwrite(save_path, np.stack(uniform_hw3), fps=fps, codec="libx264", pixelformat="yuv420p")
        print(f"[visualize_all_modalities_video] Saved to {save_path}")

    def camera_trajectory(
        self,
        camera_id: str,
        axis_length_m: float = 0.05,
        frustum_scale_m: float = 0.04,
        num_frustums: int = 20,
        save_path: str | None = None,
    ) -> None:
        """Plot the camera trajectory in 3D with coordinate frames and frustum outlines.

        Saved as interactive HTML (plotly).
        """
        data = self._load_camera_frames(camera_id)
        intrinsics_list, T_W_from_C_list = data["intrinsics"], data["extrinsics"]
        if not T_W_from_C_list:
            print("[visualize_camera_trajectory] No frames found.")
            return

        positions_W_n3 = np.array([T[:3, 3] for T in T_W_from_C_list])
        traces: list = []

        # Trajectory line
        traces.append(
            go.Scatter3d(
                x=positions_W_n3[:, 0],
                y=positions_W_n3[:, 1],
                z=positions_W_n3[:, 2],
                mode="lines",
                line={"color": "gray", "width": 2},
                name="trajectory",
            )
        )

        # Start and end markers
        traces.append(
            go.Scatter3d(
                x=[positions_W_n3[0, 0]],
                y=[positions_W_n3[0, 1]],
                z=[positions_W_n3[0, 2]],
                mode="markers",
                marker={"size": 6, "color": "green"},
                name="start",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[positions_W_n3[-1, 0]],
                y=[positions_W_n3[-1, 1]],
                z=[positions_W_n3[-1, 2]],
                mode="markers",
                marker={"size": 6, "color": "red"},
                name="end",
            )
        )

        # Coordinate frame axes + frustums at sampled frames
        frame_sample_ids = _sample_indices(len(T_W_from_C_list), num_frustums)
        axis_colors = ["red", "green", "blue"]

        for sid in frame_sample_ids:
            T_W_from_C = T_W_from_C_list[sid]
            rotation_W_from_C = T_W_from_C[:3, :3]
            translation_W_from_C = T_W_from_C[:3, 3]

            for col, color in enumerate(axis_colors):
                direction = rotation_W_from_C[:, col] * axis_length_m
                end = translation_W_from_C + direction
                traces.append(
                    go.Scatter3d(
                        x=[translation_W_from_C[0], end[0]],
                        y=[translation_W_from_C[1], end[1]],
                        z=[translation_W_from_C[2], end[2]],
                        mode="lines",
                        line={"color": color, "width": 3},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            intrinsics_33 = intrinsics_list[sid]
            xs_frustum, ys_frustum, zs_frustum = _draw_frustum(intrinsics_33, T_W_from_C, frustum_scale_m)
            traces.append(
                go.Scatter3d(
                    x=xs_frustum,
                    y=ys_frustum,
                    z=zs_frustum,
                    mode="lines",
                    line={"color": "royalblue", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        bounds = _set_equal_aspect_3d(positions_W_n3)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Camera Trajectory & Frames",
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis_title": "Z (m)",
                "xaxis": {"range": list(bounds["xlim"])},
                "yaxis": {"range": list(bounds["ylim"])},
                "zaxis": {"range": list(bounds["zlim"])},
                "aspectmode": "cube",
            },
            legend={"x": 0, "y": 1},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )

        if save_path:
            fig.write_html(save_path)
            print(f"[visualize_camera_trajectory] Saved to {save_path}")
        else:
            fig.show()

    def scene_flow_3d(  # pylint: disable=too-many-statements
        self,
        camera_id: str,
        frame_index: int = 0,
        stride: int = 8,
        arrow_scale: float = 1.0,
        point_size: float = 3.0,
        line_width: float = 3.0,
        flow_threshold_m: float = 1e-5,
        save_path: str | None = None,
    ) -> None:
        """Interactive 3-D visualisation of per-pixel scene flow on a point cloud.

        Every *stride*-th pixel is unprojected to its 3-D world position and
        rendered as a dot (coloured by the original RGB image).  Pixels with
        non-zero flow additionally get a **line** starting at the dot, whose
        length and direction match the 3-D flow vector.

        Line colour encodes *direction* by mapping the normalised 3-D flow
        vector to RGB (X -> red, Y -> green, Z -> blue) so that different
        directions receive visually distinct colours while similar directions
        remain close.  *Magnitude* modulates lightness: small motions produce
        light/pastel lines, large motions produce dark lines of the same hue.

        The result is a fully rotatable 3-D figure (saved as an ``.html`` file
        when *save_path* is given, or displayed interactively otherwise).

        Args:
            output_dir: Root output directory containing camera sub-directories.
            camera_id: Camera folder name (e.g. ``"cam0"``).
            frame_index: Which frame to visualise (matches the filename index).
            stride: Sub-sample every *stride* pixels in each spatial dimension.
            arrow_scale: Global multiplier applied to line lengths.
            point_size: Marker size for the 3-D dots.
            line_width: Line width for flow lines.
            flow_threshold_m: Minimum flow magnitude to draw a line.
            save_path: Path to save an interactive ``.html`` file.  If ``None``,
                the figure is displayed in the default browser / notebook.
        """
        # ---- Load single-frame data ----------------------------------------
        data = self._load_camera_frames(camera_id)
        if not data["rgbs"] or frame_index >= len(data["rgbs"]):
            print(f"[visualize_scene_flow_3d] No data for frame {frame_index}")
            return

        depth_hw = np.asarray(data["depths"][frame_index])
        intrinsics_33 = np.asarray(data["intrinsics"][frame_index]).reshape(3, 3).astype(np.float64)
        T_W_from_C = np.asarray(data["extrinsics"][frame_index]).reshape(4, 4).astype(np.float64)

        flow3d = data["scene_flows_3d"][frame_index] if frame_index < len(data["scene_flows_3d"]) else None
        if flow3d is None:
            print(f"[visualize_scene_flow_3d] No flow3d for frame {frame_index}")
            return
        flow3d_hw3 = np.asarray(flow3d).astype(np.float64)
        rgb_hw3 = np.asarray(data["rgbs"][frame_index])

        H, W = depth_hw.shape

        # ---- Unproject depth -> 3-D world coordinates -----------------------
        v_C_hw, u_C_hw = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        fx, fy = intrinsics_33[0, 0], intrinsics_33[1, 1]
        cx, cy = intrinsics_33[0, 2], intrinsics_33[1, 2]

        depth64_hw = depth_hw.astype(np.float64)
        depth64_hw[~np.isfinite(depth64_hw)] = 0.0
        x_C_hw = (u_C_hw.astype(np.float64) - cx) / fx * depth64_hw
        y_C_hw = (v_C_hw.astype(np.float64) - cy) / fy * depth64_hw
        z_C_hw = depth64_hw
        points_C_hw3 = np.stack([x_C_hw, y_C_hw, z_C_hw], axis=-1)  # (H, W, 3)

        rotation_W_from_C = T_W_from_C[:3, :3]
        translation_W_from_C = T_W_from_C[:3, 3]
        points_W_hw3 = (rotation_W_from_C @ points_C_hw3.reshape(-1, 3).T).T.reshape(H, W, 3) + translation_W_from_C

        # ---- Compensate camera ego-motion: convert stored camera-relative
        #      flow to world-space object-only flow so that static surfaces
        #      have zero-length arrows regardless of camera motion. -----------
        dst_frame = frame_index + 1
        if dst_frame < len(data["extrinsics"]) and data["extrinsics"][dst_frame] is not None:
            T_W_from_C_dst = np.asarray(data["extrinsics"][dst_frame]).reshape(4, 4).astype(np.float64)
            rotation_W_from_C_dst = T_W_from_C_dst[:3, :3]
            translation_W_from_C_dst = T_W_from_C_dst[:3, 3]
            points_src_C_n3 = (points_W_hw3.reshape(-1, 3) - translation_W_from_C) @ rotation_W_from_C
            points_dst_C_n3 = points_src_C_n3 + flow3d_hw3.reshape(-1, 3)
            points_dst_W_n3 = (points_dst_C_n3 @ rotation_W_from_C_dst.T) + translation_W_from_C_dst
            flow3d_hw3 = (points_dst_W_n3 - points_W_hw3.reshape(-1, 3)).reshape(H, W, 3)

        # ---- Sub-sample & flatten ------------------------------------------
        points_W_n3 = points_W_hw3[::stride, ::stride]
        flow3d_sub_hw3 = flow3d_hw3[::stride, ::stride]
        rgb_sub_hw3 = rgb_hw3[::stride, ::stride]
        depth_sub_hw = depth_hw[::stride, ::stride]

        points_flat_W_n3 = points_W_n3.reshape(-1, 3)
        flow3d_flat_n3 = flow3d_sub_hw3.reshape(-1, 3)
        rgb_flat_n3 = rgb_sub_hw3.reshape(-1, 3)
        depth_flat_n = depth_sub_hw.reshape(-1)

        valid = np.isfinite(depth_flat_n) & np.all(np.isfinite(points_flat_W_n3), axis=-1)
        points_valid_n3 = points_flat_W_n3[valid]
        flow_valid_n3 = flow3d_flat_n3[valid]
        colors_valid_n3 = rgb_flat_n3[valid]

        magnitude_n = np.linalg.norm(flow_valid_n3, axis=-1)
        has_flow = magnitude_n > flow_threshold_m

        # ---- Dynamic points (non-zero flow) --------------------------------
        points_dynamic_W_n3 = points_valid_n3[has_flow]
        flow3d_dynamic_n3 = flow_valid_n3[has_flow]
        magnitude_dynamic_n = magnitude_n[has_flow]
        num_dynamic = len(points_dynamic_W_n3)

        if num_dynamic > 0:
            flow_direction_n3 = flow3d_dynamic_n3 / (magnitude_dynamic_n[:, np.newaxis] + 1e-8)
            base_rgb_n3 = (flow_direction_n3 + 1.0) / 2.0

            magnitude_max = magnitude_dynamic_n.max()
            magnitude_norm_n = magnitude_dynamic_n / (magnitude_max + 1e-8)
            t_blend = magnitude_norm_n[:, np.newaxis]
            white_offset = (1.0 - t_blend) * 0.55
            dark_scale = 1.0 - 0.5 * t_blend
            arrow_rgb = np.clip(base_rgb_n3 * dark_scale + white_offset, 0, 1)

            points_end_W_n3 = points_dynamic_W_n3 + flow3d_dynamic_n3 * arrow_scale

            line_x_n = np.empty(3 * num_dynamic)
            line_y_n = np.empty(3 * num_dynamic)
            line_z_n = np.empty(3 * num_dynamic)

            line_x_n[0::3] = points_dynamic_W_n3[:, 0]
            line_x_n[1::3] = points_end_W_n3[:, 0]
            line_x_n[2::3] = np.nan

            line_y_n[0::3] = points_dynamic_W_n3[:, 1]
            line_y_n[1::3] = points_end_W_n3[:, 1]
            line_y_n[2::3] = np.nan

            line_z_n[0::3] = points_dynamic_W_n3[:, 2]
            line_z_n[1::3] = points_end_W_n3[:, 2]
            line_z_n[2::3] = np.nan

            line_rgb = np.repeat(arrow_rgb, 3, axis=0)
            line_colors = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in line_rgb]

        # ---- Build plotly figure -------------------------------------------
        traces = []

        all_colors = [f"rgb({r},{g},{b})" for r, g, b in colors_valid_n3]
        traces.append(
            go.Scatter3d(
                x=points_valid_n3[:, 0],
                y=points_valid_n3[:, 1],
                z=points_valid_n3[:, 2],
                mode="markers",
                marker={"size": point_size, "color": all_colors, "opacity": 0.4},
                name="Point cloud",
                hoverinfo="skip",
            )
        )

        if num_dynamic > 0:
            traces.append(
                go.Scatter3d(
                    x=line_x_n,
                    y=line_y_n,
                    z=line_z_n,
                    mode="lines",
                    line={"color": line_colors, "width": line_width},
                    name="Flow vectors",
                    hoverinfo="skip",
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"3D Scene Flow -- Frame {frame_index} ({camera_id})",
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis_title": "Z (m)",
                "aspectmode": "data",
            },
            legend={"x": 0, "y": 1},
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )

        if save_path:
            fig.write_html(save_path)
            print(f"[visualize_scene_flow_3d] Saved to {save_path}")
        else:
            fig.show()

    def dynamic_mesh_trajectories(  # pylint: disable=too-many-statements,too-many-branches
        self,
        step_stride: int = 5,
        point_stride: int = 8,
        sphere_size: float = 3.0,
        line_width: float = 2.0,
        normal_length_m: float = 0.01,
        save_path: str | None = None,
    ) -> None:
        """Interactive 3-D visualisation of dynamic-object mesh point trajectories.

        Loads the mesh samples and per-step poses, reconstructs world-space
        positions at every *step_stride*-th step, and plots a sub-sample (every
        *point_stride*-th point) as spheres connected by trajectory lines.

        Each object gets a distinct colour.  Spheres are drawn at every shown
        step; consecutive steps are joined by lines of the same colour.

        Args:
            output_dir: Root output directory (same as ``OUTPUT_DIR``).
            step_stride: Show every N-th time step.
            point_stride: Sub-sample factor for points (e.g. 8 = show 1/8).
            sphere_size: Marker size for point spheres.
            line_width: Width of trajectory lines.
            normal_length_m: Length (metres) of normal-vector lines.
            save_path: Path to save an ``.html`` file.
        """
        num_steps, T_W_from_localbody_arrays, mesh_samples = _load_dynamic_objects_from_hdf5(self._hdf5_path)
        if num_steps is None:
            print("[visualize_dynamic_mesh_trajectories] No dynamic objects in HDF5, skipping.")
            return

        shown_steps = list(range(0, num_steps, step_stride))
        if shown_steps[-1] != num_steps - 1:
            shown_steps.append(num_steps - 1)

        all_keys = sorted(mesh_samples.se3_localbody_from_point_arrays.keys())
        if not all_keys:
            print("[visualize_dynamic_mesh_trajectories] No mesh samples found.")
            return

        key_colors = {}
        for i, key in enumerate(all_keys):
            hue = (0.1 + 0.6180339887498948 * i) % 1.0  # golden ratio conjugate for maximal hue spacing
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            key_colors[key] = (int(r * 255), int(g * 255), int(b * 255))

        step_data: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for s in shown_steps:
            step_data[s] = reconstruct_mesh_points_at_step(mesh_samples, T_W_from_localbody_arrays, s)

        layer_initial = "initial"
        layer_trajectory = "trajectories"
        layer_norm_init = "normals_initial"
        layer_norm_trajectory = "normals_trajectory"

        traces: list = []
        trace_layers: list[str] = []

        first_step = shown_steps[0]
        for key in all_keys:
            if key not in step_data[first_step]:
                continue
            r, g, b = key_colors[key]

            all_points0_W_n3 = step_data[first_step][key][0]
            if all_points0_W_n3.shape[0] < 4:
                continue

            try:
                tri = Delaunay(all_points0_W_n3)
                simplices = tri.simplices
                i_idx = simplices[:, 0]
                j_idx = simplices[:, 1]
                k_idx = simplices[:, 2]
            except QhullError:
                continue

            traces.append(
                go.Mesh3d(
                    x=all_points0_W_n3[:, 0],
                    y=all_points0_W_n3[:, 1],
                    z=all_points0_W_n3[:, 2],
                    i=i_idx,
                    j=j_idx,
                    k=k_idx,
                    color=f"rgb({r},{g},{b})",
                    opacity=0.05,
                    name=f"{key} mesh t=0",
                    legendgroup=key,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            trace_layers.append(layer_initial)

        key_display_idx: dict[str, np.ndarray] = {}
        for key in all_keys:
            if key not in step_data[first_step]:
                continue
            points0_W_n3 = step_data[first_step][key][0]
            num_points = points0_W_n3.shape[0]
            num_display = max(1, num_points // point_stride)
            key_display_idx[key] = _fps_subsample(points0_W_n3, num_display)

        for key in all_keys:
            if key not in key_display_idx:
                continue
            r, g, b = key_colors[key]
            pt_indices_0 = key_display_idx[key]
            if len(pt_indices_0) == 0:
                continue
            points_sub0_W_n3 = step_data[first_step][key][0][pt_indices_0]
            dark_r, dark_g, dark_b = int(r * 0.5), int(g * 0.5), int(b * 0.5)
            traces.append(
                go.Scatter3d(
                    x=points_sub0_W_n3[:, 0],
                    y=points_sub0_W_n3[:, 1],
                    z=points_sub0_W_n3[:, 2],
                    mode="markers",
                    marker={
                        "size": sphere_size * 1.3,
                        "color": f"rgb({dark_r},{dark_g},{dark_b})",
                        "opacity": 1.0,
                    },
                    name=f"{key} samples t=0",
                    legendgroup=key,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            trace_layers.append(layer_initial)

            normals_sub0_W_n3 = step_data[first_step][key][1][pt_indices_0]
            tips0_W_n3 = points_sub0_W_n3 + normals_sub0_W_n3 * normal_length_m
            normal_line_x_n = np.empty(3 * len(points_sub0_W_n3))
            normal_line_y_n = np.empty(3 * len(points_sub0_W_n3))
            normal_line_z_n = np.empty(3 * len(points_sub0_W_n3))
            normal_line_x_n[0::3] = points_sub0_W_n3[:, 0]
            normal_line_x_n[1::3] = tips0_W_n3[:, 0]
            normal_line_x_n[2::3] = np.nan
            normal_line_y_n[0::3] = points_sub0_W_n3[:, 1]
            normal_line_y_n[1::3] = tips0_W_n3[:, 1]
            normal_line_y_n[2::3] = np.nan
            normal_line_z_n[0::3] = points_sub0_W_n3[:, 2]
            normal_line_z_n[1::3] = tips0_W_n3[:, 2]
            normal_line_z_n[2::3] = np.nan
            traces.append(
                go.Scatter3d(
                    x=normal_line_x_n,
                    y=normal_line_y_n,
                    z=normal_line_z_n,
                    mode="lines",
                    line={"color": f"rgb({dark_r},{dark_g},{dark_b})", "width": 1.5},
                    legendgroup=key,
                    showlegend=False,
                    hoverinfo="skip",
                    visible=False,
                )
            )
            trace_layers.append(layer_norm_init)

        num_shown = len(shown_steps)

        for key in all_keys:
            r, g, b = key_colors[key]

            if key not in key_display_idx:
                continue
            pt_indices = key_display_idx[key]
            num_subsampled = len(pt_indices)
            if num_subsampled == 0:
                continue

            for shown_step_idx, step_id in enumerate(shown_steps):
                if key not in step_data[step_id]:
                    continue
                points_W_n3, normals_W_n3 = step_data[step_id][key]
                points_sub_W_n3 = points_W_n3[pt_indices]
                normals_sub_W_n3 = normals_W_n3[pt_indices]

                time_fraction = shown_step_idx / max(num_shown - 1, 1)
                fade = 0.4 + 0.5 * time_fraction
                marker_color = f"rgba({r},{g},{b},{fade:.2f})"

                traces.append(
                    go.Scatter3d(
                        x=points_sub_W_n3[:, 0],
                        y=points_sub_W_n3[:, 1],
                        z=points_sub_W_n3[:, 2],
                        mode="markers",
                        marker={"size": sphere_size, "color": marker_color, "opacity": fade},
                        name=f"{key} t={step_id}",
                        legendgroup=key,
                        showlegend=(step_id == first_step),
                        hovertext=[f"{key} t={step_id} pt={i}" for i in pt_indices],
                        hoverinfo="text",
                    )
                )
                trace_layers.append(layer_trajectory)

                tips_W_n3 = points_sub_W_n3 + normals_sub_W_n3 * normal_length_m
                tip_line_x_n = np.empty(3 * num_subsampled)
                tip_line_y_n = np.empty(3 * num_subsampled)
                tip_line_z_n = np.empty(3 * num_subsampled)
                tip_line_x_n[0::3] = points_sub_W_n3[:, 0]
                tip_line_x_n[1::3] = tips_W_n3[:, 0]
                tip_line_x_n[2::3] = np.nan
                tip_line_y_n[0::3] = points_sub_W_n3[:, 1]
                tip_line_y_n[1::3] = tips_W_n3[:, 1]
                tip_line_y_n[2::3] = np.nan
                tip_line_z_n[0::3] = points_sub_W_n3[:, 2]
                tip_line_z_n[1::3] = tips_W_n3[:, 2]
                tip_line_z_n[2::3] = np.nan
                traces.append(
                    go.Scatter3d(
                        x=tip_line_x_n,
                        y=tip_line_y_n,
                        z=tip_line_z_n,
                        mode="lines",
                        line={"color": f"rgba({r},{g},{b},{fade:.2f})", "width": 1.5},
                        legendgroup=key,
                        showlegend=False,
                        hoverinfo="skip",
                        visible=False,
                    )
                )
                trace_layers.append(layer_norm_trajectory)

            line_color = f"rgba({r},{g},{b},0.35)"
            for s_prev, s_curr in zip(shown_steps[:-1], shown_steps[1:]):
                if key not in step_data[s_prev] or key not in step_data[s_curr]:
                    continue
                points_prev_W_n3 = step_data[s_prev][key][0][pt_indices]
                points_curr_W_n3 = step_data[s_curr][key][0][pt_indices]

                line_x_n = np.empty(3 * num_subsampled)
                line_y_n = np.empty(3 * num_subsampled)
                line_z_n = np.empty(3 * num_subsampled)

                line_x_n[0::3] = points_prev_W_n3[:, 0]
                line_x_n[1::3] = points_curr_W_n3[:, 0]
                line_x_n[2::3] = np.nan

                line_y_n[0::3] = points_prev_W_n3[:, 1]
                line_y_n[1::3] = points_curr_W_n3[:, 1]
                line_y_n[2::3] = np.nan

                line_z_n[0::3] = points_prev_W_n3[:, 2]
                line_z_n[1::3] = points_curr_W_n3[:, 2]
                line_z_n[2::3] = np.nan

                traces.append(
                    go.Scatter3d(
                        x=line_x_n,
                        y=line_y_n,
                        z=line_z_n,
                        mode="lines",
                        line={"color": line_color, "width": line_width},
                        legendgroup=key,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                trace_layers.append(layer_trajectory)

        num_traces = len(traces)
        initial_indices = [i for i in range(num_traces) if trace_layers[i] == layer_initial]
        trajectory_indices = [i for i in range(num_traces) if trace_layers[i] == layer_trajectory]
        norm_init_indices = [i for i in range(num_traces) if trace_layers[i] == layer_norm_init]
        norm_trajectory_indices = [i for i in range(num_traces) if trace_layers[i] == layer_norm_trajectory]

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Dynamic Object Mesh Point Trajectories",
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis_title": "Z (m)",
                "aspectmode": "data",
            },
            legend={"x": 0, "y": 1},
            margin={"l": 0, "r": 0, "t": 60, "b": 0},
        )

        _toggle_snippet = (
            """
    <style>
    .toggle-btn {
        padding: 6px 14px; margin: 4px 6px 4px 0; cursor: pointer;
        border: 2px solid #888; border-radius: 5px; font-size: 13px;
        font-weight: 600; user-select: none; display: inline-block;
    }
    .toggle-btn.on  { background: #d0eaff; border-color: #3388cc; color: #1a5276; }
    .toggle-btn.off { background: #f0f0f0; border-color: #bbb;    color: #888; }
    </style>
    <div style="text-align:left; padding:4px 8px; position:relative; z-index:1000;">
      <span id="btn-initial" class="toggle-btn on"
            onclick="toggleLayer('initial')">Meshes + Initial Points</span>
      <span id="btn-traj" class="toggle-btn on"
            onclick="toggleLayer('traj')">Points + Trajectories</span>
      <span id="btn-normals" class="toggle-btn off"
            onclick="toggleLayer('normals')">Normals</span>
    </div>
    <script>
    var layerState = {initial: true, traj: true, normals: false};
    var layerIndices = {
        initial:   __PH_INITIAL__,
        traj:      __PH_TRAJ__,
        normInit:  __PH_NORM_INIT__,
        normTraj:  __PH_NORM_TRAJ__
    };
    function applyVisibility() {
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        var showNormInit = layerState.normals && layerState.initial;
        var showNormTraj = layerState.normals && layerState.traj;
        var overrides = {};
        layerIndices.initial.forEach(function(i)  { overrides[i] = layerState.initial; });
        layerIndices.traj.forEach(function(i)     { overrides[i] = layerState.traj; });
        layerIndices.normInit.forEach(function(i) { overrides[i] = showNormInit; });
        layerIndices.normTraj.forEach(function(i) { overrides[i] = showNormTraj; });
        var visArr = [];
        for (var i = 0; i < gd.data.length; i++) {
            visArr.push(i in overrides ? overrides[i] : (gd.data[i].visible !== false));
        }
        Plotly.restyle(gd, {'visible': visArr});
    }
    function toggleLayer(name) {
        layerState[name] = !layerState[name];
        applyVisibility();
        var btn = document.getElementById('btn-' + name);
        btn.className = 'toggle-btn ' + (layerState[name] ? 'on' : 'off');
    }
    </script>
    """.replace("__PH_INITIAL__", json.dumps(initial_indices))
            .replace("__PH_TRAJ__", json.dumps(trajectory_indices))
            .replace("__PH_NORM_INIT__", json.dumps(norm_init_indices))
            .replace("__PH_NORM_TRAJ__", json.dumps(norm_trajectory_indices))
        )

        if save_path:
            html_str = fig.to_html(full_html=True, include_plotlyjs=True)
            html_str = html_str.replace("<body>", "<body>" + _toggle_snippet)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            print(f"[visualize_dynamic_mesh_trajectories] Saved to {save_path}")
        else:
            fig.show()

    def generate_all(self) -> None:
        """Render all grid, video, trajectory, and 3D flow visualizations.

        Produces per-camera visualizations (modality grids, videos, camera
        trajectory plots, scene-flow 3D views) as well as a combined
        dynamic-mesh trajectory visualization.
        """
        for cam_id in self._camera_ids:
            visualization_dir = os.path.join(self._output_dir, "visualizations", cam_id)
            os.makedirs(visualization_dir, exist_ok=True)
            print(f"-- {cam_id} --")

            self.all_modalities_grid(
                cam_id,
                num_samples=self._num_visualization_samples,
                depth_cmap="Spectral_r",
                save_path=os.path.join(visualization_dir, "data_vis.png"),
            )

            self.all_modalities_video(
                cam_id,
                fps=5,
                depth_cmap="Spectral_r",
                save_path=os.path.join(visualization_dir, "data_vis.mp4"),
            )

            self.camera_trajectory(
                cam_id,
                axis_length_m=0.05,
                frustum_scale_m=0.04,
                num_frustums=self._num_visualization_samples,
                save_path=os.path.join(visualization_dir, "camera_trajectory_3d.html"),
            )

            self.scene_flow_3d(
                cam_id,
                frame_index=self._scene_flow_visualization_frame,
                stride=8,
                arrow_scale=1.0,
                save_path=os.path.join(
                    visualization_dir,
                    f"scene_flow_3d_frame{self._scene_flow_visualization_frame}.html",
                ),
            )

            print(f"Visualizations saved to {visualization_dir}")

        dynamic_visualization_root = os.path.join(self._output_dir, "visualizations", "dynamic_objects")
        os.makedirs(dynamic_visualization_root, exist_ok=True)
        self.dynamic_mesh_trajectories(
            step_stride=1,
            point_stride=32,
            save_path=os.path.join(dynamic_visualization_root, "dynamic_mesh_trajectories.html"),
        )
