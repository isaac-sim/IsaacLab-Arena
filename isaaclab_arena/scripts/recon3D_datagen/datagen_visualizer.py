# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for recon3D data generation outputs.

Provides functions to:
- Combined grid: color, depth, flow2d, normals, semantics in one figure.
- Plot 3D camera trajectory with coordinate frames and optional frustums.
- Individual grids for RGB, depth, normals, optical flow, semantic segmentation.
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Subfolder names (must match IsaacLabArenaWriter)
_SUBFOLDER_COLOR = "color"
_SUBFOLDER_DEPTH = "depth"
_SUBFOLDER_FLOW2D = "flow2d"
_SUBFOLDER_NORMAL = "normal"
_SUBFOLDER_EXTRINSIC = "extrinsic"
_SUBFOLDER_INTRINSIC = "intrinsic"
_SUBFOLDER_SEMANTIC = "semantic"
_SUBFOLDER_FLOW3D = "flow3d"


def _load_frames(output_dir: str, camera_id: str) -> dict:
    """Load all frames written by :class:`IsaacLabArenaWriter`.

    Expects layout output_dir/{camera_id}/color, depth, ... (e.g. cam0/color).
    Filenames are 10-digit numeric (e.g. 0000000000.png).

    Returns:
        Dictionary with keys: rgbs, depths, intrinsics, extrinsics, normals,
        optical_flows, semantics, semantic_infos, frame_indices.
    """
    cam_dir = os.path.join(output_dir, camera_id)
    color_dir = os.path.join(cam_dir, _SUBFOLDER_COLOR)
    rgb_pattern = os.path.join(color_dir, "*.png")
    all_rgb = glob.glob(rgb_pattern)
    # Only 10-digit numeric filenames (0000000000.png, 0000000001.png, ...)
    rgb_paths = sorted(
        p for p in all_rgb if re.match(r"^\d{10}\.png$", os.path.basename(p))
    )

    result: dict = {
        "rgbs": [],
        "depths": [],
        "intrinsics": [],
        "extrinsics": [],
        "normals": [],
        "optical_flows": [],
        "scene_flows_3d": [],
        "semantics": [],
        "semantic_infos": [],
        "frame_indices": [],
    }

    depth_dir = os.path.join(cam_dir, _SUBFOLDER_DEPTH)
    intr_dir = os.path.join(cam_dir, _SUBFOLDER_INTRINSIC)
    extr_dir = os.path.join(cam_dir, _SUBFOLDER_EXTRINSIC)
    normal_dir = os.path.join(cam_dir, _SUBFOLDER_NORMAL)
    flow_dir = os.path.join(cam_dir, _SUBFOLDER_FLOW2D)
    sem_dir = os.path.join(cam_dir, _SUBFOLDER_SEMANTIC)

    for rgb_path in rgb_paths:
        basename = os.path.basename(rgb_path)
        frame_idx = int(basename.split(".")[0])

        depth_path = os.path.join(depth_dir, f"{frame_idx:010d}.npy")
        intr_path = os.path.join(intr_dir, f"{frame_idx:010d}.npy")
        extr_path = os.path.join(extr_dir, f"{frame_idx:010d}.npy")

        if not (os.path.exists(depth_path) and os.path.exists(intr_path) and os.path.exists(extr_path)):
            continue

        result["rgbs"].append(np.array(Image.open(rgb_path)))
        result["depths"].append(np.load(depth_path))
        result["intrinsics"].append(np.load(intr_path))
        result["extrinsics"].append(np.load(extr_path))
        result["frame_indices"].append(frame_idx)

        normals_path = os.path.join(normal_dir, f"{frame_idx:010d}.npy")
        result["normals"].append(np.load(normals_path) if os.path.exists(normals_path) else None)

        flow_path = os.path.join(flow_dir, f"{frame_idx:010d}.npy")
        result["optical_flows"].append(np.load(flow_path) if os.path.exists(flow_path) else None)

        flow3d_dir = os.path.join(cam_dir, _SUBFOLDER_FLOW3D)
        flow3d_path = os.path.join(flow3d_dir, f"{frame_idx:010d}.npy")
        result["scene_flows_3d"].append(np.load(flow3d_path) if os.path.exists(flow3d_path) else None)

        sem_path = os.path.join(sem_dir, f"{frame_idx:010d}.png")
        result["semantics"].append(np.array(Image.open(sem_path)) if os.path.exists(sem_path) else None)

        sem_json_path = os.path.join(sem_dir, f"{frame_idx:010d}.json")
        if os.path.exists(sem_json_path):
            with open(sem_json_path) as f:
                result["semantic_infos"].append(json.load(f))
        else:
            result["semantic_infos"].append(None)

    return result


def _sample_indices(total: int, num_samples: int) -> list[int]:
    """Return *num_samples* evenly-spaced indices from ``[0, total)``."""
    if total <= num_samples:
        return list(range(total))
    return [int(round(i * (total - 1) / (num_samples - 1))) for i in range(num_samples)]


def _colorize_depth(depth: np.ndarray, cmap_name: str = "Spectral") -> np.ndarray:
    """Normalize a depth map to [0, 1] and apply a matplotlib colormap.

    Returns:
        (H, W, 3) uint8 array.
    """
    d = depth.copy().astype(np.float64)
    finite_mask = np.isfinite(d)
    if finite_mask.any():
        d_min = d[finite_mask].min()
        d_max = d[finite_mask].max()
        d[~finite_mask] = d_max
    else:
        d_min, d_max = 0.0, 1.0

    if d_max - d_min < 1e-8:
        d_norm = np.zeros_like(d)
    else:
        d_norm = (d - d_min) / (d_max - d_min)

    colormap = cm.get_cmap(cmap_name)
    colored = (colormap(d_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def _colorize_normals(normals: np.ndarray) -> np.ndarray:
    """Map surface normals from [-1, 1] to [0, 255] for visualization.

    Returns:
        (H, W, 3) uint8 array.
    """
    n = normals.copy().astype(np.float64)
    n = np.nan_to_num(n, nan=0.0)
    return ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def _colorize_flow_fast(flow: np.ndarray) -> np.ndarray:
    """Vectorised HSV-based optical flow visualization.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    from matplotlib.colors import hsv_to_rgb as mpl_hsv_to_rgb

    dx = flow[..., 0].astype(np.float64)
    dy = flow[..., 1].astype(np.float64)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)

    mag_max = mag.max() if mag.max() > 0 else 1.0

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.float64)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag / mag_max

    rgb = (mpl_hsv_to_rgb(hsv) * 255).clip(0, 255).astype(np.uint8)
    return rgb


def _colorize_flow3d(flow3d: np.ndarray) -> np.ndarray:
    """Colorize 3D scene flow by displacement magnitude.

    Args:
        flow3d: (H, W, 3) float32 array of 3D displacement vectors (metres).

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    mag = np.linalg.norm(flow3d, axis=-1)
    mag_max = mag.max() if mag.max() > 1e-8 else 1.0
    mag_norm = mag / mag_max

    colormap = cm.get_cmap("inferno")
    colored = (colormap(mag_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def _build_semantic_legend(semantic_info: dict | None) -> list[Patch]:
    """Build matplotlib legend patches from the semantic JSON metadata."""
    if semantic_info is None:
        return []
    objects = semantic_info.get("objects", [])
    patches = []
    for obj in objects:
        rgba = obj.get("rgba", (128, 128, 128, 255))
        color = tuple(c / 255.0 for c in rgba[:3])
        name = obj.get("object_name", obj.get("class_name", "unknown"))
        label = f"{name} ({obj.get('pixel_count', '?')} px)"
        patches.append(Patch(facecolor=color, edgecolor="black", label=label))
    return patches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_rgb_grid(
    output_dir: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show a grid of sampled RGB frames (e.g. camera_id=\"cam0\")."""
    data = _load_frames(output_dir, camera_id)
    rgbs, frame_indices = data["rgbs"], data["frame_indices"]
    if not rgbs:
        print("[visualize_rgb_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    n = len(sample_ids)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    for i, idx in enumerate(sample_ids):
        r, c = divmod(i, ncols)
        axes[r, c].imshow(rgbs[idx])
        axes[r, c].set_title(f"Frame {frame_indices[idx]}", fontsize=9)

    fig.suptitle("Sampled RGB Frames", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_rgb_grid] Saved to {save_path}")
    else:
        plt.show()


def visualize_depth_grid(
    output_dir: str,
    camera_id: str,
    num_samples: int = 8,
    cmap: str = "turbo",
    save_path: str | None = None,
) -> None:
    """Show colorized depth images alongside their RGB counterparts."""
    data = _load_frames(output_dir, camera_id)
    rgbs, depths, frame_indices = data["rgbs"], data["depths"], data["frame_indices"]
    if not rgbs:
        print("[visualize_depth_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    n = len(sample_ids)

    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(sample_ids):
        axes[i, 0].imshow(rgbs[idx])
        axes[i, 0].set_title(f"RGB – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 0].axis("off")

        depth_colored = _colorize_depth(depths[idx], cmap_name=cmap)
        axes[i, 1].imshow(depth_colored)
        axes[i, 1].set_title(f"Depth – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle("RGB & Colorized Depth", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_depth_grid] Saved to {save_path}")
    else:
        plt.show()


def visualize_normals_grid(
    output_dir: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show RGB alongside colorized surface normals."""
    data = _load_frames(output_dir, camera_id)
    rgbs, normals_list, frame_indices = data["rgbs"], data["normals"], data["frame_indices"]
    if not rgbs:
        print("[visualize_normals_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    valid = [i for i in sample_ids if normals_list[i] is not None]
    if not valid:
        print("[visualize_normals_grid] No normal maps found.")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(valid):
        axes[i, 0].imshow(rgbs[idx])
        axes[i, 0].set_title(f"RGB – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_colorize_normals(normals_list[idx]))
        axes[i, 1].set_title(f"Normals – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle("RGB & Surface Normals", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_normals_grid] Saved to {save_path}")
    else:
        plt.show()


def visualize_optical_flow_grid(
    output_dir: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show RGB alongside HSV-colorized optical flow."""
    data = _load_frames(output_dir, camera_id)
    rgbs, flows, frame_indices = data["rgbs"], data["optical_flows"], data["frame_indices"]
    if not rgbs:
        print("[visualize_optical_flow_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    valid = [i for i in sample_ids if flows[i] is not None]
    if not valid:
        print("[visualize_optical_flow_grid] No optical flow data found.")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(valid):
        axes[i, 0].imshow(rgbs[idx])
        axes[i, 0].set_title(f"RGB – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_colorize_flow_fast(flows[idx]))
        axes[i, 1].set_title(f"Optical Flow – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle("RGB & Dense Optical Flow", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_optical_flow_grid] Saved to {save_path}")
    else:
        plt.show()


def visualize_semantic_segmentation_grid(
    output_dir: str,
    camera_id: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show RGB alongside semantic segmentation with a per-object legend.

    The legend is built from the JSON metadata written alongside each
    semantic PNG by :class:`IsaacLabArenaWriter`. Each entry shows the
    object colour, class name, and pixel count.
    """
    data = _load_frames(output_dir, camera_id)
    rgbs = data["rgbs"]
    semantics = data["semantics"]
    sem_infos = data["semantic_infos"]
    frame_indices = data["frame_indices"]
    if not rgbs:
        print("[visualize_semantic_segmentation_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    valid = [i for i in sample_ids if semantics[i] is not None]
    if not valid:
        print("[visualize_semantic_segmentation_grid] No semantic segmentation data found.")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(valid):
        axes[i, 0].imshow(rgbs[idx])
        axes[i, 0].set_title(f"RGB – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 0].axis("off")

        sem_rgba = semantics[idx]
        sem_rgb = sem_rgba[..., :3] if sem_rgba.shape[-1] == 4 else sem_rgba
        axes[i, 1].imshow(sem_rgb)
        axes[i, 1].set_title(f"Semantic – Frame {frame_indices[idx]}", fontsize=9)
        axes[i, 1].axis("off")

        legend_patches = _build_semantic_legend(sem_infos[idx] if idx < len(sem_infos) else None)
        if legend_patches:
            axes[i, 1].legend(
                handles=legend_patches, loc="upper left", fontsize=6,
                framealpha=0.8, handlelength=1.2, handleheight=1.0,
            )

    fig.suptitle("RGB & Semantic Segmentation", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_semantic_segmentation_grid] Saved to {save_path}")
    else:
        plt.show()


def visualize_all_modalities_grid(
    output_dir: str,
    camera_id: str,
    num_samples: int = 8,
    depth_cmap: str = "Spectral",
    save_path: str | None = None,
) -> None:
    """Single figure: for each sampled frame show all modalities in one row.

    Layout: num_samples rows x 6 columns
    (RGB | Depth | Flow 2D | Flow 3D | Normals | Semantics).
    Missing data is shown as a dark placeholder with 'N/A'.
    """
    data = _load_frames(output_dir, camera_id)
    rgbs = data["rgbs"]
    depths = data["depths"]
    flows = data["optical_flows"]
    flows_3d = data["scene_flows_3d"]
    normals_list = data["normals"]
    semantics = data["semantics"]
    sem_infos = data["semantic_infos"]
    frame_indices = data["frame_indices"]

    if not rgbs:
        print("[visualize_all_modalities_grid] No frames found.")
        return

    sample_ids = _sample_indices(len(rgbs), num_samples)
    n = len(sample_ids)
    ncols = 6
    nrows = n

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Color", "Depth", "Flow 2D", "Flow 3D", "Normals", "Semantics"]

    for i, idx in enumerate(sample_ids):
        frame_idx = frame_indices[idx]
        h, w = rgbs[idx].shape[:2]
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        blank[:] = 40

        # Column 0: RGB
        axes[i, 0].imshow(rgbs[idx])
        axes[i, 0].set_title(col_titles[0] if i == 0 else "", fontsize=10)
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(f"Frame {frame_idx}", fontsize=9)

        # Column 1: Depth
        if depths[idx] is not None:
            axes[i, 1].imshow(_colorize_depth(depths[idx], cmap_name=depth_cmap))
        else:
            axes[i, 1].imshow(blank)
            axes[i, 1].text(w // 2, h // 2, "N/A", ha="center", va="center", color="gray", fontsize=12)
        axes[i, 1].set_title(col_titles[1] if i == 0 else "", fontsize=10)
        axes[i, 1].axis("off")

        # Column 2: Optical flow (2D)
        if flows[idx] is not None:
            axes[i, 2].imshow(_colorize_flow_fast(flows[idx]))
        else:
            axes[i, 2].imshow(blank)
            axes[i, 2].text(w // 2, h // 2, "N/A", ha="center", va="center", color="gray", fontsize=12)
        axes[i, 2].set_title(col_titles[2] if i == 0 else "", fontsize=10)
        axes[i, 2].axis("off")

        # Column 3: Scene flow (3D)
        if flows_3d[idx] is not None:
            axes[i, 3].imshow(_colorize_flow3d(flows_3d[idx]))
        else:
            axes[i, 3].imshow(blank)
            axes[i, 3].text(w // 2, h // 2, "N/A", ha="center", va="center", color="gray", fontsize=12)
        axes[i, 3].set_title(col_titles[3] if i == 0 else "", fontsize=10)
        axes[i, 3].axis("off")

        # Column 4: Normals
        if normals_list[idx] is not None:
            axes[i, 4].imshow(_colorize_normals(normals_list[idx]))
        else:
            axes[i, 4].imshow(blank)
            axes[i, 4].text(w // 2, h // 2, "N/A", ha="center", va="center", color="gray", fontsize=12)
        axes[i, 4].set_title(col_titles[4] if i == 0 else "", fontsize=10)
        axes[i, 4].axis("off")

        # Column 5: Semantic segmentation
        if semantics[idx] is not None:
            sem_rgba = semantics[idx]
            sem_rgb = sem_rgba[..., :3] if sem_rgba.shape[-1] == 4 else sem_rgba
            axes[i, 5].imshow(sem_rgb)
            legend_patches = _build_semantic_legend(sem_infos[idx] if idx < len(sem_infos) else None)
            if legend_patches:
                axes[i, 5].legend(
                    handles=legend_patches,
                    loc="upper left",
                    fontsize=5,
                    framealpha=0.8,
                    handlelength=1.0,
                    handleheight=0.8,
                )
        else:
            axes[i, 5].imshow(blank)
            axes[i, 5].text(w // 2, h // 2, "N/A", ha="center", va="center", color="gray", fontsize=12)
        axes[i, 5].set_title(col_titles[5] if i == 0 else "", fontsize=10)
        axes[i, 5].axis("off")

    fig.suptitle("Color | Depth | Flow 2D | Flow 3D | Normals | Semantics", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_all_modalities_grid] Saved to {save_path}")
    else:
        plt.show()


def visualize_camera_trajectory(
    output_dir: str,
    camera_id: str,
    axis_length: float = 0.05,
    frustum_scale: float = 0.04,
    num_frustums: int = 20,
    save_path: str | None = None,
) -> None:
    """Plot the camera trajectory in 3D with coordinate frames and frustum outlines."""
    data = _load_frames(output_dir, camera_id)
    intrinsics_list, extrinsics_list = data["intrinsics"], data["extrinsics"]
    if not extrinsics_list:
        print("[visualize_camera_trajectory] No frames found.")
        return

    positions = np.array([T[:3, 3] for T in extrinsics_list])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-", color="gray", linewidth=1, label="trajectory")
    ax.scatter(*positions[0], color="green", s=60, zorder=5, label="start")
    ax.scatter(*positions[-1], color="red", s=60, zorder=5, label="end")

    frame_sample_ids = _sample_indices(len(extrinsics_list), num_frustums)
    axis_colors = ["r", "g", "b"]

    for sid in frame_sample_ids:
        T = extrinsics_list[sid]
        R = T[:3, :3]
        t = T[:3, 3]

        for col, color in enumerate(axis_colors):
            direction = R[:, col] * axis_length
            ax.quiver(t[0], t[1], t[2], direction[0], direction[1], direction[2], color=color, arrow_length_ratio=0.15)

        K = intrinsics_list[sid]
        _draw_frustum(ax, K, T, frustum_scale)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera Trajectory & Frames")
    ax.legend(fontsize=8)
    _set_equal_aspect_3d(ax, positions)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_camera_trajectory] Saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3-D frustum drawing helpers
# ---------------------------------------------------------------------------


def _draw_frustum(
    ax: Axes3D,
    K: np.ndarray,
    T_cam2world: np.ndarray,
    depth: float,
    color: str = "royalblue",
    linewidth: float = 0.6,
) -> None:
    """Draw a small wireframe camera frustum in world coordinates."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    corners_px = np.array([
        [0, 0],
        [2 * cx, 0],
        [2 * cx, 2 * cy],
        [0, 2 * cy],
    ], dtype=np.float64)

    corners_cam = np.zeros((4, 3))
    for i, (u, v) in enumerate(corners_px):
        corners_cam[i] = [(u - cx) / fx * depth, (v - cy) / fy * depth, depth]

    R = T_cam2world[:3, :3]
    t = T_cam2world[:3, 3]

    corners_world = (R @ corners_cam.T).T + t
    apex = t

    for cw in corners_world:
        ax.plot([apex[0], cw[0]], [apex[1], cw[1]], [apex[2], cw[2]], color=color, linewidth=linewidth)

    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            color=color,
            linewidth=linewidth,
        )


def _set_equal_aspect_3d(ax: Axes3D, points: np.ndarray) -> None:
    """Set equal aspect ratio on a 3-D matplotlib axis."""
    mid = points.mean(axis=0)
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2
    span = max(span, 0.1)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
