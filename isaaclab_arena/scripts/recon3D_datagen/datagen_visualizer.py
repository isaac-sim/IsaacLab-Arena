# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for recon3D data generation outputs.

Provides functions to:
- Display a grid of sampled RGB frames.
- Display colorized depth images alongside their RGB counterparts.
- Plot 3D camera trajectory with coordinate frames and optional frustums.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_frames(
    output_dir: str,
    camera_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Load all frames written by :class:`IsaacLabArenaWriter`.

    Returns:
        Tuple of (rgbs, depths, intrinsics_list, extrinsics_list, frame_indices)
        sorted by frame index.
    """
    rgb_pattern = os.path.join(output_dir, f"*.{camera_name}_rgb.png")
    rgb_paths = sorted(glob.glob(rgb_pattern))

    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    extrinsics_list: list[np.ndarray] = []
    frame_indices: list[int] = []

    for rgb_path in rgb_paths:
        basename = os.path.basename(rgb_path)
        frame_idx = int(basename.split(".")[0])

        depth_path = os.path.join(output_dir, f"{frame_idx:04d}.{camera_name}_depth.npy")
        intr_path = os.path.join(output_dir, f"{frame_idx:04d}.{camera_name}_intrinsics.npy")
        extr_path = os.path.join(output_dir, f"{frame_idx:04d}.{camera_name}_extrinsics.npy")

        if not (os.path.exists(depth_path) and os.path.exists(intr_path) and os.path.exists(extr_path)):
            continue

        rgbs.append(np.array(Image.open(rgb_path)))
        depths.append(np.load(depth_path))
        intrinsics_list.append(np.load(intr_path))
        extrinsics_list.append(np.load(extr_path))
        frame_indices.append(frame_idx)

    return rgbs, depths, intrinsics_list, extrinsics_list, frame_indices


def _sample_indices(total: int, num_samples: int) -> list[int]:
    """Return *num_samples* evenly-spaced indices from ``[0, total)``."""
    if total <= num_samples:
        return list(range(total))
    return [int(round(i * (total - 1) / (num_samples - 1))) for i in range(num_samples)]


def _colorize_depth(depth: np.ndarray, cmap_name: str = "Spectral") -> np.ndarray:
    """Normalize a depth map to [0, 1] and apply a matplotlib colormap.

    Infinite/NaN values are clamped to the finite maximum.

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_rgb_grid(
    output_dir: str,
    camera_name: str,
    num_samples: int = 8,
    save_path: str | None = None,
) -> None:
    """Show a grid of sampled RGB frames from the generated dataset.

    Args:
        output_dir: Directory written to by :class:`IsaacLabArenaWriter`.
        camera_name: Camera identifier used during writing.
        num_samples: Number of frames to sample uniformly.
        save_path: If given, save the figure instead of displaying it.
    """
    rgbs, _, _, _, frame_indices = _load_frames(output_dir, camera_name)
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
    camera_name: str,
    num_samples: int = 8,
    cmap: str = "turbo",
    save_path: str | None = None,
) -> None:
    """Show colorized depth images alongside their RGB counterparts.

    Args:
        output_dir: Directory written to by :class:`IsaacLabArenaWriter`.
        camera_name: Camera identifier used during writing.
        num_samples: Number of frames to sample uniformly.
        cmap: Matplotlib colormap name for depth coloring.
        save_path: If given, save the figure instead of displaying it.
    """
    rgbs, depths, _, _, frame_indices = _load_frames(output_dir, camera_name)
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


def visualize_camera_trajectory(
    output_dir: str,
    camera_name: str,
    axis_length: float = 0.05,
    frustum_scale: float = 0.04,
    num_frustums: int = 20,
    save_path: str | None = None,
) -> None:
    """Plot the camera trajectory in 3D with coordinate frames and frustum outlines.

    The 4x4 extrinsics are interpreted as camera-to-world transforms (as written
    by :class:`IsaacLabArenaWriter`).  Each sampled camera pose is drawn as an
    RGB axis triad (X=red, Y=green, Z=blue/optical axis) and a wireframe
    frustum whose shape reflects the intrinsic matrix.

    Args:
        output_dir: Directory written to by :class:`IsaacLabArenaWriter`.
        camera_name: Camera identifier used during writing.
        axis_length: Length of each coordinate-axis arrow (metres).
        frustum_scale: Depth of the visualized frustum pyramid (metres).
        num_frustums: Max number of frustum wireframes to draw.
        save_path: If given, save the figure instead of displaying it.
    """
    _, _, intrinsics_list, extrinsics_list, frame_indices = _load_frames(output_dir, camera_name)
    if not extrinsics_list:
        print("[visualize_camera_trajectory] No frames found.")
        return

    positions = np.array([T[:3, 3] for T in extrinsics_list])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-", color="gray", linewidth=1, label="trajectory")
    ax.scatter(*positions[0], color="green", s=60, zorder=5, label="start")
    ax.scatter(*positions[-1], color="red", s=60, zorder=5, label="end")

    # Draw coordinate frames & frustums for a subset of poses
    frame_sample_ids = _sample_indices(len(extrinsics_list), num_frustums)
    axis_colors = ["r", "g", "b"]

    for sid in frame_sample_ids:
        T = extrinsics_list[sid]
        R = T[:3, :3]
        t = T[:3, 3]

        # Axis triad
        for col, color in enumerate(axis_colors):
            direction = R[:, col] * axis_length
            ax.quiver(t[0], t[1], t[2], direction[0], direction[1], direction[2], color=color, arrow_length_ratio=0.15)

        # Frustum wireframe using intrinsics
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
    """Draw a small wireframe camera frustum in world coordinates.

    The frustum is a pyramid with its apex at the camera centre and base at
    distance *depth* along the optical axis.  The four base corners are
    computed by back-projecting the image corners using the intrinsic matrix.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Image corner coordinates (pixel)
    corners_px = np.array([
        [0, 0],
        [2 * cx, 0],
        [2 * cx, 2 * cy],
        [0, 2 * cy],
    ], dtype=np.float64)

    # Back-project to camera frame at z = depth
    corners_cam = np.zeros((4, 3))
    for i, (u, v) in enumerate(corners_px):
        corners_cam[i] = [(u - cx) / fx * depth, (v - cy) / fy * depth, depth]

    R = T_cam2world[:3, :3]
    t = T_cam2world[:3, 3]

    corners_world = (R @ corners_cam.T).T + t
    apex = t

    # Four edges from apex to each base corner
    for cw in corners_world:
        ax.plot([apex[0], cw[0]], [apex[1], cw[1]], [apex[2], cw[2]], color=color, linewidth=linewidth)

    # Base rectangle
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
    """Set equal aspect ratio on a 3-D matplotlib axis so cameras don't look squished."""
    mid = points.mean(axis=0)
    span = (points.max(axis=0) - points.min(axis=0)).max() / 2
    span = max(span, 0.1)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
