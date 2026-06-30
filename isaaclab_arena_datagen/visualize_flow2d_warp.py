# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Visually validate stored 2D optical flow by forward-warping color frames.

For a consecutive stored-frame pair the "from" color image is forward-warped
along flow2d and written next to the actual "to" image, so the two can be
flipped back and forth in a viewer to confirm flow2d survived frame subsampling.
"""

from __future__ import annotations

import argparse
import h5py
import json
import numpy as np
import os

import hdf5plugin  # noqa: F401  -- registers Zstd so compressed datasets are readable
from PIL import Image

from isaaclab_arena_datagen.io import hdf5_keys as Keys


def read_camera_ids(seq_group: h5py.Group) -> list[str]:
    """Return the ordered camera ids stored on the sequence group."""
    raw = seq_group.attrs[Keys.ATTR_CAMERA_IDS]
    if isinstance(raw, (bytes, str)):
        return list(json.loads(raw))
    return [c.decode() if isinstance(c, bytes) else str(c) for c in raw]


def open_sequence_group(h5: h5py.File) -> h5py.Group:
    """Return the single per-file sequence group (sequence_000000)."""
    name = Keys.sequence_group_name(0)
    assert name in h5, f"Expected group {name!r} in dataset; found {list(h5.keys())}"
    return h5[name]


def forward_warp(from_rgb: np.ndarray, flow_dxdy: np.ndarray) -> np.ndarray:
    """Forward-scatter every from-frame pixel to (x+dx, y+dy); out-of-bounds dropped.

    Args:
        from_rgb: (H, W, 3) uint8 source image.
        flow_dxdy: (H, W, 2) float pixel displacement mapping frame i -> i+1.

    Returns:
        (H, W, 3) uint8 warped image with black holes where nothing landed.
    """
    height, width = from_rgb.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    dst_x = np.rint(xs + flow_dxdy[..., 0]).astype(np.int64)
    dst_y = np.rint(ys + flow_dxdy[..., 1]).astype(np.int64)

    in_bounds = (dst_x >= 0) & (dst_x < width) & (dst_y >= 0) & (dst_y < height)
    warped = np.zeros_like(from_rgb)
    warped[dst_y[in_bounds], dst_x[in_bounds]] = from_rgb[ys[in_bounds], xs[in_bounds]]
    return warped


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute photometric error between two uint8 images."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def emit_pair(
    cam_group: h5py.Group,
    cam_id: str,
    from_idx: int,
    output_dir: str,
) -> None:
    """Write the to-image and the warped-from-image for one stored-frame pair."""
    color = cam_group[Keys.COLOR]
    flow = cam_group[Keys.FLOW2D]

    num_frames = color.shape[0]
    assert from_idx + 1 < num_frames, f"from frame {from_idx} has no successor (num_frames={num_frames})"
    assert from_idx < flow.shape[0], f"flow2d has {flow.shape[0]} rows; no entry for from frame {from_idx}"

    from_rgb = np.asarray(color[from_idx])[..., :3]
    to_rgb = np.asarray(color[from_idx + 1])[..., :3]
    flow_dxdy = np.asarray(flow[from_idx], dtype=np.float32)

    assert flow_dxdy.shape[:2] == from_rgb.shape[:2], (
        f"flow2d resolution {flow_dxdy.shape[:2]} != color resolution {from_rgb.shape[:2]}; "
        "cannot warp without rescaling"
    )

    warped = forward_warp(from_rgb, flow_dxdy)

    raw_err = mean_abs_error(from_rgb, to_rgb)
    warped_err = mean_abs_error(warped, to_rgb)
    verdict = "OK (warped < raw)" if warped_err < raw_err else "SUSPECT (warped >= raw)"
    print(f"{cam_id} frame {from_idx}->{from_idx + 1}: raw_err={raw_err:.3f}  warped_err={warped_err:.3f}  {verdict}")

    stem = f"{cam_id}_f{from_idx:04d}"
    to_path = os.path.join(output_dir, f"{stem}_a_to.png")
    warped_path = os.path.join(output_dir, f"{stem}_b_warped_from.png")
    Image.fromarray(to_rgb).save(to_path)
    Image.fromarray(warped).save(warped_path)
    print(f"  wrote {to_path}")
    print(f"  wrote {warped_path}")


def main() -> None:
    """Parse CLI arguments and emit warped/to image pairs for the chosen frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to a datagen dataset.h5")
    parser.add_argument("--output-dir", required=True, help="Directory for the emitted PNG pairs")
    parser.add_argument("--camera", default=None, help="Camera id (default: first camera)")
    parser.add_argument("--frame", type=int, default=0, help="The 'from' frame index (default: 0)")
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=1,
        help="Number of consecutive pairs starting at --frame (default: 1)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.dataset, "r") as h5:
        seq_group = open_sequence_group(h5)
        camera_ids = read_camera_ids(seq_group)
        assert camera_ids, "Sequence group lists no cameras"

        cam_id = args.camera if args.camera is not None else camera_ids[0]
        assert cam_id in seq_group, f"Camera {cam_id!r} not in {camera_ids}"
        cam_group = seq_group[cam_id]

        for offset in range(args.num_pairs):
            emit_pair(cam_group, cam_id, args.frame + offset, args.output_dir)


if __name__ == "__main__":
    main()
