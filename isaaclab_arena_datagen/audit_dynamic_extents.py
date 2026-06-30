# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Audit the world-space extent of each dynamic object's sampled mesh points.

Reconstructs world points from the stored per-object body pose and body-frame mesh
samples, then flags any object larger than a plausible tabletop size, which catches
unit/scale errors (e.g. a centimeter-authored asset reconstructing 100x too big).
"""

from __future__ import annotations

import argparse
import glob
import h5py
import numpy as np
import os

import hdf5plugin  # noqa: F401  (registers the Zstd filter used by the datasets)

POSES_GROUP = "dynamic_objects/poses"
MESH_GROUP = "dynamic_objects/mesh_samples"


def _object_world_extents(dataset_path: str, frame: int) -> dict[str, float]:
    """Return the max world-space bounding-box side length per top-level object."""
    extents: dict[str, np.ndarray] = {}
    with h5py.File(dataset_path, "r") as f:
        seq = f[next(iter(f.keys()))]

        def visit(rel: str, obj: object) -> None:
            pose_path = f"{POSES_GROUP}/{rel}"
            if not isinstance(obj, h5py.Dataset) or pose_path not in seq:
                return
            pose = np.asarray(seq[pose_path], np.float32)
            pts_body = np.asarray(obj, np.float32)[:, :, 3]
            fr = min(frame, pose.shape[0] - 1)
            world = pts_body @ pose[fr, :, :3].T + pose[fr, :, 3]
            top = rel.split("/")[0]
            extents[top] = np.vstack([extents.get(top, np.empty((0, 3), np.float32)), world])

        seq[MESH_GROUP].visititems(visit)
    return {name: float((w.max(0) - w.min(0)).max()) for name, w in extents.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A dataset.h5 file or a directory searched recursively for them.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to reconstruct poses at.")
    parser.add_argument("--max-size", type=float, default=3.0, help="Flag objects whose largest side exceeds this (m).")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", "*.h5"), recursive=True))
    else:
        files = [args.path]
    assert files, f"no .h5 files found under {args.path}"

    flagged = 0
    for path in files:
        print(f"\n{path}")
        for name, size in sorted(_object_world_extents(path, args.frame).items()):
            mark = "  <-- TOO BIG" if size > args.max_size else ""
            flagged += size > args.max_size
            print(f"  {name:40s} max side = {size:8.3f} m{mark}")
    print(f"\n{flagged} object(s) exceeded {args.max_size} m")


if __name__ == "__main__":
    main()
