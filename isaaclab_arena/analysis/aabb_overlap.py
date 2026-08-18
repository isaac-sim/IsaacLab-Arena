# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read recorded object bounding boxes back out of a trajectory dataset and compare their footprints.

Complements the runtime pick-and-place success signal, which fires on contact force plus low
velocity and so accepts an object merely resting against its destination's rim. The horizontal
overlap between the two objects' boxes distinguishes "over the destination" from "touching it".

Needs neither Isaac Sim nor pxr: the boxes are already stored in the dataset by
``isaaclab_arena.analysis.recording_annotation``, and ``AxisAlignedBoundingBox`` is plain torch.
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any

from isaaclab_arena.analysis.recording_annotation import BOUNDING_BOXES_GROUP
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

SUPPORTED_FORMAT_VERSION = 1
"""Dataset format whose ``root_pose`` quaternions are XYZW, matching ``rotated_by_quat``."""


def find_demo_name(hdf5_path: str | Path, env_id: int, episode_in_env: int) -> str:
    """Return the demo that recorded one episode, joining a dataset to an ``episode_results`` record.

    Args:
        hdf5_path: An annotated recorded dataset.
        env_id: The episode's ``env_id``, as recorded in the results file.
        episode_in_env: The episode's index within that env.
    """
    import h5py

    with h5py.File(Path(hdf5_path), "r") as hdf5_file:
        for name, demo in hdf5_file["data"].items():
            episode = demo["episode_id"]
            if int(episode["env_id"][0]) == env_id and int(episode["episode_in_env"][0]) == episode_in_env:
                return name
    raise KeyError(f"No demo with env_id={env_id}, episode_in_env={episode_in_env} in {hdf5_path}")


def _local_bounding_box(hdf5_file: Any, demo: Any, object_name: str) -> AxisAlignedBoundingBox:
    """Return ``object_name``'s stored object-frame box, picking the row for the env ``demo`` ran in."""
    group = hdf5_file[BOUNDING_BOXES_GROUP]
    assert object_name in group, (
        f"No bounding box stored for {object_name!r}; annotate the dataset with "
        "isaaclab_arena.analysis.recording_annotation --bounding-boxes first."
    )
    object_group = group[object_name]
    assert "error" not in object_group.attrs, f"No bounding box for {object_name!r}: {object_group.attrs['error']}"

    minimum_points = torch.as_tensor(object_group["min_point"][:], dtype=torch.float32)
    maximum_points = torch.as_tensor(object_group["max_point"][:], dtype=torch.float32)
    # One row means the box is the same for every env; otherwise a row per env's spawned variant.
    row = 0 if minimum_points.shape[0] == 1 else int(demo["episode_id"]["env_id"][0])
    return AxisAlignedBoundingBox(min_point=minimum_points[row], max_point=maximum_points[row])


def read_object_aabbs(
    hdf5_path: str | Path,
    child_name: str,
    destination_name: str,
    *,
    demo: str = "demo_0",
    frame: int = -1,
) -> tuple[AxisAlignedBoundingBox, AxisAlignedBoundingBox]:
    """Return world-space bounding boxes for two recorded objects at one frame of one demo.

    Each object's stored local box is rotated and translated by its recorded ``root_pose``. Because a
    rotated box is refitted to stay axis-aligned, the result is exact only at 90-degree multiples and
    conservative (larger than the object) otherwise.

    Args:
        hdf5_path: A dataset already annotated with ``--bounding-boxes``.
        child_name: The manipulated object, e.g. a pick-and-place task's ``pick_up_object``.
        destination_name: The object it is placed on or in.
        demo: Which demo to read; ``find_demo_name`` resolves one from an ``episode_results`` record.
        frame: Timestep within the demo, defaulting to the last one.

    Returns:
        ``(child_box, destination_box)`` in world coordinates.
    """
    import h5py

    with h5py.File(Path(hdf5_path), "r") as hdf5_file:
        format_version = int(hdf5_file.attrs.get("format_version", -1))
        assert format_version == SUPPORTED_FORMAT_VERSION, (
            f"{Path(hdf5_path).name} has format_version={format_version}; only "
            f"{SUPPORTED_FORMAT_VERSION} (XYZW quaternions) is supported."
        )
        demo_group = hdf5_file["data"][demo]
        boxes = []
        for object_name in (child_name, destination_name):
            local_box = _local_bounding_box(hdf5_file, demo_group, object_name)
            root_pose = demo_group["states"]["rigid_object"][object_name]["root_pose"][frame]
            position = tuple(float(value) for value in root_pose[:3])
            rotation_xyzw = tuple(float(value) for value in root_pose[3:7])
            boxes.append(local_box.rotated_by_quat(rotation_xyzw).translated(position))
    return boxes[0], boxes[1]


def xy_overlap_fraction(child: AxisAlignedBoundingBox, destination: AxisAlignedBoundingBox) -> float:
    """Return the fraction of ``child``'s horizontal footprint that lies over ``destination``'s.

    Dividing by the child's own area rather than by the union or the destination's area keeps the
    measure meaningful when the two differ greatly in size: a small object fully inside a large
    container scores 1.0. The value says nothing about height, so it complements rather than replaces
    the contact-based success signal.

    Args:
        child: World-space box of the manipulated object.
        destination: World-space box of the object it should end up on or in.

    Returns:
        A value in ``[0, 1]``: 1.0 when the child's footprint is entirely over the destination's,
        0.0 when they do not overlap.
    """
    child_minimum, child_maximum = child.min_point[0], child.max_point[0]
    destination_minimum, destination_maximum = destination.min_point[0], destination.max_point[0]

    overlap = torch.clamp(
        torch.minimum(child_maximum[:2], destination_maximum[:2])
        - torch.maximum(child_minimum[:2], destination_minimum[:2]),
        min=0.0,
    )
    child_extent = child_maximum[:2] - child_minimum[:2]
    child_area = float(child_extent[0] * child_extent[1])
    assert child_area > 0.0, f"Child bounding box has no horizontal area: {child}"

    return float(overlap[0] * overlap[1]) / child_area
