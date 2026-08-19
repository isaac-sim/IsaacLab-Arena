# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read recorded object bounding boxes back out of a trajectory dataset and compare their footprints.

Complements the runtime pick-and-place success signal, which fires on contact force plus low velocity
and so accepts an object merely resting against its destination's rim. The horizontal overlap between
the two objects' boxes distinguishes "over the destination" from "touching it".

This module is deliberately self-contained -- ``h5py`` and ``numpy`` only, no Arena, Isaac Sim, pxr or
torch imports -- so it can be vendored into a separate analysis repository as a single file. That is
why it restates the HDF5 layout it reads and carries its own small bounding-box type instead of
reusing ``isaaclab_arena.utils.bounding_box``; Arena's test suite pins both against their in-tree
counterparts so the copy cannot silently drift. The boxes themselves are written by
``isaaclab_arena.analysis.recording_annotation --bounding-boxes``, which does need Isaac Sim.
"""

from __future__ import annotations

import h5py
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOUNDING_BOXES_GROUP = "bounding_boxes"
"""Top-level HDF5 group the annotation tool writes each object's local box to."""

SUPPORTED_FORMAT_VERSION = 1
"""Dataset format whose ``root_pose`` quaternions are XYZW, matching ``rotated_by_quat``."""


@dataclass(frozen=True)
class AxisAlignedBoundingBox:
    """A world-axis-aligned box, held as its two opposite corners in whichever frame the caller placed it.

    Args:
        min_point: Lower corner ``(x, y, z)``.
        max_point: Upper corner ``(x, y, z)``.
    """

    min_point: np.ndarray
    max_point: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_point", np.asarray(self.min_point, dtype=np.float64).reshape(3))
        object.__setattr__(self, "max_point", np.asarray(self.max_point, dtype=np.float64).reshape(3))
        assert np.all(self.min_point <= self.max_point), f"min_point must not exceed max_point: {self}"

    @property
    def size(self) -> np.ndarray:
        """Extent ``(width, depth, height)``."""
        return self.max_point - self.min_point

    @property
    def center(self) -> np.ndarray:
        """Midpoint of the box."""
        return 0.5 * (self.min_point + self.max_point)

    def corners(self) -> np.ndarray:
        """Return the box's eight corners as a ``(8, 3)`` array, in no particular order."""
        lower, upper = self.min_point, self.max_point
        return np.array(
            [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
        )

    def rotated_by_quat(self, rotation_xyzw: tuple[float, float, float, float]) -> AxisAlignedBoundingBox:
        """Refit to the axis-aligned box enclosing this box rotated about its origin by a quaternion.

        Rotating an axis-aligned box generally yields one that is not, so the result is the tightest
        enclosing axis-aligned box: exact at 90-degree multiples and conservative (larger than the
        object) otherwise.

        Args:
            rotation_xyzw: Rotation quaternion as ``(x, y, z, w)``, the order recorded by
                ``SUPPORTED_FORMAT_VERSION`` datasets.
        """
        quaternion = np.asarray(rotation_xyzw, dtype=np.float64).reshape(4)
        axis, angle_cosine = quaternion[:3], quaternion[3]
        corners = self.corners()
        # Rotate each corner by v + 2w(a x v) + 2a x (a x v), the quaternion sandwich product expanded.
        axis_cross = np.cross(axis, corners)
        rotated = corners + 2.0 * angle_cosine * axis_cross + 2.0 * np.cross(axis, axis_cross)
        return AxisAlignedBoundingBox(min_point=rotated.min(axis=0), max_point=rotated.max(axis=0))

    def translated(self, offset: tuple[float, float, float]) -> AxisAlignedBoundingBox:
        """Return this box shifted by ``offset``."""
        offset = np.asarray(offset, dtype=np.float64).reshape(3)
        return AxisAlignedBoundingBox(min_point=self.min_point + offset, max_point=self.max_point + offset)


def find_demo_name(hdf5_path: str | Path, env_id: int, episode_in_env: int) -> str:
    """Return the demo that recorded one episode, joining a dataset to an ``episode_results`` record.

    Demos are stored in completion order, so ``demo_0`` is not necessarily the episode from env 0.

    Args:
        hdf5_path: An annotated recorded dataset.
        env_id: The episode's ``env_id``, as recorded in the results file.
        episode_in_env: The episode's index within that env.
    """
    with h5py.File(Path(hdf5_path), "r") as hdf5_file:
        for name, demo in hdf5_file["data"].items():
            episode = demo["episode_id"]
            if int(episode["env_id"][0]) == env_id and int(episode["episode_in_env"][0]) == episode_in_env:
                return name
    raise KeyError(f"No demo with env_id={env_id}, episode_in_env={episode_in_env} in {hdf5_path}")


def _local_bounding_box(hdf5_file: Any, demo: Any, object_name: str) -> AxisAlignedBoundingBox:
    """Return ``object_name``'s stored object-frame box, picking the row for the env ``demo`` ran in."""
    assert BOUNDING_BOXES_GROUP in hdf5_file, (
        f"No {BOUNDING_BOXES_GROUP!r} group in this dataset; annotate it with "
        "isaaclab_arena.analysis.recording_annotation --bounding-boxes first."
    )
    group = hdf5_file[BOUNDING_BOXES_GROUP]
    assert object_name in group, (
        f"No bounding box stored for {object_name!r}; annotate the dataset with "
        "isaaclab_arena.analysis.recording_annotation --bounding-boxes first."
    )
    object_group = group[object_name]
    assert "error" not in object_group.attrs, f"No bounding box for {object_name!r}: {object_group.attrs['error']}"

    minimum_points = np.asarray(object_group["min_point"][:], dtype=np.float64)
    maximum_points = np.asarray(object_group["max_point"][:], dtype=np.float64)
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
    overlap = np.clip(
        np.minimum(child.max_point[:2], destination.max_point[:2])
        - np.maximum(child.min_point[:2], destination.min_point[:2]),
        a_min=0.0,
        a_max=None,
    )
    child_extent = child.size[:2]
    child_area = float(child_extent[0] * child_extent[1])
    assert child_area > 0.0, f"Child bounding box has no horizontal area: {child}"

    return float(overlap[0] * overlap[1]) / child_area
