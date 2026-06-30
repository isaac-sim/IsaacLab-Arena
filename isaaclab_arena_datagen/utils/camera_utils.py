# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Camera coordinate helpers for the data-generation pipeline."""

from __future__ import annotations

import numpy as np

from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory, Coord3D

# Generic last-resort view for environments that do not define
# ``get_default_cameras`` and when no camera is supplied via the CLI. An elevated
# front-oblique pose (rather than a degenerate straight-down view) so the look-at
# basis is well-conditioned and a tabletop near the origin is visible.
DEFAULT_CAMERA = CameraViewTrajectory(
    position=(0.0, -1.0, 1.0),
    target=(0.0, 0.0, 0.0),
    focal_length_mm=24.0,
)


def resolve_coord(
    coord: Coord3D | list[Coord3D],
    step: int = 0,
) -> Coord3D:
    """Return the (x, y, z) tuple for a camera coordinate at the given step.

    Supports both static and dynamic (per-step) camera coordinates.  A static
    coordinate is a single ``(x, y, z)`` tuple that is returned unchanged
    regardless of *step*.  A dynamic coordinate is a list of such tuples, one
    per simulation step, from which the entry at *step* is selected.

    Args:
        coord: Either a single ``(x, y, z)`` tuple (static) or a list of
            ``(x, y, z)`` tuples indexed by simulation step (dynamic).
        step: Zero-based simulation step index used to look up dynamic
            coordinates.  Ignored for static coordinates.  Defaults to ``0``.

    Returns:
        The resolved ``(x, y, z)`` coordinate for the requested step.
    """
    if isinstance(coord, tuple):
        return coord
    return coord[step]


def sample_front_hemisphere_cameras(
    num_cameras: int,
    radius: float,
    center: Coord3D = (0.0, 0.0, 0.0),
    front_dir: Coord3D = (1.0, 0.0, 0.0),
    focal_length_mm: float = 24.0,
    min_height: float = 0.1,
    seed: int | None = None,
) -> list[CameraViewTrajectory]:
    """Sample *num_cameras* look-at cameras uniformly over the front hemisphere.

    Cameras sit at *radius* from *center* on the 180-degree hemisphere facing
    *front_dir* and look at *center*. Sampling is area-uniform; positions below
    *min_height* (world z) are rejected so none end up under the floor.
    Re-randomises every call unless *seed* is given.

    Args:
        num_cameras: Number of cameras to place.
        radius: Distance of every camera from *center* (metres).
        center: World-frame look-at point (e.g. the robot / workspace centre).
        front_dir: Direction the hemisphere faces (need not be normalised).
        focal_length_mm: Focal length for every camera.
        min_height: Minimum world z; lower samples are rejected and re-drawn.
        seed: RNG seed; ``None`` (default) re-randomises each call.

    Returns:
        A list of *num_cameras* look-at :class:`CameraViewTrajectory`.
    """
    if num_cameras < 1:
        raise ValueError(f"num_cameras must be >= 1, got {num_cameras}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    rng = np.random.default_rng(seed)
    center_arr = np.asarray(center, dtype=np.float64)
    front = np.asarray(front_dir, dtype=np.float64)
    front_norm = np.linalg.norm(front)
    if front_norm < 1e-9:
        raise ValueError("front_dir must be a non-zero vector.")
    front /= front_norm

    cameras: list[CameraViewTrajectory] = []
    max_attempts = max(1000, num_cameras * 1000)
    attempts = 0
    while len(cameras) < num_cameras and attempts < max_attempts:
        attempts += 1
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction /= norm
        # Fold onto the front hemisphere: negating a uniform sphere sample keeps
        # the distribution uniform on the chosen half.
        if float(np.dot(direction, front)) < 0.0:
            direction = -direction
        position = center_arr + radius * direction
        if position[2] < min_height:
            continue
        cameras.append(
            CameraViewTrajectory(
                position=tuple(float(c) for c in position),
                target=tuple(float(c) for c in center_arr),
                focal_length_mm=focal_length_mm,
            )
        )

    if len(cameras) < num_cameras:
        raise ValueError(
            f"Sampled only {len(cameras)}/{num_cameras} valid camera positions in {attempts} attempts. "
            f"min_height={min_height} likely excludes most of the hemisphere for center={center}, "
            f"radius={radius}; lower min_height or raise center."
        )
    return cameras


def validate_camera_configs(cameras: list[CameraViewTrajectory], num_steps: int) -> None:
    """Verify that every dynamic camera coordinate has the correct step count.

    Delegates to :meth:`CameraViewTrajectory.validate_trajectory_length` for
    each camera.

    Args:
        cameras: Camera configurations to validate.
        num_steps: Expected number of simulation steps.

    Raises:
        ValueError: If a dynamic coordinate list length does not match
            *num_steps*.
    """
    for cam in cameras:
        cam.validate_trajectory_length(num_steps)
