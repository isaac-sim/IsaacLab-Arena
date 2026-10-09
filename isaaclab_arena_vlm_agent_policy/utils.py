# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for VLM agent policies."""

from __future__ import annotations

import numpy as np
import torch

from PIL import Image


def encoded_intrinsics(matrix, source_hw, max_edge):
    """Return intrinsics and image dimensions matching the JPEG encoder's thumbnail sizing."""
    height, width = source_hw
    thumbnail = Image.new("L", (width, height))
    thumbnail.thumbnail((max_edge, max_edge))
    encoded_width, encoded_height = thumbnail.size
    intrinsics = np.array(matrix, dtype=float, copy=True)
    intrinsics[0] *= encoded_width / width
    intrinsics[1] *= encoded_height / height
    return intrinsics.tolist(), [encoded_height, encoded_width]


def compute_next_reference_pose(
    reference_pose: np.ndarray,
    goal_pose: np.ndarray,
    *,
    max_position_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    """Compute the next commanded EEF pose toward a goal within per-step motion limits.

    Args:
        reference_pose: Previous commanded XYZ/XYZW pose, initialized from the measured pose for a new goal.
        goal_pose: Desired XYZ/XYZW pose in the same coordinate frame as the reference.
        max_position_step_m: Maximum translation of the reference this step, in meters.
        max_rotation_step_rad: Maximum rotation of the reference this step, in radians.

    Returns:
        A new XYZ/XYZW reference pose, reaching the goal when both errors are within the step limits.
    """
    from isaaclab.utils.math import normalize, quat_box_minus, quat_slerp

    # Move along the straight line to the goal, capping distance without overshooting.
    result = reference_pose.copy()
    delta = goal_pose[:3] - reference_pose[:3]
    result[:3] += delta * min(1.0, max_position_step_m / max(np.linalg.norm(delta), 1e-12))

    # Normalize XYZW quaternions into independent tensors; SLERP may negate its target in place.
    start = normalize(torch.as_tensor(reference_pose[3:7], dtype=torch.float64))
    end = normalize(torch.as_tensor(goal_pose[3:7], dtype=torch.float64))

    # Measure the shortest rotation and choose the fraction allowed in one step.
    rotation_error = torch.linalg.vector_norm(quat_box_minus(end, start)).item()
    fraction = min(1.0, max_rotation_step_rad / max(rotation_error, 1e-12))

    # Interpolate orientation and combine it with the next position, preserving both inputs.
    result[3:7] = quat_slerp(start, end, fraction).numpy()
    return result
