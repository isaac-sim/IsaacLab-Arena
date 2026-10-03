# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID state conversion matching the GR00T N1.7 training convention."""

import numpy as np
from scipy.spatial.transform import Rotation


def droid_pose_to_eef9d(pose: np.ndarray) -> np.ndarray:
    """Convert batched base-frame XYZ/wxyz poses to GR00T's XYZ/rotation-6D state."""
    assert pose.ndim == 2 and pose.shape[1] == 7, "Expected (batch, 7) XYZ/wxyz poses"
    assert np.isfinite(pose).all(), "DROID end-effector poses must be finite"
    # DROID converts its measured quaternion to extrinsic xyz Euler angles.
    # GR00T intentionally reinterprets those values with TFG's XYZ convention,
    # applies the egocentric correction, then flattens the FIRST TWO ROWS.
    # Match gr00t/data/state_action/droid_frame.py at 51d4c89; directly taking
    # rows/columns of the simulator quaternion matrix would produce different state.
    euler = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_euler("xyz")
    correction = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    rotation = Rotation.from_euler("XYZ", euler).as_matrix() @ correction
    return np.concatenate((pose[:, :3], rotation[:, :2, :].reshape(-1, 6)), axis=-1).astype(np.float32)
