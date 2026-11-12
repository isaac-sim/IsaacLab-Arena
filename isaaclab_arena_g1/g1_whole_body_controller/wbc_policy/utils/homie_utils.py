# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and quaternion helpers for the OpenHomie policy."""

import numpy as np
import yaml
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file, converting the default angles, command scale and initial command to arrays.

    Args:
        config_path: Path to the YAML file.

    Returns:
        The configuration dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert lists to numpy arrays where needed
    array_keys = ["default_angles", "cmd_scale", "cmd_init"]
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    return config


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector `v` by the inverse of quaternion `q`.

    Args:
        q: Quaternion in wxyz order.
        v: Vector of shape (3,).

    Returns:
        The rotated vector.
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array([
        v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
        + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
        + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
        + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
        + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
        + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
        + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
    ])


def get_gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Get the gravity direction expressed in the body frame.

    Args:
        quat: Body orientation quaternion in wxyz order.

    Returns:
        The unit gravity vector in the body frame.
    """
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)
