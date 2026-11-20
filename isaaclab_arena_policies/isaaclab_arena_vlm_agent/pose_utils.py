# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot-root pose helpers for VLM observation adapters."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def read_robot_root_poses_w(base, env_id: int) -> dict[str, np.ndarray]:
    """Return the world pose of each commanded asset root, as XYZ then XYZW, indexed by asset name."""
    roots = {}
    for name in base.action_manager.active_terms:
        asset_name = base.action_manager.get_term(name).cfg.asset_name
        if asset_name not in roots:
            roots[asset_name] = base.arena_world.get_pose_w(asset_name)[env_id].detach().cpu().numpy()
    return roots


def poses_in_robot_roots(
    roots: dict[str, np.ndarray], t_W_F: np.ndarray, q_W_F: np.ndarray
) -> dict[str, dict[str, list[float]]]:
    """Express a world pose of frame F in each robot root frame R.

    Args:
        roots: World poses T_W_R of robot roots, as XYZ then XYZW, indexed by asset name.
        t_W_F: Position of F in the world frame, in meters.
        q_W_F: XYZW rotation from F into the world frame.

    Returns:
        Position and XYZW quaternion of F in each root frame, indexed by asset name.
    """
    poses = {}
    for asset_name, T_W_R in roots.items():
        q_R_W = Rotation.from_quat(T_W_R[3:]).inv()
        poses[asset_name] = {
            "position_root_m": q_R_W.apply(t_W_F - T_W_R[:3]).tolist(),
            "quaternion_root_xyzw": (q_R_W * Rotation.from_quat(q_W_F)).as_quat().tolist(),
        }
    return poses
