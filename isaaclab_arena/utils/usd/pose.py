# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdSkel

from isaaclab_arena.utils.pose import Pose


def get_prim_pose_in_default_prim_frame(
    prim: Usd.Prim, stage: Usd.Stage, prim_world_delta: np.ndarray | None = None
) -> Pose:
    """Get the pose of a prim in the default prim's local frame.

    Args:
        prim: The prim to get the pose of.
        stage: The stage to get the default prim from.
        prim_world_delta: Optional row-vector 4x4 motion of the prim in stage world space,
            as returned by compute_posed_prim_world_deltas.

    Returns:
        The pose of the prim in the default prim's local frame.
    """
    # Get the default prim of the stage
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Stage does not have a default prim set.")

    # Compute prim's transform in default prim's local frame
    xformable_prim = UsdGeom.Xformable(prim)
    xformable_default = UsdGeom.Xformable(default_prim)

    prim_T_world = xformable_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if prim_world_delta is not None:
        prim_T_world = prim_T_world * Gf.Matrix4d(prim_world_delta.tolist())
    default_T_world = xformable_default.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # matrix_default_to_world may be singular if default prim is the pseudo-root. Warn user.
    if default_T_world.GetDeterminant() == 0:
        raise RuntimeError("Default prim's world transform is singular.")

    default_T_world = default_T_world.GetInverse()
    prim_T_default = prim_T_world * default_T_world

    pos, rot, _ = UsdSkel.DecomposeTransform(prim_T_default)
    rot_tuple = (rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2], rot.GetReal())
    pos_tuple = (pos[0], pos[1], pos[2])
    return Pose(position_xyz=pos_tuple, rotation_xyzw=rot_tuple)
