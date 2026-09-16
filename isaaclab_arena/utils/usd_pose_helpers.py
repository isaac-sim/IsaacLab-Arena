# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pxr import Usd, UsdGeom, UsdSkel

from isaaclab_arena.utils.pose import Pose


def get_prim_pose_in_default_prim_frame(prim: Usd.Prim, stage: Usd.Stage) -> Pose:
    """Get the pose of a prim in the default prim's local frame.

    Scale is intentionally omitted because ``Pose`` represents only a rigid transform.

    Args:
        prim: The prim to get the pose of.
        stage: The stage to get the default prim from.

    Returns:
        The pose of the prim in the default prim's local frame.
    """
    # Get the default prim of the stage
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Stage does not have a default prim set.")

    # O is the prim frame, P the default-prim frame, and W the USD world frame.
    xformable_prim = UsdGeom.Xformable(prim)
    xformable_default = UsdGeom.Xformable(default_prim)

    T_W_O = xformable_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    T_W_P = xformable_default.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # A singular transform cannot define a reference frame.
    if T_W_P.GetDeterminant() == 0:
        raise RuntimeError("Default prim's world transform is singular.")

    T_P_W = T_W_P.GetInverse()
    # USD uses row vectors, reversing the multiplication order of our frame notation.
    T_P_O = T_W_O * T_P_W

    pos, rot, _ = UsdSkel.DecomposeTransform(T_P_O)
    rot_tuple = (rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2], rot.GetReal())
    pos_tuple = (pos[0], pos[1], pos[2])
    return Pose(position_xyz=pos_tuple, rotation_xyzw=rot_tuple)
