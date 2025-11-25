# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pxr import Usd, UsdGeom, UsdSkel

from isaaclab_arena.utils.pose import Pose


def get_prim_pose_in_default_prim_frame(prim: Usd.Prim, stage: Usd.Stage) -> Pose:
    """Get the pose of a prim in the default prim's local frame.

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

    # Compute prim's transform in default prim's local frame
    xformable_prim = UsdGeom.Xformable(prim)
    xformable_default = UsdGeom.Xformable(default_prim)

    prim_T_world = xformable_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    default_T_world = xformable_default.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # matrix_default_to_world may be singular if default prim is the pseudo-root. Warn user.
    if default_T_world.GetDeterminant() == 0:
        raise RuntimeError("Default prim's world transform is singular.")

    default_T_world = default_T_world.GetInverse()
    prim_T_default = prim_T_world * default_T_world

    pos, rot, _ = UsdSkel.DecomposeTransform(prim_T_default)
    rot_tuple = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
    pos_tuple = (pos[0], pos[1], pos[2])
    return Pose(position_xyz=pos_tuple, rotation_wxyz=rot_tuple)
