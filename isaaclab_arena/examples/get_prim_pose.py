# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%


from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%


import numpy as np
from typing import Tuple

from pxr import Usd, UsdGeom, UsdSkel

usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/assets_for_tests/reference_object_test_kitchen.usd"


def get_prim_pose_in_default_prim_frame(prim: Usd.Prim) -> tuple[tuple, tuple]:
    # Get the default prim of the stage
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Stage does not have a default prim set.")

    print(f"default_prim: {default_prim}, default_prim path: {default_prim.GetPath()}")

    # Compute prim's transform in default prim's local frame
    xformable_prim = UsdGeom.Xformable(prim)
    xformable_default = UsdGeom.Xformable(default_prim)

    prim_T_world = xformable_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    default_T_world = xformable_default.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # matrix_default_to_world may be singular if default prim is the pseudo-root. Warn user.
    import numpy as np

    if default_T_world.GetDeterminant() == 0:
        raise RuntimeError("Default prim's world transform is singular.")

    default_T_world = default_T_world.GetInverse()
    prim_T_default = prim_T_world * default_T_world

    pos, rot, _ = UsdSkel.DecomposeTransform(prim_T_default)
    rot_tuple = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
    pos_tuple = (pos[0], pos[1], pos[2])
    return pos_tuple, rot_tuple


stage = Usd.Stage.Open(usd_path)
prim = stage.GetPrimAtPath("/kitchen/_03_cracker_box")
print(f"prim: {prim}")

pos, rot = get_prim_pose_in_default_prim_frame(prim)
print(f"Position relative to default prim: {pos}")
print(f"Orientation (quaternion wxyz) relative to default prim: {rot}")

# %%

import numpy as np

pos_np = np.array(pos)
pos_np_gt = np.array((3.69020713150969, -0.804121657812894, 1.2531903565606817))

pos_np_diff = pos_np - pos_np_gt
print(f"Position difference: {pos_np_diff}")


# %%


prim_path = "{ENV_REGEX_NS}/kitchen/_03_cracker_box"

# Remove the ENV_REGEX_NS prefix
prim_path = prim_path.replace("{ENV_REGEX_NS}", "")
print(f"prim_path: {prim_path}")

# Get the prim from the stage
prim = stage.GetPrimAtPath(prim_path)
print(f"prim: {prim}")


# %%
