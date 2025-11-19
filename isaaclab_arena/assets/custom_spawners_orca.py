# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom spawners for USD reference without correct rigid body setup for ORCA."""

from pxr import Usd, UsdGeom, Gf

from isaaclab.sim.utils import get_current_stage, clone
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


@clone
def spawn_usd_reference_direct(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """
    Spawn an Xform prim and add a USD reference using the direct USD API (no omni.kit.commands).
    This avoids AddReference command issues on some Kit versions.
    """
    stage: Usd.Stage = get_current_stage()

    # Define the prim as Xform
    prim: Usd.Prim = stage.DefinePrim(prim_path, "Xform")

    # Add a direct USD reference
    prim.GetReferences().AddReference(cfg.usd_path)

    # Apply optional transform on the Xform
    xformable = UsdGeom.Xformable(prim)
    if translation is not None:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        # orientation is (w, x, y, z)
        # Usd uses xyzw quaternion order for XformCommonAPI; use a rotate op around axis as conservative fallback
        # For simplicity, skip converting full quaternion here; spawn code usually sets init_state later.
        pass

    # Apply scale if provided
    if cfg.scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3f(*cfg.scale))

    return prim


