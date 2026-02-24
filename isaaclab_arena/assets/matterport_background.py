# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Matterport 3D scene background for VLN tasks.

Uses standard ``spawn_from_usd`` for visual rendering of the Matterport
scene, plus an invisible ground plane at z=0 for collision. This provides
both correct rendering and physics without relying on mesh collision
(which doesn't work with the current IsaacLab GPU physics).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@clone
def _spawn_matterport_with_ground(prim_path, cfg, *args, **kwargs):
    """Spawn Matterport USD + ground plane + indoor lighting.

    Matterport 3D scans contain geometry and textures but no light
    sources.  This function adds everything needed to render and
    simulate inside a Matterport scene:
      - The USD scene itself (visual mesh with textures).
      - An invisible ground plane at z=0 for physics collision.
      - Indoor lighting matching NaVILA-Bench:
          DomeLight(500)  — ambient fill under ceilings
          DistantLight(1000) — directional light
          DiskLight x2(10000) — area lights at ceiling height
    """
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, *args, **kwargs)

    # Invisible ground plane at z=0 for physics collision
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        visible=False,
    )
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # Indoor lighting — Matterport scenes have no built-in lights.
    # Matches NaVILA-Bench h1_matterport_base_cfg.py lighting setup.
    sim_utils.DomeLightCfg(intensity=500.0, color=(1.0, 1.0, 1.0)).func(
        "/World/MatterportDomeLight", sim_utils.DomeLightCfg(intensity=500.0, color=(1.0, 1.0, 1.0))
    )
    sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/MatterportDistantLight", sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
    )
    disk_cfg = sim_utils.DiskLightCfg(intensity=10000.0, color=(1.0, 1.0, 1.0), radius=50.0)
    disk_cfg.func("/World/MatterportDisk1", disk_cfg)
    disk_cfg.func("/World/MatterportDisk2", disk_cfg)
    # Position the disk lights at ceiling height
    from pxr import UsdGeom, Gf
    stage = prim.GetStage()
    for path, pos in [("/World/MatterportDisk1", (0.0, 0.0, 2.6)), ("/World/MatterportDisk2", (-1.0, 0.0, 2.6))]:
        disk_prim = stage.GetPrimAtPath(path)
        if disk_prim.IsValid():
            xformable = UsdGeom.Xformable(disk_prim)
            ops = xformable.GetOrderedXformOps()
            for op in ops:
                if op.GetOpName() == "xformOp:translate":
                    op.Set(Gf.Vec3d(*pos))
                    break
            else:
                xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))

    print(f"[MatterportBackground] Scene at {prim_path} + ground plane + lighting")
    return prim


@register_asset
class MatterportBackground(LibraryBackground):
    """Matterport 3D scene with invisible ground plane for collision."""

    name = "matterport"
    tags = ["background"]
    usd_path = None
    initial_pose = Pose.identity()
    object_min_z = -0.5

    def __init__(self, usd_path: str):
        self.usd_path = usd_path
        self.spawn_cfg_addon = {"func": _spawn_matterport_with_ground}
        super().__init__()
