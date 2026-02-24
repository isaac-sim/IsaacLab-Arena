# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Matterport 3D scene background for VLN tasks.

This background can be spawned as a shared global scene at
``/World/matterport`` so the Matterport USD is not cloned per environment.
We keep the invisible ground plane as an optional fallback while validating
whether the referenced USD mesh participates in PhysX collision reliably.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.utils import clone
from pxr import Usd, UsdGeom

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


def _get_mesh_collision_cfg(mesh_collider_approximation: str):
    """Return the IsaacLab mesh-collision config for the requested approximation."""
    if mesh_collider_approximation == "triangle":
        return sim_utils.TriangleMeshPropertiesCfg()
    if mesh_collider_approximation == "convex_decomposition":
        return sim_utils.ConvexDecompositionPropertiesCfg()
    if mesh_collider_approximation == "sdf":
        return sim_utils.SDFMeshPropertiesCfg()
    raise ValueError(
        "Unsupported mesh collider approximation "
        f"'{mesh_collider_approximation}'. Expected one of: triangle, convex_decomposition, sdf."
    )


def _iter_descendant_prims(root_prim):
    """Yield descendant prims, including children hidden behind instance proxies."""
    for child in root_prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
        yield child
        yield from _iter_descendant_prims(child)


def _apply_explicit_mesh_colliders(stage, root_prim_path: str, physics_material_path: str, approximation: str) -> int:
    """Apply explicit collider schemas to all descendant Mesh prims."""
    mesh_collision_cfg = _get_mesh_collision_cfg(approximation)
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        return 0
    mesh_count = 0
    for prim in _iter_descendant_prims(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = str(prim.GetPath())
        sim_utils.define_collision_properties(
            prim_path,
            sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            stage=stage,
        )
        sim_utils.define_mesh_collision_properties(prim_path, mesh_collision_cfg, stage=stage)
        sim_utils.bind_physics_material(prim_path, physics_material_path, stage=stage)
        mesh_count += 1
    return mesh_count


def _spawn_collision_overlay(
    prim_path: str,
    stage,
    usd_path: str,
    physics_material_path: str,
    approximation: str,
) -> int:
    """Spawn a hidden collision-only USD and apply mesh colliders to it."""
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    collision_prim_path = f"{prim_path}/collision"
    collision_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    collision_prim = _spawn_from_usd_file(collision_prim_path, usd_path, collision_cfg)
    if collision_prim.IsValid():
        UsdGeom.Imageable(collision_prim).MakeInvisible()
    sim_utils.define_collision_properties(
        collision_prim_path,
        sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        stage=stage,
    )
    sim_utils.bind_physics_material(collision_prim_path, physics_material_path, stage=stage)
    return _apply_explicit_mesh_colliders(
        stage=stage,
        root_prim_path=collision_prim_path,
        physics_material_path=physics_material_path,
        approximation=approximation,
    )


@clone
def _spawn_matterport_with_ground(prim_path, cfg, *args, **kwargs):
    """Spawn Matterport USD + optional ground plane + indoor lighting.

    Matterport 3D scans contain geometry and textures but no light
    sources.  This function adds everything needed to render and
    simulate inside a Matterport scene:
      - The USD scene itself (visual mesh with textures).
      - Optional invisible ground plane for fallback collision support.
      - Indoor lighting matching NaVILA-Bench:
          DomeLight(500)  — ambient fill under ceilings
          DistantLight(1000) — directional light
          DiskLight x2(10000) — area lights at ceiling height
    """
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, *args, **kwargs)

    # Try the same top-level collider setup used by NaVILA-Bench.
    collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)

    physics_mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )
    mat_path = f"{prim_path}/matterportPhysicsMaterial"
    physics_mat.func(mat_path, physics_mat)
    sim_utils.bind_physics_material(prim.GetPrimPath(), mat_path)

    explicit_mesh_colliders = getattr(cfg, "explicit_mesh_colliders", False)
    mesh_collider_approximation = getattr(cfg, "mesh_collider_approximation", "triangle")
    explicit_mesh_count = 0
    if explicit_mesh_colliders:
        explicit_mesh_count = _apply_explicit_mesh_colliders(
            stage=prim.GetStage(),
            root_prim_path=str(prim.GetPrimPath()),
            physics_material_path=mat_path,
            approximation=mesh_collider_approximation,
        )

    collision_overlay_usd_path = getattr(cfg, "collision_overlay_usd_path", None)
    collision_overlay_mesh_count = 0
    if collision_overlay_usd_path:
        collision_overlay_mesh_count = _spawn_collision_overlay(
            prim_path=prim_path,
            stage=prim.GetStage(),
            usd_path=collision_overlay_usd_path,
            physics_material_path=mat_path,
            approximation=mesh_collider_approximation,
        )

    ground_plane_z = getattr(cfg, "ground_plane_z", 0.0)
    if ground_plane_z is not None:
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=physics_mat,
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
    from pxr import Gf
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

    ground_str = f"ground(z={ground_plane_z})" if ground_plane_z is not None else "ground(disabled)"
    collider_str = "top-level collision"
    if explicit_mesh_colliders:
        collider_str += f" + explicit-mesh({explicit_mesh_count}, {mesh_collider_approximation})"
    if collision_overlay_usd_path:
        collider_str += f" + collision-overlay({collision_overlay_mesh_count}, {mesh_collider_approximation})"
    print(f"[MatterportBackground] {prim_path} + {collider_str} + {ground_str} + lighting")
    return prim


@register_asset
class MatterportBackground(LibraryBackground):
    """Matterport 3D scene shared globally across environments."""

    name = "matterport"
    tags = ["background"]
    usd_path = None
    initial_pose = Pose.identity()
    object_min_z = -0.5

    def __init__(
        self,
        usd_path: str,
        ground_plane_z: float | None = 0.0,
        use_global_prim: bool = False,
        explicit_mesh_colliders: bool = False,
        mesh_collider_approximation: str = "triangle",
        collision_overlay_usd_path: str | None = None,
    ):
        self.usd_path = usd_path
        self.ground_plane_z = ground_plane_z
        self.use_global_prim = use_global_prim
        self.explicit_mesh_colliders = explicit_mesh_colliders
        self.mesh_collider_approximation = mesh_collider_approximation
        self.collision_overlay_usd_path = collision_overlay_usd_path
        self.spawn_cfg_addon = {"func": _spawn_matterport_with_ground}
        self.asset_cfg_addon = {"collision_group": -1} if use_global_prim else {}
        prim_path = "/World/matterport" if use_global_prim else None
        super().__init__(prim_path=prim_path)

    def _generate_base_cfg(self) -> AssetBaseCfg:
        """Generate an AssetBaseCfg, optionally using a shared global prim path."""
        prim_path = "/World/matterport" if self.use_global_prim else self.prim_path
        object_cfg = AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                **self.spawn_cfg_addon,
            ),
            **self.asset_cfg_addon,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        object_cfg.spawn.ground_plane_z = self.ground_plane_z
        object_cfg.spawn.explicit_mesh_colliders = self.explicit_mesh_colliders
        object_cfg.spawn.mesh_collider_approximation = self.mesh_collider_approximation
        object_cfg.spawn.collision_overlay_usd_path = self.collision_overlay_usd_path
        return object_cfg
