# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bake a registered volume deformable's geometry into a reusable TetMesh USD.

Accepts procedural or USD-backed Arena assets with one volume simulation mesh and
one visual mesh, including the cube and Teddy Bear presets for Newton and PhysX.
Input is an AssetRegistry name, not a USD path. Surface deformables are unsupported.
The output contains geometry only; source materials and textures are not copied.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def bake_deformable_mesh(asset_name: str, output: Path) -> None:
    """Spawn an asset once and save its visual and simulation geometry without backend schemas.

    Args:
        asset_name: Registered Arena deformable asset name.
        output: Destination USD file; must not already exist.
    """
    import isaaclab.sim as sim_utils
    from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
    from isaaclab_physx.physics import PhysxCfg
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    assert not output.exists(), f"Refusing to overwrite an existing topology: {output}"
    asset = AssetRegistry().get_asset_by_name(asset_name)()
    assert isinstance(asset, DeformableObject), f"{asset_name} is not a deformable asset."
    physics = NewtonCfg(solver_cfg=VBDSolverCfg()) if asset.physics_preset is PhysicsBackend.NEWTON else PhysxCfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", physics=physics))
    try:
        cfg = asset.spawner_cfg
        root = cfg.func("/Source", cfg)
        prims = list(Usd.PrimRange(root))
        tetrahedra = [prim for prim in prims if prim.IsA(UsdGeom.TetMesh)]
        surfaces = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
        assert len(tetrahedra) == 1, "Baking requires exactly one volume simulation TetMesh."
        assert len(surfaces) == 1, "Baking requires exactly one visual Mesh."

        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(output))
        body = UsdGeom.Xform.Define(stage, "/Deformable")
        stage.SetDefaultPrim(body.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        source_tet = UsdGeom.TetMesh(tetrahedra[0])
        tet = UsdGeom.TetMesh.Define(stage, "/Deformable/simulation")
        tet.CreateTetVertexIndicesAttr(source_tet.GetTetVertexIndicesAttr().Get())
        tet.CreateSurfaceFaceVertexIndicesAttr(source_tet.GetSurfaceFaceVertexIndicesAttr().Get())
        tet.CreateOrientationAttr(source_tet.GetOrientationAttr().Get())
        tet.CreatePurposeAttr(UsdGeom.Tokens.guide)
        source_mesh = UsdGeom.Mesh(surfaces[0])
        mesh = UsdGeom.Mesh.Define(stage, "/Deformable/visual")
        mesh.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get())
        mesh.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        xform_cache = UsdGeom.XformCache()
        for source, destination in ((source_tet, tet), (source_mesh, mesh)):
            T_W_M = xform_cache.GetLocalToWorldTransform(source.GetPrim())
            points_W = []
            for point_M in source.GetPointsAttr().Get():
                points_W.append(Gf.Vec3f(T_W_M.Transform(Gf.Vec3d(point_M))))
            destination.CreatePointsAttr(points_W)
        stage.GetRootLayer().Save()
        print(f"Baked {asset_name}: {len(tet.GetPointsAttr().Get())} nodes -> {output}", flush=True)
    finally:
        sim.clear_instance()


def main() -> None:
    """Bake a volume asset with the current meshing settings in an initialized Kit runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        required=True,
        help="Registered Arena volume deformable name, e.g. deformable_cube_newton or deformable_teddy_bear_physx.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New .usd or .usda file to create.")
    args = parser.parse_args()
    app_launcher = AppLauncher({"headless": True})
    try:
        bake_deformable_mesh(args.asset, args.output.resolve())
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    main()
