# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fit Arena library visuals to the workcell's physical dimensions."""

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena.assets.object_library import (
    BrownBox,
    CrackerBox,
    GregorysCoffeeCupObjaverseRobolab,
    OfficeTable,
    PlateLargeVompRobolab,
    PurpleCrate,
    SugarBox,
)
from isaaclab_arena_environments.spatial_memory.workcells.geometry import COLORS


def _fit_asset(stage, path, asset, bounds, *, fixed_prop=False):
    """Fit the visible USD to parent-frame bounds and leave contact handling to the workcell."""
    mount = UsdGeom.Xform.Define(stage, path)
    model = stage.DefinePrim(path.AppendChild("Model"))
    model.GetReferences().AddReference(asset.usd_path)
    assert model.GetChildren(), f"Could not compose {asset.usd_path}"
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    source = cache.ComputeLocalBound(model).ComputeAlignedRange()
    assert not source.IsEmpty() and min(source.GetSize()) > 0, asset.usd_path
    scale = Gf.CompDiv(bounds.GetSize(), source.GetSize())
    mount.AddTranslateOp().Set(bounds.GetMidpoint() - Gf.CompMult(source.GetMidpoint(), scale))
    mount.AddScaleOp().Set(scale)
    remove = [UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI]
    if not fixed_prop:
        remove.extend((UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI))
    for prim in Usd.PrimRange(model):
        for api in remove:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)


def _finish_hardware(stage, root):
    """Bind shared metal finishes to the workcell frame and exposed steel fixtures."""
    finishes = {}
    for key, color, metallic, roughness in (
        ("frame", (0.10, 0.14, 0.16), 0.35, 0.40),
        ("steel", (0.52, 0.55, 0.58), 0.83, 0.30),
    ):
        path = root.AppendChild("Materials").AppendChild(key)
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        finishes[Gf.Vec3f(*COLORS[key])] = material
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Gprim):
            colors = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if colors and colors[0] in finishes:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(finishes[colors[0]])


def add_library_visuals(stage, scene):
    """Dress the registered workcell after authoring its bodies, supports and joints."""
    root = stage.GetDefaultPrim().GetPath()
    _finish_hardware(stage, root)
    fixtures = root.AppendChild("Fixtures")
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    top = stage.GetPrimAtPath(fixtures.AppendChild("BenchTop"))
    if top:
        bounds = cache.ComputeLocalBound(top).ComputeAlignedRange()
        bounds.SetMin(Gf.Vec3d(bounds.GetMin()[0], bounds.GetMin()[1], 0))
        _fit_asset(stage, fixtures.AppendChild("OfficeTable"), OfficeTable, bounds)
        for prim in stage.GetPrimAtPath(fixtures).GetChildren():
            if prim == top or prim.GetName().startswith("BenchLeg"):
                UsdGeom.Imageable(prim).MakeInvisible()

    # Anchor each model in the collision shape frame so a tilted dish or parcel follows its body.
    for body, shape, asset in (
        ("plate", "HeldPlate", PlateLargeVompRobolab),
        ("neighbor", "Neighbor", PlateLargeVompRobolab),
        ("cup0", "Cup0", GregorysCoffeeCupObjaverseRobolab),
        ("cup1", "Cup1", GregorysCoffeeCupObjaverseRobolab),
        ("load", "load", BrownBox),
        ("load", "TallItem", BrownBox),
        ("crate", "HeldCrate", BrownBox),
        ("crate", "Crate", BrownBox),
        ("crate", "UpperCrate", BrownBox),
        ("parcel", "Parcel", BrownBox),
        ("stock_a", "stock_a", CrackerBox),
        ("stock_b", "stock_b", SugarBox),
        ("stock_c", "stock_c", CrackerBox),
    ):
        parent = root.AppendChild("Body_" + body)
        proxy = stage.GetPrimAtPath(parent.AppendChild(shape))
        if proxy:
            frame = UsdGeom.Xform.Define(stage, parent.AppendChild("Appearance"))
            frame.AddTransformOp().Set(UsdGeom.Xformable(proxy).GetLocalTransformation())
            bounds = cache.ComputeUntransformedBound(proxy).ComputeAlignedRange()
            _fit_asset(stage, frame.GetPath().AppendChild("Visual"), asset, bounds)
            UsdGeom.Imageable(proxy).MakeInvisible()

    if scene == "tote_extraction":
        tote = stage.GetPrimAtPath(root.AppendChild("Body_tote"))
        bounds = Gf.Range3d()
        for prim in tote.GetChildren():
            if prim.GetName() != "Grip":
                bounds.UnionWith(cache.ComputeLocalBound(prim).ComputeAlignedRange())
                UsdGeom.Imageable(prim).MakeInvisible()
        _fit_asset(stage, tote.GetPath().AppendChild("Bin"), PurpleCrate, bounds)

    if scene in {"panel_corner", "aisle_retreat", "trolley_dock", "pillar_pass"}:
        # A packing station beyond the aisle gives the four transport fixtures a shared warehouse setting.
        table = Gf.Range3d(Gf.Vec3d(-0.8, 1.55, 0), Gf.Vec3d(0.8, 2.35, 0.76))
        _fit_asset(stage, fixtures.AppendChild("PackingTable"), OfficeTable, table, fixed_prop=True)
        carton = Gf.Range3d(Gf.Vec3d(-0.50, 1.68, 0.76), Gf.Vec3d(-0.10, 2.02, 1.04))
        _fit_asset(stage, fixtures.AppendChild("PackedCarton"), BrownBox, carton, fixed_prop=True)
