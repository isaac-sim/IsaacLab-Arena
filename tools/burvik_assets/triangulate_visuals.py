# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Refresh Blender visual exports with explicit triangles, preserving tested physics."""

import bpy
import os
from pathlib import Path

from mathutils import Matrix
from pxr import Sdf, Usd, UsdGeom

root_dir = Path(os.environ.get("BURVIK_ASSET_DIR", Path(__file__).resolve().parents[2] / "artifacts/ikea_burvik"))
visual_collection = bpy.data.collections["BURVIK_Visuals"]
names = ["tray", "leg_1", "leg_2", "leg_3", "leg_4", "handle"]
for name in names:
    if name == "handle":
        helper = Path(__file__).with_name("recover_handle.py")
        exec(compile(helper.read_text(), str(helper), "exec"), {"__name__": "__main__", "__file__": str(helper)})
        continue
    filename = str(root_dir / "assets" / f"{name}.usd")
    previous = Sdf.Layer.CreateAnonymous()
    previous.TransferContent(Sdf.Layer.FindOrOpen(filename))
    old_stage = Usd.Stage.Open(previous)
    old_root = old_stage.GetPrimAtPath("/Asset")
    obj = bpy.data.objects[name]
    transform = obj.matrix_basis.copy()
    obj.matrix_basis = Matrix.Identity(4)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    for child in obj.children:
        if child.name in visual_collection.objects:
            child.select_set(True)
    bpy.context.view_layer.update()
    bpy.ops.wm.usd_export(
        filepath=filename,
        selected_objects_only=True,
        export_materials=True,
        generate_preview_surface=True,
        generate_materialx_network=False,
        export_textures_mode="NEW",
        overwrite_textures=True,
        relative_paths=True,
        root_prim_path="/Asset",
        export_animation=False,
        export_lights=False,
        export_cameras=False,
        triangulate_meshes=True,
    )
    obj.matrix_basis = transform
    layer = Sdf.Layer.FindOrOpen(filename)
    layer.Reload()
    stage = Usd.Stage.Open(layer)
    new_root = stage.GetPrimAtPath("/Asset")
    stage.SetDefaultPrim(new_root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    new_root.SetMetadata("apiSchemas", old_root.GetMetadata("apiSchemas"))
    new_root.SetCustomData(old_root.GetCustomData())
    for prop in old_root.GetAuthoredProperties():
        if not prop.GetName().startswith("xformOp"):
            Sdf.CopySpec(previous, prop.GetPath(), layer, prop.GetPath())
    for path in ["/Asset/Collisions", "/Asset/PhysicsMaterial"]:
        Sdf.CopySpec(previous, path, layer, path)
    counts = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and "/Collisions/" not in str(prim.GetPath()):
            counts.extend(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
    assert max(counts) == 3, f"{name} contains untriangulated visuals"
    layer.Save()
    print(name, "visual triangles", len(counts))
bpy.ops.wm.save_as_mainfile(filepath=str(root_dir / "burvik_source.blend"))
