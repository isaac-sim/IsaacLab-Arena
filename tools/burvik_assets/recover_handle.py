# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the saved Blender handle with explicit triangles and its existing collision proxies."""

import bpy
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

root = Path(os.environ.get("BURVIK_ASSET_DIR", Path(__file__).resolve().parents[2] / "artifacts/ikea_burvik"))
stage = Usd.Stage.CreateNew(str(root / "assets" / "handle.usd"))
asset = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
stage.SetDefaultPrim(asset)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1)
UsdPhysics.RigidBodyAPI.Apply(asset)
mass = UsdPhysics.MassAPI.Apply(asset)
mass.CreateMassAttr(0.32)
mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.216))
mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0038, 0.008, 0.0043))
asset.AddAppliedSchema("PhysxRigidBodyAPI")
for key, typ, value in [
    ("solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
    ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 8),
    ("enableCCD", Sdf.ValueTypeNames.Bool, True),
    ("linearDamping", Sdf.ValueTypeNames.Float, 0.04),
    ("angularDamping", Sdf.ValueTypeNames.Float, 0.08),
]:
    asset.CreateAttribute("physxRigidBody:" + key, typ).Set(value)
physics = UsdShade.Material.Define(stage, "/Asset/PhysicsMaterial")
mat_api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
mat_api.CreateStaticFrictionAttr(0.65)
mat_api.CreateDynamicFrictionAttr(0.5)
mat_api.CreateRestitutionAttr(0)
materials = {}
for name, color, metallic in [
    ("WarmWhite", (0.86, 0.85, 0.78), 0),
    ("Steel", (0.4, 0.43, 0.45), 0.85),
]:
    material = UsdShade.Material.Define(stage, "/Asset/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, "/Asset/Materials/" + name + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3 if metallic == 0 else 0.24)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    materials[name] = material
count = 0
deps = bpy.context.evaluated_depsgraph_get()
for child in bpy.data.objects["handle"].children:
    is_collider = child.name.startswith("COL_")
    evaluated = child.evaluated_get(deps)
    data = evaluated.to_mesh()
    if not data:
        continue
    data.calc_loop_triangles()
    name = child.name.replace(".", "_")
    geometry = UsdGeom.Mesh.Define(stage, "/Asset/" + ("Collisions/" if is_collider else "Visuals/") + name)
    geometry.CreatePointsAttr([Gf.Vec3f(*(child.matrix_local @ v.co)) for v in data.vertices])
    geometry.CreateFaceVertexCountsAttr([3] * len(data.loop_triangles))
    geometry.CreateFaceVertexIndicesAttr([i for t in data.loop_triangles for i in t.vertices])
    geometry.CreateSubdivisionSchemeAttr("none")
    if is_collider:
        geometry.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr("convexHull")
        geometry.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        geometry.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.00015)
        geometry.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0)
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(physics, materialPurpose="physics")
        count += 1
    else:
        normal_matrix = child.matrix_local.to_3x3().inverted().transposed()
        geometry.CreateNormalsAttr([Gf.Vec3f(*(normal_matrix @ v.normal).normalized()) for v in data.vertices])
        geometry.SetNormalsInterpolation("vertex")
        key = "Steel" if "pin" in child.name.lower() else "WarmWhite"
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(materials[key])
    evaluated.to_mesh_clear()
assert count == 41, count
stage.GetRootLayer().Save()
print(
    "Recovered handle from saved Blender geometry; 41 original convex collision proxies.",
    flush=True,
)
