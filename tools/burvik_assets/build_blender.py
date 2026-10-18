# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the reference-derived BURVIK assets in the live Blender MCP session."""

import bpy
import json
import math
import numpy as np
import os
from pathlib import Path

from mathutils import Vector
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

OUT = Path(os.environ.get("BURVIK_ASSET_DIR", Path(__file__).resolve().parents[2] / "artifacts/ikea_burvik"))
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Preserve the user's original scene; work in a separate scene.
scene = bpy.data.scenes.new("BURVIK_Assembly_Assets")
bpy.context.window.scene = scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
visuals = bpy.data.collections.new("BURVIK_Visuals")
scene.collection.children.link(visuals)
collision_collection = bpy.data.collections.new("BURVIK_Collision_Proxies")
scene.collection.children.link(collision_collection)
collision_collection.hide_render = True


def material(name, color, roughness, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    mat.diffuse_color = (*color, 1)
    return mat


wood = material("Clear_lacquered_light_birch", (0.68, 0.45, 0.25), 0.36)
ivory = material("Warm_white_powder_coat", (0.86, 0.85, 0.78), 0.30)
steel = material("Smooth_steel_pins", (0.40, 0.43, 0.45), 0.24, 0.85)
rubber = material("Warm_grey_foot_caps", (0.24, 0.25, 0.22), 0.72)

# A small procedural image exports as a portable UsdPreviewSurface texture.
n = 1024
xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
rng = np.random.default_rng(7)
grain = 0.018 * np.sin(yy * 350 + 5 * np.sin(xx * 10))
grain += 0.009 * np.sin(yy * 1300 + 4 * np.sin(xx * 21))
grain += 0.005 * rng.normal(size=(n, n))
rgba = np.ones((n, n, 4), dtype=np.float32)
for k, base in enumerate((0.79, 0.64, 0.46)):
    rgba[:, :, k] = np.clip(base + grain, 0, 1)
tex_image = bpy.data.images.new("Birch_grain", width=n, height=n)
tex_image.pixels.foreach_set(rgba.ravel())
tex_image.filepath_raw = str(ASSETS / "birch_grain.png")
tex_image.file_format = next(
    i.identifier for i in tex_image.bl_rna.properties["file_format"].enum_items if i.identifier == "PNG"
)
tex_image.save()
tex = wood.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = tex_image
node = next(n for n in wood.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
wood.node_tree.links.new(tex.outputs["Color"], node.inputs["Base Color"])

parts = {}
colliders = {}


def adopt(obj, name, part, mat=None):
    obj.name = name
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    visuals.objects.link(obj)
    obj.parent = parts[part]
    if mat:
        obj.data.materials.append(mat)
    return obj


def new_part(name):
    obj = bpy.data.objects.new(name, None)
    visuals.objects.link(obj)
    parts[name] = obj
    colliders[name] = []


def mesh(name, verts, faces, part, mat):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    return adopt(bpy.data.objects.new(name, data), name, part, mat)


def bevel(obj, amount=0.0006):
    mod = obj.modifiers.new("Manufactured_edge_radius", "BEVEL")
    mod.width = amount
    mod.segments = 3
    return obj


def tube(name, a, b, r1, r2, part, mat, vertices=64, edge=True):
    a, b = Vector(a), Vector(b)
    axis = b - a
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=r1,
        radius2=r2,
        depth=axis.length,
        location=(a + b) / 2,
    )
    obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = axis.to_track_quat("Z", "Y")
    adopt(obj, name, part, mat)
    for face in obj.data.polygons:
        face.use_smooth = len(face.vertices) == 4
    if edge:
        bevel(obj, min(0.00045, r1 / 8))
    return obj


def boolean_hole(obj, a, b, radius):
    cutter = tube("Temporary_hole_tool", a, b, radius, radius, obj.parent.name, None, edge=False)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Open_socket", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def convex(part, name, verts, faces):
    colliders[part].append({"name": name, "vertices": [list(v) for v in verts], "faces": faces})
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("COL_" + name, data)
    collision_collection.objects.link(obj)
    obj.parent = parts[part]
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj.hide_set(True)


def prism(part, name, polygon, low, high):
    count = len(polygon)
    verts = [(x, y, low) for x, y in polygon] + [(x, y, high) for x, y in polygon]
    faces = [list(reversed(range(count))), list(range(count, count * 2))]
    for i in range(count):
        j = (i + 1) % count
        faces.append([i, j, j + count, i + count])
    convex(part, name, verts, faces)


def rod_collider(part, name, a, b, r1, r2, sides=24):
    a, b = Vector(a), Vector(b)
    rot = (b - a).to_track_quat("Z", "Y")
    verts = []
    for center, radius in ((a, r1), (b, r2)):
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            verts.append(tuple(center + rot @ Vector((radius * math.cos(angle), radius * math.sin(angle), 0))))
    faces = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, j + sides, i + sides])
    convex(part, name, verts, faces)


def annulus(part, name, center, axis, inner, outer, length, segments=32):
    center, axis = Vector(center), Vector(axis).normalized()
    rot = axis.to_track_quat("Z", "Y")
    for i in range(segments):
        t0, t1 = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
        xy = [
            (inner * math.cos(t0), inner * math.sin(t0)),
            (outer * math.cos(t0), outer * math.sin(t0)),
            (outer * math.cos(t1), outer * math.sin(t1)),
            (inner * math.cos(t1), inner * math.sin(t1)),
        ]
        verts = []
        for z in (-length / 2, length / 2):
            for x, y in xy:
                verts.append(tuple(center + rot @ Vector((x, y, z))))
        convex(
            part,
            f"{name}_{i:02}",
            verts,
            [
                [3, 2, 1, 0],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],
        )


def clip(poly, axis, limit, keep_lower):
    result = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        va, vb = a[axis] - limit, b[axis] - limit
        inside_a = va <= 1e-10 if keep_lower else va >= -1e-10
        inside_b = vb <= 1e-10 if keep_lower else vb >= -1e-10
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            fraction = va / (va - vb)
            result.append(tuple(a[j] + fraction * (b[j] - a[j]) for j in range(2)))
    return result


new_part("tray")
rim = tube(
    "Bent_birch_tray_rim",
    (0, 0, 0),
    (0, 0, 0.060),
    0.19,
    0.19,
    "tray",
    wood,
    160,
    False,
)
boolean_hole(rim, (0, 0, -0.01), (0, 0, 0.07), 0.184)
bevel(rim, 0.0008)
floor = tube(
    "Recessed_birch_tray_floor",
    (0, 0, 0.023),
    (0, 0, 0.035),
    0.185,
    0.185,
    "tray",
    wood,
    160,
    False,
)

sockets = []
for i in range(4):
    angle = i * math.pi / 2
    radial = Vector((math.cos(angle), math.sin(angle), 0))
    seat = radial * 0.162 + Vector((0, 0, 0.004))
    foot = radial * 0.200 + Vector((0, 0, -0.420))
    axis = (seat - foot).normalized()
    sockets.append({"seat": list(seat), "axis": list(axis), "foot": list(foot)})
    boss = tube(
        f"Leg_socket_boss_{i + 1}",
        seat,
        seat + axis * 0.024,
        0.020,
        0.020,
        "tray",
        wood,
        64,
        False,
    )
    boolean_hole(boss, seat - axis * 0.004, seat + axis * 0.030, 0.0046)
    bevel(boss, 0.00035)
    annulus("tray", f"socket_{i + 1}", seat + axis * 0.012, axis, 0.0046, 0.020, 0.024, 24)
    if i in (0, 2):
        boolean_hole(floor, seat - axis * 0.01, seat + axis * 0.07, 0.0054)
bevel(floor, 0.0004)
annulus("tray", "rim", (0, 0, 0.03), (0, 0, 1), 0.184, 0.19, 0.06, 64)

# Convex floor partition: holes stay open in both visual and collision geometry.
hole_x = 0.162 - 0.038 / 0.424 * (0.029 - 0.004)
xs = [-0.19, -hole_x - 0.014, -hole_x + 0.014, hole_x - 0.014, hole_x + 0.014, 0.19]
ys = [-0.19, -0.014, 0.014, 0.19]
circle = [(0.184 * math.cos(i * 2 * math.pi / 128), 0.184 * math.sin(i * 2 * math.pi / 128)) for i in range(128)]
for ix in range(len(xs) - 1):
    for iy in range(len(ys) - 1):
        if ix in (1, 3) and iy == 1:
            center_x = -hole_x if ix == 1 else hole_x
            for k in range(32):
                angles = (2 * math.pi * k / 32, 2 * math.pi * (k + 1) / 32)
                inner, outer = [], []
                for angle in angles:
                    co, si = math.cos(angle), math.sin(angle)
                    d = 0.014 / max(abs(co), abs(si))
                    inner.append((center_x + 0.0054 * co, 0.0054 * si))
                    outer.append((center_x + d * co, d * si))
                prism(
                    "tray",
                    f"floor_socket_{ix}_{k}",
                    [inner[0], outer[0], outer[1], inner[1]],
                    0.023,
                    0.035,
                )
        else:
            polygon = circle
            for ax, lim, lower in [
                (0, xs[ix], False),
                (0, xs[ix + 1], True),
                (1, ys[iy], False),
                (1, ys[iy + 1], True),
            ]:
                if polygon:
                    polygon = clip(polygon, ax, lim, lower)
            if len(polygon) >= 3:
                prism("tray", f"floor_tile_{ix}_{iy}", polygon, 0.023, 0.035)

# Each leg's local origin is its shoulder; +Z follows its insertion axis.
for i in range(4):
    part = f"leg_{i + 1}"
    new_part(part)
    length = (Vector(sockets[i]["seat"]) - Vector(sockets[i]["foot"])).length
    tube(
        "Tapered_powder_coated_leg",
        (0, 0, -length + 0.004),
        (0, 0, 0),
        0.0055,
        0.0095,
        part,
        ivory,
    )
    tube(
        "Protective_foot",
        (0, 0, -length),
        (0, 0, -length + 0.008),
        0.0058,
        0.0058,
        part,
        rubber,
    )
    tube("Unthreaded_push_pin", (0, 0, 0), (0, 0, 0.017), 0.004, 0.004, part, steel)
    tube(
        "Pin_lead_in_chamfer",
        (0, 0, 0.017),
        (0, 0, 0.018),
        0.004,
        0.0034,
        part,
        steel,
        edge=False,
    )
    rod_collider(part, "tapered_leg", (0, 0, -length + 0.004), (0, 0, 0), 0.0055, 0.0095)
    rod_collider(part, "foot", (0, 0, -length), (0, 0, -length + 0.008), 0.0058, 0.0058)
    rod_collider(part, "pin", (0, 0, 0), (0, 0, 0.017), 0.004, 0.004, 32)
    rod_collider(part, "pin_tip", (0, 0, 0.017), (0, 0, 0.018), 0.004, 0.0034, 32)

new_part("handle")
# Smooth rounded polygon reproducing the flattened crown and angled shoulders.
control = [
    (-0.1588, 0.040),
    (-0.144, 0.223),
    (-0.125, 0.270),
    (-0.054, 0.342),
    (-0.037, 0.354),
    (0.037, 0.354),
    (0.054, 0.342),
    (0.125, 0.270),
    (0.144, 0.223),
    (0.1588, 0.040),
]
points = [Vector((x, 0, z)) for x, z in control]
for _ in range(2):
    smooth = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        smooth.extend([a * 0.75 + b * 0.25, a * 0.25 + b * 0.75])
    smooth.append(points[-1])
    points = smooth
# The rounded crown's outer surface is exactly 0.360 m above tray origin.
top = max(p.z for p in points)
for p in points:
    p.z = 0.04 + (p.z - 0.04) * (0.354 - 0.04) / (top - 0.04)
curve = bpy.data.curves.new("Bent_handle_centerline", "CURVE")
curve.dimensions = "3D"
curve.bevel_depth = 0.006
curve.bevel_resolution = 4
spline = curve.splines.new("POLY")
spline.points.add(len(points) - 1)
for dst, src in zip(spline.points, points):
    dst.co = (*src, 1)
obj = bpy.data.objects.new("Rounded_carry_handle", curve)
adopt(obj, obj.name, "handle", ivory)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.convert(target="MESH")
obj.select_set(False)
for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
    rod_collider("handle", f"handle_tube_{i}", a, b, 0.006, 0.006, 12)
for i in (0, 2):
    seat, axis = Vector(sockets[i]["seat"]), Vector(sockets[i]["axis"])
    base = seat + axis * (0.036 / axis.z)
    bottom = base - axis * 0.014
    tube(f"Handle_push_pin_{i}", bottom, base, 0.004, 0.004, "handle", steel)
    rod_collider("handle", f"handle_pin_{i}", bottom, base, 0.004, 0.004, 32)

# Export local-coordinate assets; physics is authored on the single default root.
masses = {"tray": 1.25, "handle": 0.32, **{f"leg_{i + 1}": 0.24 for i in range(4)}}
for name, root in parts.items():
    if name == "handle":
        helper = Path(__file__).with_name("recover_handle.py")
        exec(compile(helper.read_text(), str(helper), "exec"), {"__name__": "__main__", "__file__": str(helper)})
        continue
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for child in root.children:
        if child.name in visuals.objects:
            child.select_set(True)
    path = ASSETS / f"{name}.usd"
    bpy.ops.wm.usd_export(
        filepath=str(path),
        selected_objects_only=True,
        export_materials=True,
        generate_preview_surface=True,
        generate_materialx_network=False,
        export_textures_mode="NEW",
        relative_paths=True,
        root_prim_path="/Asset",
        export_animation=False,
        export_lights=False,
        export_cameras=False,
        triangulate_meshes=True,
    )
    stage = Usd.Stage.Open(str(path))
    asset = stage.GetPrimAtPath("/Asset")
    stage.SetDefaultPrim(asset)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.RigidBodyAPI.Apply(asset)
    mass = UsdPhysics.MassAPI.Apply(asset)
    mass.CreateMassAttr(masses[name])
    if name.startswith("leg"):
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, -0.20))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00365, 0.00365, 0.000009))
    elif name == "tray":
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.029))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.012, 0.012, 0.023))
    else:
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.216))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0038, 0.008, 0.0043))
    asset.AddAppliedSchema("PhysxRigidBodyAPI")
    for attr, typ, value in [
        ("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
        ("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 8),
        ("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool, True),
        ("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, 0.04),
        ("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, 0.08),
    ]:
        asset.CreateAttribute(attr, typ).Set(value)
    phys_mat = UsdShade.Material.Define(stage, "/Asset/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.65)
    pm.CreateDynamicFrictionAttr(0.5)
    pm.CreateRestitutionAttr(0)
    UsdGeom.Scope.Define(stage, "/Asset/Collisions")
    for c in colliders[name]:
        geo = UsdGeom.Mesh.Define(stage, "/Asset/Collisions/" + c["name"])
        geo.CreatePointsAttr([Gf.Vec3f(*v) for v in c["vertices"]])
        geo.CreateFaceVertexCountsAttr([len(f) for f in c["faces"]])
        geo.CreateFaceVertexIndicesAttr([v for f in c["faces"] for v in f])
        geo.CreateSubdivisionSchemeAttr("none")
        geo.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geo.GetPrim()).CreateApproximationAttr("convexHull")
        geo.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        geo.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.00015)
        geo.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(phys_mat, materialPurpose="physics")
    asset.SetCustomDataByKey("burvik:mass_source", "Engineering estimate; not measured")
    asset.SetCustomDataByKey("burvik:connection", "Smooth pin; latch joint only after verified seating")
    stage.GetRootLayer().Save()

# Assembled preview in Blender, retaining separate editable rigid parts.
for i in range(4):
    root = parts[f"leg_{i + 1}"]
    root.location = Vector(sockets[i]["seat"]) + Vector((0, 0, 0.42))
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Vector(sockets[i]["axis"]).to_track_quat("Z", "Y")
parts["tray"].location.z = 0.42
parts["handle"].location.z = 0.42
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.region_3d.view_distance = 1.5
        area.spaces.active.region_3d.view_location = (0, 0, 0.39)
scene.world = bpy.data.worlds.new("BURVIK_Studio_World")
scene.world.color = (0.3, 0.3, 0.3)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "burvik_source.blend"))
spec = {
    "diameter_m": 0.38,
    "overall_height_m": 0.78,
    "rim_height_m": 0.06,
    "tray_origin_height_m": 0.42,
    "pin_diameter_m": 0.008,
    "socket_diameter_m": 0.0092,
    "radial_clearance_m": 0.0006,
    "leg_insertion_m": 0.018,
    "handle_insertion_m": 0.014,
    "masses_kg": masses,
    "sockets": sockets,
    "collider_counts": {k: len(v) for k, v in colliders.items()},
    "notes": [
        "Exterior dimensions use supplied centimetre labels.",
        "Leg length, splay, wall thickness, sockets, mass and inertia are estimated.",
        "Threaded connectors replaced with chamfered smooth pins and seated latches.",
        "Physics uses convex pieces around open bores; no convex hull closes a socket.",
    ],
}
(OUT / "specification.json").write_text(json.dumps(spec, indent=2))
print(
    json.dumps({
        "assets": list(parts),
        "colliders": spec["collider_counts"],
        "output": str(OUT),
    })
)
