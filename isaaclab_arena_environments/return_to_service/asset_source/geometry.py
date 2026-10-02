# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Blender geometry and USD export primitives for the service-cell assets."""

from __future__ import annotations

import bpy
import json
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from mathutils import Euler, Vector
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics

from .uv import UV_DOUBLED_AREA_TOLERANCE, surface_area_tolerance

PALETTE = {
    "navy": ((0.025, 0.050, 0.079), 0.38, 0.05),
    "teal": ((0.025, 0.34, 0.32), 0.32, 0.08),
    "orange": ((0.96, 0.29, 0.065), 0.30, 0.0),
    "rubber": ((0.022, 0.029, 0.032), 0.78, 0.0),
    "steel": ((0.48, 0.54, 0.60), 0.27, 0.86),
    "ivory": ((0.77, 0.82, 0.82), 0.36, 0.0),
    "filter_paper": ((0.76, 0.70, 0.55), 0.91, 0.0),
    "foam": ((0.10, 0.13, 0.15), 0.95, 0.0),
    "screen": ((0.016, 0.046, 0.054), 0.18, 0.12),
    "cyan": ((0.10, 0.74, 0.73), 0.32, 0.0),
    "amber": ((1.0, 0.57, 0.035), 0.30, 0.0),
    "red": ((0.62, 0.075, 0.035), 0.42, 0.0),
    "worktop": ((0.38, 0.43, 0.43), 0.72, 0.0),
    "floor": ((0.24, 0.27, 0.29), 0.88, 0.0),
    "debris": ((0.46, 0.27, 0.10), 0.89, 0.0),
}


def make_materials(output_dir: Path) -> dict[str, bpy.types.Material]:
    """Create original UV-mapped PBR textures with deterministic microstructure.

    Args:
        output_dir: Directory receiving the generated texture PNGs.

    Returns:
        Named materials with USD-compatible image texture nodes.
    """
    texture_dir = output_dir / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(73129)
    materials = {}
    size = 512
    for name, (color, roughness, metallic) in PALETTE.items():
        material = bpy.data.materials.new(f"RTS_{name}")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Roughness"].default_value = roughness
        fine = rng.normal(0, 0.012 if name != "foam" else 0.055, (size, size))
        channels = {
            "base_color": np.clip(np.array(color)[None, None, :] * (1 + fine[:, :, None]), 0, 1),
            "roughness": np.repeat(np.clip(roughness + fine, 0, 1)[:, :, None], 3, axis=2),
        }
        for channel, rgb in channels.items():
            pixels = np.ones((size, size, 4), dtype=np.float32)
            pixels[:, :, :3] = rgb
            texture = bpy.data.images.new(f"RTS_{name}_{channel}", size, size, alpha=False)
            texture.colorspace_settings.name = "Non-Color"
            texture.pixels.foreach_set(pixels.ravel())
            texture.filepath_raw = str(texture_dir / f"{name}_{channel}.png")
            texture.file_format = "PNG"
            texture.save()
            node = material.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = texture
            target = "Base Color" if channel == "base_color" else "Roughness"
            material.node_tree.links.new(node.outputs["Color"], shader.inputs[target])
        materials[name] = material
    return materials


@dataclass
class Asset:
    """Own one rigid part's visuals, collision primitives and affordance metadata."""

    name: str
    """Stable asset identifier used by the task's manifest."""
    mass: float
    """Suggested physical mass in kilograms."""
    collection: bpy.types.Collection
    """Objects belonging to this part, expressed in its local frame."""
    colliders: list[dict] = field(default_factory=list)
    """Conservative primitive collision proxies preserving access cavities."""
    affordances: dict = field(default_factory=dict)
    """Assembly, grasp, interaction and placement coordinates."""


class Geometry:
    """Build consistent rounded manufacturing geometry in an isolated scene."""

    def __init__(self, scene: bpy.types.Scene, materials: dict[str, bpy.types.Material]):
        self.scene = scene
        self.materials = materials
        self.assets: dict[str, Asset] = {}
        self.current: Asset | None = None

    def asset(self, name: str, mass: float, **affordances) -> Asset:
        """Start a rigid part with its origin, metadata and dedicated collection."""
        collection = bpy.data.collections.new(f"RTS_{name}")
        self.scene.collection.children.link(collection)
        self.current = Asset(name, mass, collection, affordances=affordances)
        self.assets[name] = self.current
        return self.current

    def finish(self, obj, name, material, bevel=0.0, smooth=False):
        """Assign a primitive to the active part and finish its visible edges."""
        assert self.current is not None
        obj.name = f"{self.current.name}_{name}"
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        self.current.collection.objects.link(obj)
        obj.data.materials.append(self.materials[material])
        if bevel:
            modifier = obj.modifiers.new("Manufactured_edge_radius", "BEVEL")
            modifier.width = bevel
            modifier.segments = 3
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        for polygon in obj.data.polygons:
            polygon.use_smooth = smooth
        if smooth or bevel:
            modifier = obj.modifiers.new("Face_weighted_normals", "WEIGHTED_NORMAL")
            modifier.keep_sharp = True
        self.ensure_uv(obj)
        return obj

    def ensure_uv(self, obj):
        """Preserve complete UV maps, or unwrap if any visible face lacks mapped area."""
        layer = obj.data.uv_layers.active
        needs_projection = layer is None
        area_tolerance = surface_area_tolerance(vertex.co for vertex in obj.data.vertices)
        if layer is not None:
            for polygon in obj.data.polygons:
                if polygon.area <= area_tolerance:
                    continue
                loops = polygon.loop_indices
                anchor = layer.data[loops[0]].uv
                has_area = False
                for index in range(1, len(loops) - 1):
                    a = layer.data[loops[index]].uv - anchor
                    b = layer.data[loops[index + 1]].uv - anchor
                    has_area |= abs(a.x * b.y - a.y * b.x) > UV_DOUBLED_AREA_TOLERANCE
                if not has_area:
                    needs_projection = True
                    break
        if not needs_projection:
            return
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        # Applying a bevel can clear face selection. Smart projection otherwise
        # silently creates a UV layer with every coordinate at the origin.
        # These generated materials contain only uniform microstructure, so a
        # whole-mesh unwrap safely repairs partial maps on text sides and bevels.
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")

    def box(self, name, location, dimensions, material="navy", bevel=0.002, collision=False, rotation=(0, 0, 0)):
        """Add a rounded cuboid and optional box collision proxy."""
        bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
        obj = bpy.context.object
        obj.dimensions = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        self.finish(obj, name, material, bevel)
        if collision:
            self.collider(name, "box", location, dimensions, rotation)
        return obj

    def cylinder(self, name, location, radius, depth, material="navy", rotation=(0, 0, 0), collision=False):
        """Add a turned part with a beveled rim and optional cylinder proxy."""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64, radius=radius, depth=depth, location=location, rotation=rotation
        )
        obj = self.finish(bpy.context.object, name, material, min(0.001, depth / 5), smooth=True)
        if collision:
            self.collider(name, "cylinder", location, (radius, depth), rotation)
        return obj

    def ring(self, name, location, outer, inner, depth, material="navy", rotation=(0, 0, 0), collision=False):
        """Add a manifold open tube; segmented proxies preserve the hollow bore."""
        vertices = []
        faces = []
        segments = 64
        for z, radius in ((-depth / 2, outer), (depth / 2, outer), (-depth / 2, inner), (depth / 2, inner)):
            for i in range(segments):
                angle = math.tau * i / segments
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend((
                (i, j, segments + j, segments + i),
                (2 * segments + j, 2 * segments + i, 3 * segments + i, 3 * segments + j),
                (j, i, 2 * segments + i, 2 * segments + j),
                (segments + i, segments + j, 3 * segments + j, 3 * segments + i),
            ))
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        self.scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = rotation
        self.finish(obj, name, material, min(0.0005, (outer - inner) / 4), smooth=True)
        if collision:
            matrix = Euler(rotation).to_matrix()
            radius = (outer + inner) / 2
            for i in range(16):
                angle = math.tau * i / 16
                center = Vector(location) + matrix @ Vector((radius * math.cos(angle), radius * math.sin(angle), 0))
                orientation = (matrix @ Euler((0, 0, angle)).to_matrix()).to_euler()
                self.collider(
                    name, "box", center, (outer - inner, 2 * inner * math.tan(math.pi / 16), depth), orientation
                )
        return obj

    def label(self, text, location, size=0.010, material="ivory", rotation=(0, 0, 0)):
        """Create sharp, self-contained engraved-style mesh lettering."""
        curve = bpy.data.curves.new("Label", "FONT")
        curve.body = text
        curve.size = size
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.extrude = 0.00008
        curve.resolution_u = 4
        obj = bpy.data.objects.new("Label", curve)
        self.scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = rotation
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")
        return self.finish(bpy.context.object, "label", material)

    def swept_tube(self, name, centers, outer, inner, material="teal"):
        """Sweep a manifold hollow tube along a planar XY centerline."""
        vertices, faces = [], []
        segments = 32
        for i, center in enumerate(centers):
            before = Vector(centers[max(0, i - 1)])
            after = Vector(centers[min(len(centers) - 1, i + 1)])
            tangent = (after - before).normalized()
            normal = Vector((-tangent.y, tangent.x, 0))
            for radius in (outer, inner):
                for j in range(segments):
                    angle = math.tau * j / segments
                    offset = radius * (math.cos(angle) * normal + Vector((0, 0, math.sin(angle))))
                    vertices.append(Vector(center) + offset)
        stride = 2 * segments
        for i in range(len(centers) - 1):
            for j in range(segments):
                k = (j + 1) % segments
                a, b = i * stride, (i + 1) * stride
                faces.append((a + j, a + k, b + k, b + j))
                faces.append((a + segments + k, a + segments + j, b + segments + j, b + segments + k))
        for end, reverse in ((0, False), ((len(centers) - 1) * stride, True)):
            for j in range(segments):
                k = (j + 1) % segments
                face = (end + j, end + segments + j, end + segments + k, end + k)
                faces.append(tuple(reversed(face)) if reverse else face)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        self.scene.collection.objects.link(obj)
        self.finish(obj, name, material, smooth=True)
        for start, end in zip(centers[:-1], centers[1:]):
            direction = Vector(end) - Vector(start)
            rotation = direction.to_track_quat("Z", "Y").to_euler()
            self.collider(name, "cylinder", (Vector(start) + Vector(end)) / 2, (outer, direction.length), rotation)
        return obj

    def collider(self, feature, shape, location, dimensions, rotation=(0, 0, 0)):
        """Record a primitive collision shape in the part's local frame."""
        assert self.current is not None
        quaternion = Euler(rotation).to_quaternion()
        self.current.colliders.append({
            "feature": feature,
            "shape": shape,
            "position": list(location),
            "dimensions": list(dimensions),
            "rotation_xyzw": [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
        })


def export_asset(asset: Asset, output_dir: Path) -> dict:
    """Export a visual USD with material textures and conservative collision shapes.

    Args:
        asset: Rigid part whose geometry is expressed in its local frame.
        output_dir: Directory receiving the USD.

    Returns:
        Manifest entry with bounds, mass, collision count and affordances.
    """
    bpy.ops.object.select_all(action="DESELECT")
    vertices = []
    for obj in asset.collection.objects:
        obj.select_set(True)
        vertices.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    path = output_dir / f"{asset.name}.usdc"
    bpy.ops.wm.usd_export(
        filepath=str(path),
        selected_objects_only=True,
        visible_objects_only=False,
        root_prim_path="/Asset",
        export_materials=True,
        export_textures=True,
        overwrite_textures=True,
        relative_paths=True,
        export_lights=False,
        export_cameras=False,
        generate_preview_surface=True,
        export_subdivision="BEST_MATCH",
        meters_per_unit=1.0,
    )
    stage = Usd.Stage.Open(str(path))
    root = stage.GetPrimAtPath("/Asset")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for category in ("status", "reading"):
        paths = {}
        for key, object_name in asset.affordances.get(f"{category}_object_names", {}).items():
            for prim in stage.Traverse():
                if prim.GetName() == Tf.MakeValidIdentifier(object_name):
                    paths[key] = str(prim.GetPath())
                    visible = category == "status" and key == "idle"
                    imageable = UsdGeom.Imageable(prim)
                    imageable.CreateVisibilityAttr(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
                    break
            assert key in paths, f"Missing exported {category} mesh: {object_name}"
        if paths:
            asset.affordances[f"{category}_paths"] = paths
    for i, collider in enumerate(asset.colliders):
        collider_path = f"/Asset/Collisions/shape_{i:03d}"
        dimensions = collider["dimensions"]
        if collider["shape"] == "box":
            geom = UsdGeom.Cube.Define(stage, collider_path)
            geom.CreateSizeAttr(1.0)
            geom.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        else:
            geom = UsdGeom.Cylinder.Define(stage, collider_path)
            geom.CreateRadiusAttr(dimensions[0])
            geom.CreateHeightAttr(dimensions[1])
            geom.CreateAxisAttr(UsdGeom.Tokens.z)
        transform = UsdGeom.Xformable(geom.GetPrim())
        translate = transform.AddTranslateOp()
        translate.Set(Gf.Vec3d(*collider["position"]))
        quaternion = collider["rotation_xyzw"]
        orient = transform.AddOrientOp()
        orient.Set(Gf.Quatf(quaternion[3], Gf.Vec3f(*quaternion[:3])))
        operations = transform.GetOrderedXformOps()
        transform.SetXformOpOrder([translate, orient] + [op for op in operations if op != translate and op != orient])
        geom.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        geom.GetPrim().SetCustomDataByKey("arena:feature", collider["feature"])
    stage.GetRootLayer().Save()
    bounds = np.array(vertices)
    return {
        "file": path.name,
        "root_prim": "/Asset",
        "mass_kg": asset.mass,
        "bounds_min": bounds.min(axis=0).tolist(),
        "bounds_max": bounds.max(axis=0).tolist(),
        "collision_count": len(asset.colliders),
        "affordances": asset.affordances,
    }


def write_manifest(assets: dict[str, dict], output_dir: Path):
    """Write the authoritative local coordinate and export interface."""
    manifest = {
        "schema_version": 1,
        "units": "meters",
        "up_axis": "Z",
        "license": "Apache-2.0",
        "generated_with": f"Blender {bpy.app.version_string}",
        "assets": assets,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
