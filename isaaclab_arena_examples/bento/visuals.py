# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic food meshes without downloading external asset packs."""

from __future__ import annotations

import numpy as np

from isaaclab_arena_examples.bento.meal import TCP_OFFSET


class MeshParts:
    """Collect colored triangle meshes into a single USD visual mesh."""

    def __init__(self, seed=17):
        import trimesh

        self.trimesh = trimesh
        self.rng = np.random.default_rng(seed)
        self.parts, self.colors = [], []

    def add(self, mesh, color, variation=0.0):
        """Append a mesh with restrained per-vertex color variation."""
        color = np.broadcast_to(color, (len(mesh.vertices), 3)).copy()
        if variation:
            color *= self.rng.uniform(1 - variation, 1 + variation, (len(mesh.vertices), 1))
        self.parts.append(mesh)
        self.colors.append(np.clip(color, 0, 1))

    def ellipsoid(self, center, radii, color, detail=1, roughness=0.0, rotation=None):
        """Append an ellipsoid, optionally with a craggy surface."""
        mesh = self.trimesh.creation.icosphere(subdivisions=detail)
        if roughness:
            mesh.vertices *= self.rng.uniform(1 - roughness, 1, (len(mesh.vertices), 1))
        mesh.vertices *= np.asarray(radii)
        if rotation is not None:
            mesh.vertices = mesh.vertices @ rotation.T
        mesh.vertices += np.asarray(center)
        self.add(mesh, color, variation=0.08)

    def rounded_box(self, center, size, color, exponent=0.25):
        """Append a soft rectangular food portion using a superellipsoid."""
        mesh = self.trimesh.creation.uv_sphere(count=[16, 24])
        vertices = np.asarray(mesh.vertices)
        mesh.vertices = np.sign(vertices) * np.abs(vertices) ** exponent * np.asarray(size) / 2 + center
        self.add(mesh, color, variation=0.035)

    def write(self, stage, path, fit_size=None):
        """Author one triangle mesh with vertex colors and no collision geometry."""
        from pxr import Sdf, UsdGeom, UsdShade, Vt

        mesh = self.trimesh.util.concatenate(self.parts)
        if fit_size is not None:
            bounds = mesh.bounds
            mesh.vertices = (mesh.vertices - bounds.mean(axis=0)) * np.asarray(fit_size) / (bounds[1] - bounds[0])
        colors = np.concatenate(self.colors).astype(np.float32)
        usd = UsdGeom.Mesh.Define(stage, path)
        usd.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
        usd.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
        usd.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).ravel()))
        usd.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd.CreateDoubleSidedAttr(True)
        usd.CreatePurposeAttr(UsdGeom.Tokens.render)
        usd.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(colors))
        # PreviewSurface reads the same vertex colors in Kit and in the CUDA capture.
        material = UsdShade.Material.Define(stage, path + "Material")
        reader = UsdShade.Shader.Define(stage, path + "Material/VertexColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")

        shader = UsdShade.Shader.Define(stage, path + "Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(usd.GetPrim()).Bind(material)
        return usd


def add_food_visual(stage, path, food):
    """Add rice grains, rolled egg, fried coating, or branched broccoli to a body."""
    parts = MeshParts(seed=31 + sum(map(ord, food.name)))
    sx, sy, sz = food.size
    if food.name == "rice":
        parts.rounded_box((0, 0, -0.004), (sx * 0.97, sy * 0.96, sz * 0.77), (0.86, 0.82, 0.67), 0.35)
        # Visible cooked grains cover the upper surface and exposed side walls.
        for _ in range(1450):
            x, y = parts.rng.uniform(-0.47, 0.47, 2) * (sx, sy)
            z = sz * (0.32 + 0.09 * (1 - (x / (sx / 2)) ** 2) * (1 - (y / (sy / 2)) ** 2))
            yaw = parts.rng.uniform(0, 2 * np.pi)
            rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            parts.ellipsoid((x, y, z), (0.0028, 0.0012, 0.0010), (0.96, 0.93, 0.84), rotation=rotation)
        for side in (-1, 1):
            for _ in range(180):
                x = parts.rng.uniform(-0.44, 0.44) * sx
                z = parts.rng.uniform(-0.32, 0.26) * sz
                parts.ellipsoid((x, side * sy * 0.468, z), (0.0029, 0.0010, 0.0011), (0.94, 0.91, 0.81))
        # An umeboshi makes the white rice portion recognizable at a glance.
        parts.ellipsoid(
            (0.015, 0.005, sz * 0.34), (0.009, 0.009, 0.0058), (0.46, 0.018, 0.014), detail=2, roughness=0.18
        )
        for _ in range(26):
            x, y = parts.rng.uniform(-0.4, 0.4, 2) * (sx, sy)
            parts.ellipsoid((x, y, sz * 0.425), (0.0010, 0.0005, 0.00035), (0.11, 0.08, 0.035), detail=0)
    elif food.name == "tamago":
        # Three separate slices; spiral strips are the rolled layers on each cut face.
        for x in (-sx / 3, 0, sx / 3):
            parts.rounded_box((x, 0, 0), (sx / 3 * 0.94, sy * 0.96, sz * 0.96), (0.94, 0.58, 0.035), 0.28)
            parts.rounded_box((x, 0, sz * 0.39), (sx / 3 * 0.90, sy * 0.88, sz * 0.12), (0.63, 0.27, 0.028), 0.35)
            for face in (-1, 1):
                t = np.linspace(0.0, 5.6 * np.pi, 180)
                r = np.linspace(0.0006, sz * 0.44, len(t))
                # Thin golden spiral in the YZ plane at each X-facing cut end.
                vertices = []
                for delta in (-0.00038, 0.00038):
                    vertices.extend(
                        np.column_stack((
                            np.full(len(t), x + face * sx / 3 * 0.474),
                            (r + delta) * np.cos(t) * sy / sz,
                            (r + delta) * np.sin(t),
                        ))
                    )
                faces = [[i, i + 1, i + len(t)] for i in range(len(t) - 1)]
                faces += [[i + 1, i + len(t) + 1, i + len(t)] for i in range(len(t) - 1)]
                mesh = parts.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                parts.add(mesh, (0.52, 0.27, 0.016))
    elif food.name == "karaage":
        for center, radius in (
            ((-0.018, -0.013, -0.013), 0.021),
            ((0.019, -0.009, -0.012), 0.021),
            ((0.002, 0.021, -0.010), 0.020),
        ):
            parts.ellipsoid(
                center, (radius, radius * 0.9, radius * 0.87), (0.46, 0.17, 0.027), detail=3, roughness=0.20
            )
            for _ in range(120):
                direction = parts.rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                pos = np.asarray(center) + direction * radius * (0.93, 0.82, 0.78)
                crumb = parts.rng.uniform(0.0013, 0.0032)
                color = (0.59, 0.27, 0.046) if parts.rng.random() > 0.3 else (0.29, 0.085, 0.014)
                parts.ellipsoid(pos, (crumb, crumb, crumb * 0.8), color, detail=0)
    else:
        # Branching stems and hundreds of small flower buds, rather than a green ball.
        for center in ((-0.018, -0.011, -0.001), (0.017, -0.008, 0.000), (0.000, 0.016, 0.009)):
            parts.ellipsoid((center[0], center[1], -0.022), (0.006, 0.005, 0.016), (0.27, 0.48, 0.075), detail=2)
            parts.ellipsoid(center, (0.021, 0.019, 0.017), (0.035, 0.19, 0.009), detail=2, roughness=0.1)
            for _ in range(170):
                direction = parts.rng.normal(size=3)
                direction[2] = abs(direction[2])
                direction /= np.linalg.norm(direction)
                pos = np.asarray(center) + direction * (0.019, 0.017, 0.015)
                radius = parts.rng.uniform(0.0014, 0.0025)
                parts.ellipsoid(pos, (radius, radius, radius), (0.075, parts.rng.uniform(0.25, 0.39), 0.013), detail=0)
    parts.write(stage, path + "/FoodVisual", fit_size=np.asarray(food.size) * 0.99)


def rounded_rectangle(width, depth, radius, z):
    """Return a counterclockwise rounded rectangle boundary."""
    points = []
    for x, y, start in (
        (width / 2 - radius, depth / 2 - radius, 0),
        (-width / 2 + radius, depth / 2 - radius, 90),
        (-width / 2 + radius, -depth / 2 + radius, 180),
        (width / 2 - radius, -depth / 2 + radius, 270),
    ):
        angles = np.deg2rad(np.linspace(start, start + 90, 12, endpoint=False))
        points.extend(
            np.column_stack((x + radius * np.cos(angles), y + radius * np.sin(angles), np.full(len(angles), z)))
        )
    return np.asarray(points)


def add_bento_visual(stage, floor):
    """Create a rounded black takeout container with a lacquer-red interior and lip."""
    parts = MeshParts()
    center = np.asarray((0.57, 0.24, floor))
    loops = [
        rounded_rectangle(0.393, 0.313, 0.019, -0.010),
        rounded_rectangle(0.410, 0.330, 0.024, 0.053),
        rounded_rectangle(0.397, 0.317, 0.019, 0.053),
        rounded_rectangle(0.386, 0.306, 0.013, 0.002),
    ]
    n = len(loops[0])
    vertices = np.concatenate(loops) + center
    for level, color in enumerate(((0.022, 0.020, 0.018), (0.027, 0.025, 0.022), (0.23, 0.036, 0.020))):
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.extend((
                (level * n + i, level * n + j, (level + 1) * n + i),
                (level * n + j, (level + 1) * n + j, (level + 1) * n + i),
            ))
        parts.add(parts.trimesh.Trimesh(vertices=vertices, faces=faces, process=False), color)
    parts.rounded_box(center + (0, 0, -0.004), (0.390, 0.310, 0.012), (0.20, 0.030, 0.018), 0.12)
    # Lower, slender dividers show the meal above them and retain four physics compartments.
    parts.rounded_box(center + (0, 0, 0.016), (0.010, 0.305, 0.032), (0.22, 0.035, 0.02), 0.12)
    parts.rounded_box(center + (0, 0, 0.016), (0.390, 0.010, 0.032), (0.22, 0.035, 0.02), 0.12)
    parts.write(stage, "/World/BentoVisual")


def add_serving_trays(stage, table_top, foods):
    """Create separate stainless prep dishes beneath the food source positions."""
    parts = MeshParts()
    for food in foods:
        center = np.asarray((*food.source_xy, table_top - 0.006))
        parts.rounded_box(center, (food.size[0] + 0.025, food.size[1] + 0.023, 0.010), (0.37, 0.42, 0.45), 0.15)
    parts.write(stage, "/World/ServingDishes")


def add_tongs(stage, ee_path, opening):
    """Create a food gripper with two animated broad paddle jaws; return translation ops."""
    from pxr import Gf, UsdGeom

    housing = MeshParts()
    housing.rounded_box((0, 0, TCP_OFFSET[2] + 0.069), (0.080, 0.050, 0.036), (0.17, 0.20, 0.22), 0.25)
    housing.rounded_box((0, 0, TCP_OFFSET[2] + 0.050), (0.162, 0.033, 0.015), (0.32, 0.36, 0.39), 0.2)
    housing.ellipsoid((0, 0, -0.021), (0.021, 0.021, 0.022), (0.30, 0.35, 0.38), detail=2)
    stem = housing.trimesh.creation.cylinder(radius=0.013, height=0.10, sections=24)
    stem.apply_translation((0, 0, -0.073))
    housing.add(stem, (0.37, 0.42, 0.45))
    housing.write(stage, ee_path + "/GripperHousing")
    ops = []
    for name, side in (("ToolJawLeft", -1), ("ToolJawRight", 1)):
        root = UsdGeom.Xform.Define(stage, ee_path + "/" + name)
        op = root.AddTranslateOp()
        op.Set(Gf.Vec3d(side * opening / 2, 0, TCP_OFFSET[2]))
        parts = MeshParts()
        parts.rounded_box((0, 0, 0.023), (0.007, 0.150, 0.066), (0.40, 0.45, 0.48), 0.2)
        parts.rounded_box((-side * 0.0038, 0, 0), (0.002, 0.145, 0.041), (0.86, 0.88, 0.84), 0.16)
        parts.write(stage, str(root.GetPath()) + "/Paddle")
        ops.append(op)
    return ops
