# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Generate the articulated laptop, hollow ceramic cup, and wooden table from local geometry."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Callable
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

LAPTOP_DEPTH = 0.24
LAPTOP_WIDTH = 0.34
LAPTOP_HINGE_POSITION = (0.112, 0.0, 0.027)
CUP_HEIGHT = 0.095
CUP_RADIUS = 0.043
TABLE_HEIGHT = 0.74


def generate_laptop_cup_assets(directory: str | Path) -> dict[str, str]:
    """Return cached USD paths, rebuilding when this generator changes.

    Args:
        directory: Writable directory for USD files, textures, and their source digest.

    Returns:
        Absolute USD paths keyed by ``laptop``, ``cup``, and ``table``. Origins are
        at the laptop and cup bottoms, and at the tabletop surface. Local +X points
        toward the laptop hinge, +Y toward the seated user's left, and +Z upward.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    asset_paths = {name: str(directory / f"{name}.usda") for name in ("laptop", "cup", "table")}
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path = directory / "manifest.json"
    expected_files = [*asset_paths.values(), directory / "wood.png", directory / "screen.png"]
    if manifest_path.exists() and all(Path(path).is_file() for path in expected_files):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_digest") == source_digest:
            return asset_paths

    _write_png(directory / "wood.png", 768, 768, _wood_pixel)
    _write_png(directory / "screen.png", 512, 768, _screen_pixel)
    _create_laptop(asset_paths["laptop"])
    _create_cup(asset_paths["cup"])
    _create_table(asset_paths["table"])
    manifest_path.write_text(json.dumps({"source_digest": source_digest}, indent=2) + "\n")
    return asset_paths


def _create_stage(root_name: str) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, f"/{root_name}").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage, root


def _material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float = 0.45,
    metallic: float = 0.0,
    texture: str | None = None,
    emissive: bool = False,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    if texture is not None:
        reader = UsdShade.Shader.Define(stage, f"{path}/TextureCoordinates")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        image = UsdShade.Shader.Define(stage, f"{path}/Texture")
        image.CreateIdAttr("UsdUVTexture")
        image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.GetInput("diffuseColor").ConnectToSource(image.ConnectableAPI(), "rgb")
        if emissive:
            surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                image.ConnectableAPI(), "rgb"
            )
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.8)
    physics_material.CreateDynamicFrictionAttr(0.65)
    physics_material.CreateRestitutionAttr(0.0)
    return material


def _finish_geometry(
    geometry: UsdGeom.Gprim, material: UsdShade.Material | None, collision: bool, hidden: bool = False
) -> None:
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(geometry.GetPrim())
        collision_api.CreateContactOffsetAttr(0.001)
        collision_api.CreateRestOffsetAttr(0.0)
    if hidden:
        geometry.CreateVisibilityAttr(UsdGeom.Tokens.invisible)


def _box(
    stage: Usd.Stage,
    path: str,
    dimensions: tuple[float, float, float],
    center: tuple[float, float, float],
    material: UsdShade.Material | None,
    collision: bool = False,
    hidden: bool = False,
) -> UsdGeom.Cube:
    geometry = UsdGeom.Cube.Define(stage, path)
    geometry.CreateSizeAttr(1.0)
    geometry.AddTranslateOp().Set(Gf.Vec3d(*center))
    geometry.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    _finish_geometry(geometry, material, collision, hidden)
    return geometry


def _mesh(
    stage: Usd.Stage,
    path: str,
    points: list[tuple[float, float, float]],
    faces: list[list[int]],
    material: UsdShade.Material,
) -> UsdGeom.Mesh:
    geometry = UsdGeom.Mesh.Define(stage, path)
    geometry.CreatePointsAttr(points)
    geometry.CreateFaceVertexCountsAttr([len(face) for face in faces])
    face_vertex_indices = []
    for face in faces:
        face_vertex_indices.extend(face)
    geometry.CreateFaceVertexIndicesAttr(face_vertex_indices)
    geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    _finish_geometry(geometry, material, collision=False)
    return geometry


def _textured_quad(
    stage: Usd.Stage,
    path: str,
    corners: list[tuple[float, float, float]],
    material: UsdShade.Material,
) -> None:
    geometry = _mesh(stage, path, corners, [[0, 1, 2, 3]], material)
    texture_coordinates = UsdGeom.PrimvarsAPI(geometry).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    texture_coordinates.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def _rigid_body(prim: Usd.Prim, mass: float, kinematic: bool = False) -> None:
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    physics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physics.CreateSolverPositionIterationCountAttr(16)
    physics.CreateSolverVelocityIterationCountAttr(4)


def _create_laptop(path: str) -> None:
    stage, root = _create_stage("Laptop")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(root)
    articulation.CreateEnabledSelfCollisionsAttr(False)
    articulation.CreateSolverPositionIterationCountAttr(16)
    articulation.CreateSolverVelocityIterationCountAttr(4)
    aluminum = _material(stage, "/Laptop/Materials/Aluminum", (0.23, 0.24, 0.26), 0.28, 0.78)
    black = _material(stage, "/Laptop/Materials/Black", (0.009, 0.011, 0.014), 0.38)
    trackpad = _material(stage, "/Laptop/Materials/Trackpad", (0.19, 0.20, 0.22), 0.4, 0.55)
    screen = _material(
        stage, "/Laptop/Materials/Screen", (0.03, 0.06, 0.03), 0.2, texture="./screen.png", emissive=True
    )
    base = UsdGeom.Xform.Define(stage, "/Laptop/base")
    _rigid_body(base.GetPrim(), 1.35)
    _box(stage, "/Laptop/base/chassis", (LAPTOP_DEPTH, LAPTOP_WIDTH, 0.022), (0.0, 0.0, 0.011), aluminum, True)
    _box(stage, "/Laptop/base/keyboard_well", (0.098, 0.278, 0.001), (0.049, 0.0, 0.0221), black)
    for row_index in range(6):
        for column_index in range(14):
            _box(
                stage,
                f"/Laptop/base/keys/key_{row_index}_{column_index}",
                (0.0117, 0.0168, 0.0012),
                (0.010 + row_index * 0.0148, (column_index - 6.5) * 0.0192, 0.0229),
                black,
            )
    _box(stage, "/Laptop/base/spacebar", (0.0117, 0.101, 0.0012), (0.010, 0.0, 0.0231), black)
    _box(stage, "/Laptop/base/trackpad_border", (0.077, 0.134, 0.0006), (-0.061, 0.0, 0.02215), black)
    _box(stage, "/Laptop/base/trackpad", (0.0758, 0.1328, 0.0007), (-0.061, 0.0, 0.0225), trackpad)
    for side in (-1, 1):
        for slot_index in range(16):
            _box(
                stage,
                f"/Laptop/base/speaker_{side + 1}/slot_{slot_index}",
                (0.001, 0.012, 0.0003),
                (0.010 + slot_index * 0.0055, side * 0.151, 0.0225),
                black,
            )
    lid = UsdGeom.Xform.Define(stage, "/Laptop/lid")
    lid.AddTranslateOp().Set(Gf.Vec3d(*LAPTOP_HINGE_POSITION))
    _rigid_body(lid.GetPrim(), 0.28)
    _box(stage, "/Laptop/lid/shell", (0.232, 0.34, 0.006), (-0.116, 0.0, 0.0), aluminum, True)
    _box(stage, "/Laptop/lid/bezel", (0.224, 0.33, 0.0004), (-0.116, 0.0, -0.0032), black)
    _textured_quad(
        stage,
        "/Laptop/lid/display",
        [(-0.224, -0.158, -0.00345), (-0.224, 0.158, -0.00345), (-0.013, 0.158, -0.00345), (-0.013, -0.158, -0.00345)],
        screen,
    )
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/Laptop/fixed_base")
    fixed_joint.CreateBody1Rel().SetTargets(["/Laptop/base"])
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Laptop/lid_hinge")
    hinge.CreateBody0Rel().SetTargets(["/Laptop/base"])
    hinge.CreateBody1Rel().SetTargets(["/Laptop/lid"])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*LAPTOP_HINGE_POSITION))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    hinge.CreateAxisAttr("Y")
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(120.0)
    PhysxSchema.PhysxJointAPI.Apply(hinge.GetPrim()).CreateJointFrictionAttr(0.35)
    damping = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    damping.CreateTypeAttr("force")
    damping.CreateStiffnessAttr(0.0)
    damping.CreateDampingAttr(0.08)
    stage.GetRootLayer().Export(path)


def _create_cup(path: str) -> None:
    stage, root = _create_stage("Cup")
    _rigid_body(root, 0.30)
    ceramic = _material(stage, "/Cup/Materials/Ceramic", (0.78, 0.77, 0.70), 0.22)
    rim = _material(stage, "/Cup/Materials/Rim", (0.16, 0.13, 0.08), 0.2, 0.35)
    segment_count = 64
    # The profile follows the outside, rim, and inside back to the solid ceramic floor.
    profile = [(0.0, 0.0), (0.035, 0.0), (0.040, 0.008), (0.043, 0.095), (0.037, 0.095), (0.034, 0.009), (0.0, 0.009)]
    points = []
    for radius, height in profile:
        for segment_index in range(segment_count):
            angle = 2.0 * math.pi * segment_index / segment_count
            points.append((radius * math.cos(angle), radius * math.sin(angle), height))
    faces = []
    for profile_index in range(len(profile) - 1):
        for segment_index in range(segment_count):
            next_index = (segment_index + 1) % segment_count
            start = profile_index * segment_count
            faces.append([
                start + segment_index,
                start + next_index,
                start + segment_count + next_index,
                start + segment_count + segment_index,
            ])
    _mesh(stage, "/Cup/ceramic_body", points, faces, ceramic)
    # Separate convex wall segments keep the cavity open to fingers and other objects.
    collision_segments = 32
    for segment_index in range(collision_segments):
        angle = 2.0 * math.pi * segment_index / collision_segments
        wall = _box(
            stage,
            f"/Cup/collisions/wall_{segment_index}",
            (0.006, 0.0082, 0.087),
            (0.039 * math.cos(angle), 0.039 * math.sin(angle), 0.0515),
            ceramic,
            True,
            True,
        )
        wall.AddRotateZOp().Set(math.degrees(angle))
        # Rotation must precede nonuniform scale in the transform stack.
        wall.SetXformOpOrder([wall.GetOrderedXformOps()[0], wall.GetOrderedXformOps()[2], wall.GetOrderedXformOps()[1]])
    bottom = UsdGeom.Cylinder.Define(stage, "/Cup/collisions/bottom")
    bottom.CreateRadiusAttr(0.038)
    bottom.CreateHeightAttr(0.009)
    bottom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0045))
    _finish_geometry(bottom, ceramic, True, True)
    rim_centers = []
    for segment_index in range(segment_count + 1):
        angle = 2.0 * math.pi * segment_index / segment_count
        rim_centers.append(Gf.Vec3d(0.040 * math.cos(angle), 0.040 * math.sin(angle), 0.094))
    _tube(stage, "/Cup/dark_rim", rim_centers, 0.0019, rim)
    handle_centers = []
    for segment_index in range(25):
        angle = -math.pi / 2.0 + math.pi * segment_index / 24
        handle_centers.append(Gf.Vec3d(0.0, -0.039 - 0.030 * math.cos(angle), 0.047 + 0.027 * math.sin(angle)))
    _tube(stage, "/Cup/handle", handle_centers, 0.0065, ceramic)
    for segment_index in range(0, 24, 2):
        start, end = handle_centers[segment_index], handle_centers[segment_index + 2]
        direction = end - start
        capsule = UsdGeom.Capsule.Define(stage, f"/Cup/collisions/handle_{segment_index}")
        capsule.CreateAxisAttr("Z")
        capsule.CreateRadiusAttr(0.0065)
        capsule.CreateHeightAttr(direction.GetLength())
        capsule.AddTranslateOp().Set((start + end) / 2.0)
        capsule.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), direction).GetQuat())
        _finish_geometry(capsule, ceramic, True, True)
    stage.GetRootLayer().Export(path)


def _tube(stage: Usd.Stage, path: str, centers: list[Gf.Vec3d], radius: float, material: UsdShade.Material) -> None:
    ring_segments = 10
    points = []
    for center_index, center in enumerate(centers):
        tangent = centers[min(center_index + 1, len(centers) - 1)] - centers[max(center_index - 1, 0)]
        tangent.Normalize()
        reference = Gf.Vec3d(0, 0, 1) if abs(tangent[2]) < 0.9 else Gf.Vec3d(1, 0, 0)
        normal = Gf.Cross(tangent, reference).GetNormalized()
        binormal = Gf.Cross(tangent, normal).GetNormalized()
        for ring_index in range(ring_segments):
            angle = 2.0 * math.pi * ring_index / ring_segments
            position = center + radius * (math.cos(angle) * normal + math.sin(angle) * binormal)
            points.append(tuple(position))
    faces = []
    for center_index in range(len(centers) - 1):
        for ring_index in range(ring_segments):
            next_index = (ring_index + 1) % ring_segments
            start = center_index * ring_segments
            faces.append([
                start + ring_index,
                start + next_index,
                start + ring_segments + next_index,
                start + ring_segments + ring_index,
            ])
    faces.append(list(reversed(range(ring_segments))))
    faces.append(list(range((len(centers) - 1) * ring_segments, len(centers) * ring_segments)))
    _mesh(stage, path, points, faces, material)


def _create_table(path: str) -> None:
    stage, root = _create_stage("Table")
    _rigid_body(root, 45.0, kinematic=True)
    wood = _material(stage, "/Table/Materials/Wood", (0.32, 0.15, 0.06), 0.39)
    grain = _material(stage, "/Table/Materials/Grain", (0.32, 0.15, 0.06), 0.39, texture="./wood.png")
    _box(stage, "/Table/top", (0.76, 1.25, 0.035), (0.0, 0.0, -0.0175), wood, True)
    _textured_quad(
        stage,
        "/Table/top_grain",
        [(-0.38, -0.625, 0.00005), (0.38, -0.625, 0.00005), (0.38, 0.625, 0.00005), (-0.38, 0.625, 0.00005)],
        grain,
    )
    for x_index, x in enumerate((-0.32, 0.32)):
        for y_index, y in enumerate((-0.555, 0.555)):
            _box(
                stage,
                f"/Table/leg_{x_index}_{y_index}",
                (0.048, 0.048, TABLE_HEIGHT - 0.035),
                (x, y, -(TABLE_HEIGHT + 0.035) / 2.0),
                wood,
                True,
            )
    for side in (-1, 1):
        _box(stage, f"/Table/apron_long_{side + 1}", (0.025, 1.11, 0.09), (side * 0.32, 0.0, -0.075), wood, True)
        _box(stage, f"/Table/apron_short_{side + 1}", (0.64, 0.025, 0.09), (0.0, side * 0.555, -0.075), wood, True)
    stage.GetRootLayer().Export(path)


def _wood_pixel(column: int, row: int) -> tuple[int, int, int]:
    across = column / 768.0
    along = row / 768.0
    plank = min(int(across * 5), 4)
    warped = across * 90 + 1.7 * math.sin(along * 7 + plank) + 0.8 * math.sin(along * 19 + across * 4)
    grain = math.sin(warped * math.pi) + 0.35 * math.sin(warped * math.pi * 3.1)
    fine_grain = math.sin(across * 2400 + math.sin(along * 33) * 3)
    shade = 8.0 * grain + 2.5 * fine_grain + 5.0 * math.sin(plank * 2.7)
    if (across * 5) % 1 < 0.004:
        shade -= 22
    return int(158 + shade), int(101 + shade * 0.8), int(54 + shade * 0.55)


def _screen_pixel(column: int, row: int) -> tuple[int, int, int]:
    horizontal = column / 512.0 - 0.5
    vertical = row / 768.0 - 0.5
    radius = math.sqrt(horizontal * horizontal + vertical * vertical)
    wave = math.sin(radius * 36 - math.atan2(vertical, horizontal) * 2)
    dot = column % 7 < 3 and row % 7 < 3
    glow = max(0.0, wave) * (0.45 + 0.55 * (1.0 - min(radius, 1.0))) if dot else 0.0
    return int(8 + 82 * glow), int(16 + 200 * glow), int(17 + 42 * glow)


def _write_png(path: Path, width: int, height: int, pixel: Callable[[int, int], tuple[int, int, int]]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        for column in range(width):
            scanlines.extend(pixel(column, row))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
    )
