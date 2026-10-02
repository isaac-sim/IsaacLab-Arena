# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the original Blender-authored service-cell assets and preview."""

from __future__ import annotations

import argparse
import bpy
import json
import math
import re
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector
from pxr import Gf, Usd, UsdGeom, UsdLux

# Load only Blender tooling: the Arena environment registry requires Isaac Sim.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asset_source.components import build_all
from asset_source.geometry import Geometry, export_asset, make_materials, write_manifest
from asset_source.validate_exports import validate_exports


def _license_bytes() -> bytes:
    """Read Arena's license from adjacent wheel metadata or its identified source checkout."""
    root = Path(__file__).resolve().parents[3]
    for distribution in distributions(path=[str(root)]):
        name = re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower()
        if name != "isaaclab-arena":
            continue
        for entry in distribution.files or ():
            if entry.name == "LICENSE.md" and any(part.endswith(".dist-info") for part in entry.parts):
                source = Path(distribution.locate_file(entry))
                if source.is_file():
                    return source.read_bytes()
    project_file = root / "pyproject.toml"
    source = root / "LICENSE.md"
    if project_file.is_file() and source.is_file():
        with project_file.open("rb") as stream:
            project = tomllib.load(stream)
        if project.get("project", {}).get("name") == "isaaclab_arena":
            return source.read_bytes()
    assert False, "Arena LICENSE.md is missing; use the source checkout or retain the installed wheel metadata."


def _pose_matrix(value: dict | list | tuple) -> Matrix:
    """Compose an XYZ position or an authored XYZ/XYZW pose for Blender.

    Args:
        value: Position sequence or pose dictionary matching runtime scene metadata.

    Returns:
        Transform mapping asset-local coordinates into its parent frame.
    """
    if isinstance(value, dict):
        assert "position_xyz" in value, "An asset pose must declare position_xyz"
        position = value["position_xyz"]
        x, y, z, w = value.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))
        rotation = Quaternion((w, x, y, z)).to_matrix().to_4x4()
        return Matrix.Translation(position) @ rotation
    assert isinstance(value, (list, tuple)) and len(value) == 3, "An asset pose must be XYZ or a pose dictionary"
    return Matrix.Translation(value)


def _preview_placements(geometry: Geometry, records: dict) -> dict[str, list[Matrix]]:
    """Compose the authoring preview's instances from local asset metadata.

    Args:
        geometry: Original asset collections and their assembly affordances.
        records: Exported asset records supplying measured bounds.

    Returns:
        World placement matrices for each visible instance of a source asset.
    """
    table_height = 0.78
    placements: dict[str, list[Matrix]] = {}

    def place(name: str, transform: Matrix) -> Matrix:
        placements.setdefault(name, []).append(transform)
        return transform

    def on_bench(name: str, x: float, y: float) -> Matrix:
        z = table_height - records[name]["bounds_min"][2] + 0.002
        return place(name, Matrix.Translation((x, y, z)))

    def attached(name: str, parent: Matrix, source: str, socket: str) -> Matrix:
        offset = geometry.assets[source].affordances[socket]
        return place(name, parent @ _pose_matrix(offset))

    # This authoring view mirrors the evaluation's compact fixture arrangement;
    # scene.py remains authoritative for robot placement and runtime physics.
    place("bench", Matrix.Translation((0.48, 0, table_height)))
    place("floor", Matrix.Translation((0.48, 0, 0)))
    on_bench("work_order", 0.14, -0.25)
    cradle = on_bench("cradle", 0.46, 0.15)
    body = attached("vacuum_body", cradle, "cradle", "body_socket")
    cup = attached("dust_cup", body, "vacuum_body", "cup_socket")
    attached("filter", body, "vacuum_body", "filter_socket")
    attached("battery", body, "vacuum_body", "battery_socket")
    attached("obstruction", cup, "dust_cup", "obstruction_socket")
    latch = geometry.assets["vacuum_body"].affordances["cup_latch"]
    place("cup_latch", body @ Matrix.Translation(latch["position_xyz"]))
    for position in geometry.assets["dust_cup"].affordances["debris_poses"]:
        place("debris", cup @ Matrix.Translation(position))

    battery_tester = on_bench("battery_tester", 0.235, 0.315)
    on_bench("airflow_tester", 0.67, 0.155)
    on_bench("airflow_adapter", 0.585, -0.010)
    on_bench("battery_bin", 0.715, -0.08)
    on_bench("filter_bin", 0.715, -0.235)
    on_bench("waste_bin", 0.645, 0.345)
    rack = on_bench("spare_rack", 0.45, 0.35)
    stock = geometry.assets["spare_rack"].affordances["stock_poses"]
    for instance, source in (
        ("battery_spare", "battery"),
        ("battery_decoy", "battery_decoy"),
        ("filter_spare", "filter"),
        ("filter_decoy", "filter_decoy"),
    ):
        place(source, rack @ _pose_matrix(stock[instance]))
    parking = on_bench("parking_tray", 0.310, 0.005)
    for name, position in geometry.assets["parking_tray"].affordances["tool_poses"].items():
        place(name, parking @ _pose_matrix(position))

    case_z = table_height - records["case_base"]["bounds_min"][2] + 0.002
    case = Matrix.Translation((0.445, -0.25, case_z)) @ Matrix.Rotation(math.pi, 4, "Z")
    place("case_base", case)
    case_geometry = geometry.assets["case_base"].affordances
    pivot = Matrix.Translation(case_geometry["lid_closed_pose"]["position_xyz"])
    place("case_lid", case @ pivot @ Matrix.Rotation(case_geometry["hinge"]["open_angle"], 4, "X"))
    latch_pivot = Matrix.Translation(case_geometry["latch"]["position_xyz"])
    latch_open_angle = math.radians(case_geometry["latch"]["angle_limits_degrees"][1])
    place("case_latch", case @ latch_pivot @ Matrix.Rotation(latch_open_angle, 4, "X"))

    release_panel = place("release_panel", Matrix.Translation((0.44, 0.435, table_height + 0.001)))
    airflow_panel = place("airflow_test_panel", Matrix.Translation((0.570, 0.020, table_height + 0.001)))
    button_poses = [
        battery_tester @ Matrix.Translation(geometry.assets["battery_tester"].affordances["test_button"]),
        airflow_panel @ Matrix.Translation(geometry.assets["airflow_test_panel"].affordances["button_pose"]),
    ]
    for offset in geometry.assets["release_panel"].affordances["button_poses"].values():
        button_poses.append(release_panel @ Matrix.Translation(offset))
    for transform in button_poses:
        place("button_base", transform)
        place("test_button", transform @ Matrix.Translation((0, 0, 0.012)))

    return placements


def _place_preview_instances(
    geometry: Geometry, placements: dict[str, list[Matrix]]
) -> dict[str, list[bpy.types.Object]]:
    """Place source objects and linked duplicates without modifying shared meshes.

    Args:
        geometry: Asset collections in the isolated authoring scene.
        placements: World transforms indexed by source asset name.

    Returns:
        Visible source objects and linked copies, grouped by source asset name.
    """
    instances = bpy.data.collections.new("RTS_PreviewInstances")
    geometry.scene.collection.children.link(instances)
    visible_objects: dict[str, list[bpy.types.Object]] = {}
    for name, asset in geometry.assets.items():
        transforms = placements.get(name, [])
        hidden_names = set(asset.affordances.get("reading_object_names", {}).values())
        for status, object_name in asset.affordances.get("status_object_names", {}).items():
            if status != "idle":
                hidden_names.add(object_name)
        for obj in asset.collection.objects:
            if not transforms:
                obj.hide_render = True
                obj.hide_set(True)
                continue
            local_transform = obj.matrix_world.copy()
            for index, transform in enumerate(transforms):
                instance = obj if index == 0 else obj.copy()
                if index:
                    instances.objects.link(instance)
                instance.matrix_world = transform @ local_transform
                instance.hide_render = obj.name in hidden_names
                instance.hide_set(instance.hide_render)
                if not instance.hide_render:
                    visible_objects.setdefault(name, []).append(instance)
    return visible_objects


def arrange_preview(geometry: Geometry, output_dir: Path):
    """Arrange an illustrative workcell and preserve it as a separate blend file.

    Args:
        geometry: Original assets in a newly generated Blender scene.
        output_dir: Export directory containing the manifest and texture images.
    """
    records = json.loads((output_dir / "manifest.json").read_text())["assets"]
    visible_objects = _place_preview_instances(geometry, _preview_placements(geometry, records))
    scene = geometry.scene
    world = bpy.data.worlds.new("RTS_Studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.17, 0.20, 0.24, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    scene.world = world
    for name, position, energy, size in (
        ("Key", (-0.5, 1.3, 3), 500, 2.2),
        ("Fill", (-1.3, 0.5, 2.2), 330, 1.8),
        ("Rim", (1.4, -1.1, 2.4), 420, 1.5),
    ):
        light = bpy.data.lights.new(f"RTS_{name}", "AREA")
        light.energy, light.shape, light.size = energy, "DISK", size
        obj = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (Vector((0.45, 0, 0.8)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.cameras.new("RTS_Overview")
    obj = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(obj)
    obj.location = (1.55, 1.70, 2.35)
    obj.rotation_euler = (Vector((0.48, 0, 0.84)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera.type, camera.ortho_scale, camera.lens = "ORTHO", 1.48, 52
    scene.camera = obj
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 1920, 1440, 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.2
    scene.render.filepath = str(output_dir / "workcell_preview.png")
    # Frame every authored workcell asset with a six-percent image margin.
    # Bench legs and the floor may extend beyond this product-focused view.
    bpy.context.view_layer.update()
    extent = 0.0
    for name, objects in visible_objects.items():
        if name in {"bench", "floor"}:
            continue
        for source in objects:
            for corner in source.bound_box:
                projected = world_to_camera_view(scene, obj, source.matrix_world @ Vector(corner))
                extent = max(extent, 2 * abs(projected.x - 0.5), 2 * abs(projected.y - 0.5))
    camera.ortho_scale *= max(1.0, extent / 0.88)

    bpy.data.libraries.write(
        str(output_dir / "return_to_service.blend"), {scene}, path_remap="RELATIVE", fake_user=True, compress=True
    )


def export_lighting(output_dir: Path):
    """Create reusable scene lighting through Blender's USD Python integration."""
    stage = Usd.Stage.CreateNew(str(output_dir / "lighting.usda"))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    light = UsdLux.DomeLight.Define(stage, "/Asset/StudioDome")
    light.CreateIntensityAttr(650)
    light.CreateColorAttr(Gf.Vec3f(0.85, 0.92, 1.0))
    stage.GetRootLayer().Save()


def build(output_dir: str | Path, render: bool = False) -> Path:
    """Generate a new isolated Blender scene and export its complete asset library.

    Args:
        output_dir: External directory for generated models, textures and manifest.
        render: Whether to render the illustrative workcell preview.

    Returns:
        Path to the generated asset manifest.
    """
    license_bytes = _license_bytes()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "LICENSE.md").write_bytes(license_bytes)
    scene = bpy.data.scenes.new("Arena_Return_To_Service")
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    bpy.context.window.scene = scene
    geometry = Geometry(scene, make_materials(output_dir))
    build_all(geometry)
    bpy.context.view_layer.update()
    exports = {}
    for name, asset in geometry.assets.items():
        exports[name] = export_asset(asset, output_dir)
    export_lighting(output_dir)
    exports["lighting"] = {"file": "lighting.usda", "root_prim": "/Asset", "affordances": {}}
    write_manifest(exports, output_dir)
    validate_exports(output_dir)
    arrange_preview(geometry, output_dir)
    if render:
        bpy.ops.render.render(write_still=True)
    print(f"Generated {len(exports)} original assets in {output_dir}")
    return output_dir / "manifest.json"


def main():
    """Run with Blender --python build_assets.py -- --output-dir PATH."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    build(args.output_dir, args.render)


if __name__ == "__main__":
    main()
