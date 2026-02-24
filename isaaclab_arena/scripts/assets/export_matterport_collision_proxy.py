# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export a scene asset into a mesh file for collision authoring workflows.

This script runs inside Isaac Sim so it can use Omniverse's asset converter.
The intended workflow is:

1. Export a raw Matterport / MP3D mesh (preferred) or a visual USD fallback to OBJ.
2. Simplify / clean the OBJ in an external mesh tool if needed.
3. Convert the cleaned OBJ into one or more collision-only USDA files.
4. Feed those USDA files to the Matterport collision overlay CLI flags.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Export a scene asset (raw mesh or USD) to OBJ or USD via Omniverse asset converter."
)
parser.add_argument(
    "--input_path",
    type=Path,
    required=True,
    help="Input asset path (raw Matterport mesh, USD, or another asset-converter supported format).",
)
parser.add_argument("--output_path", type=Path, required=True, help="Output asset path (e.g. .obj or .usd).")
parser.add_argument(
    "--merge_all_meshes",
    action="store_true",
    default=False,
    help="Request Omniverse asset converter to merge meshes when supported.",
)
parser.add_argument(
    "--ignore_materials",
    action="store_true",
    default=False,
    help="Drop materials during export when supported by the converter.",
)
parser.add_argument(
    "--export_hidden_props",
    action="store_true",
    default=False,
    help="Include hidden prims in the exported asset when supported.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import omni.kit.asset_converter as asset_converter  # noqa: E402
except ModuleNotFoundError:
    asset_converter = None

from pxr import Gf, Usd, UsdGeom  # noqa: E402


async def _convert_asset(input_path: Path, output_path: Path) -> None:
    """Convert an asset using Omniverse asset converter."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = asset_converter.get_instance()
    ctx = asset_converter.AssetConverterContext()
    ctx.ignore_materials = args_cli.ignore_materials
    ctx.export_hidden_props = args_cli.export_hidden_props
    ctx.merge_all_meshes = args_cli.merge_all_meshes

    def _progress_callback(progress: float, total_steps: float) -> bool:
        print(f"[asset_converter] progress={progress:.3f} total_steps={total_steps}")
        return True

    task = converter.create_converter_task(
        str(input_path),
        str(output_path),
        _progress_callback,
        ctx,
    )
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(
            f"Asset conversion failed: status={task.get_status()} error={task.get_error_message()}"
        )
    print(f"[asset_converter] exported: {output_path}")


def _iter_mesh_prims(stage: Usd.Stage):
    """Yield mesh prims from a USD stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            yield prim


def _transform_points(matrix: Gf.Matrix4d, points) -> list[tuple[float, float, float]]:
    """Transform USD mesh points into world-space XYZ tuples."""
    xyz_points: list[tuple[float, float, float]] = []
    for point in points:
        world = matrix.Transform(Gf.Vec3d(point[0], point[1], point[2]))
        xyz_points.append((world[0], world[1], world[2]))
    return xyz_points


def _write_obj_from_usd(input_path: Path, output_path: Path) -> None:
    """Fallback exporter: compose the USD stage and dump all meshes to OBJ."""
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xform_cache = UsdGeom.XformCache()
    vertex_offset = 1
    mesh_count = 0

    with output_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write(f"# Exported from USD: {input_path}\n")
        for prim in _iter_mesh_prims(stage):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not face_counts or not face_indices:
                continue

            local_to_world = xform_cache.GetLocalToWorldTransform(prim)
            world_points = _transform_points(local_to_world, points)
            if not world_points:
                continue

            mesh_count += 1
            obj_file.write(f"o mesh_{mesh_count}\n")
            for x, y, z in world_points:
                obj_file.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

            cursor = 0
            for face_count in face_counts:
                face = [face_indices[cursor + i] + vertex_offset for i in range(face_count)]
                cursor += face_count
                if len(face) < 3:
                    continue
                # Fan triangulation for non-triangle faces.
                for i in range(1, len(face) - 1):
                    obj_file.write(f"f {face[0]} {face[i]} {face[i + 1]}\n")

            vertex_offset += len(world_points)

    if mesh_count == 0:
        raise RuntimeError(f"No meshes found in USD stage: {input_path}")
    print(f"[pxr_obj_export] exported {mesh_count} meshes to: {output_path}")


def main() -> None:
    try:
        if asset_converter is not None:
            asyncio.get_event_loop().run_until_complete(_convert_asset(args_cli.input_path, args_cli.output_path))
        else:
            if args_cli.output_path.suffix.lower() != ".obj":
                raise RuntimeError(
                    "Asset converter is unavailable in this Isaac Sim image. "
                    "Fallback exporter currently supports only OBJ output."
                )
            _write_obj_from_usd(args_cli.input_path, args_cli.output_path)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
