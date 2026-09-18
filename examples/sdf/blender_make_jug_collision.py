# Copyright (c) 2026, Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Use Blender to make a single solid watertight jug collision mesh.

Run with Blender, not the normal Python interpreter:

    blender -b --python examples/sdf/blender_make_jug_collision.py -- \\
        /tmp/jug_body.obj /tmp/jug_cap.obj --output-obj /tmp/jug_collision.obj

The source meshes may contain a hollow jug body and a separate cap.  Blender's
voxel remesher converts their combined volume to one closed outer shell, so the
interior cavity is filled and small seams between the cap and body are sealed.
The voxel size controls the fidelity of the collision proxy.
"""

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    """Parse arguments following Blender's ``--`` separator."""
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="从 visual OBJ 生成精确的实心水密 collision OBJ。"
    )
    parser.add_argument("input_objs", type=Path, nargs="+")
    parser.add_argument("--output-obj", required=True, type=Path)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0e-3,
        help="实心化体素边长（m），默认 1 mm。",
    )
    args = parser.parse_args(arguments)
    assert all(path.is_file() for path in args.input_objs), "存在找不到的输入 OBJ。"
    assert args.voxel_size > 0.0, "--voxel-size 必须大于 0。"
    return args


def _clear_scene() -> None:
    """Remove every object from Blender's startup scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _apply_world_transform(obj: bpy.types.Object) -> None:
    """Bake one imported object's transform into its mesh vertices."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.select_set(False)


def _is_watertight(mesh: bpy.types.Mesh) -> bool:
    """Return whether every edge of a mesh has exactly two incident faces."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    result = all(edge.is_manifold for edge in bm.edges)
    bm.free()
    return result


def _solidify(objects: list[bpy.types.Object], voxel_size: float) -> bpy.types.Object:
    """Join objects and reconstruct their filled volume with Blender voxel remesh."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    result = objects[0]
    bpy.context.view_layer.objects.active = result
    bpy.ops.object.join()
    result.data.remesh_voxel_size = voxel_size
    result.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    result.name = "CollisionMesh"
    result.data.name = "CollisionMesh"
    return result


def main() -> None:
    """Import visual meshes, make one solid outer shell, and export it as OBJ."""
    args = parse_args()
    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    _clear_scene()
    for input_obj in args.input_objs:
        bpy.ops.wm.obj_import(filepath=str(input_obj))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    assert objects, "未导入任何 Mesh。"

    for obj in objects:
        _apply_world_transform(obj)
        assert _is_watertight(
            obj.data
        ), f"{obj.name} 不是水密 visual mesh，无法进行实心化。"

    collision = _solidify(objects, args.voxel_size)
    assert _is_watertight(collision.data), "实心化结果不是水密网格。"
    bpy.ops.object.select_all(action="DESELECT")
    collision.select_set(True)
    bpy.context.view_layer.objects.active = collision
    bpy.ops.wm.obj_export(
        filepath=str(args.output_obj),
        export_selected_objects=True,
        export_materials=False,
        export_normals=False,
        export_uv=False,
    )
    print(
        f"[INFO] 已导出实心水密 collision mesh：{args.output_obj}，"
        f"顶点={len(collision.data.vertices)}，面={len(collision.data.polygons)}。"
    )


if __name__ == "__main__":
    main()
