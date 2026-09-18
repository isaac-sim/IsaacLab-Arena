# Copyright (c) 2026, Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a high-fidelity solid jug collision USD through Blender's exact Boolean backend.

Example:

    .venv/bin/python examples/sdf/make_jug_collision_blender.py \\
        scene/fstylejug_a01/fstylejug_a01_inst_physx.usd \\
        scene/tmp/fstylejug_a01_collision.usda
"""

import argparse
import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh


def _load_sdf_tool():
    """Load the shared USD reader and USD writer without making examples a package."""
    tool_path = Path(__file__).with_name("build_usd_combined_sdf.py")
    spec = importlib.util.spec_from_file_location("build_usd_combined_sdf", tool_path)
    assert spec is not None and spec.loader is not None, f"无法加载工具：{tool_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Parse collision-proxy generation arguments."""
    parser = argparse.ArgumentParser(
        description="用 Blender EXACT Boolean 生成水壶实心水密 collision USD。"
    )
    parser.add_argument("input_usd", type=Path)
    parser.add_argument("output_usd", type=Path)
    args = parser.parse_args()
    assert args.input_usd.is_file(), f"输入 USD 不存在：{args.input_usd}"
    return args


def main() -> None:
    """Convert USD meshes to OBJ, run Blender, and convert the result back to collision USD."""
    args = parse_args()
    tool = _load_sdf_tool()
    vertices, indices, mesh_paths = tool.load_combined_mesh(args.input_usd)
    source_mesh = trimesh.Trimesh(
        vertices=vertices, faces=indices.reshape(-1, 3), process=False
    )
    components = source_mesh.split(only_watertight=False)
    assert len(components) >= 2, "需要至少两个部件（瓶身和瓶盖）才能执行精确并集。"
    assert all(
        component.is_watertight for component in components
    ), "输入部件必须均为水密网格。"
    blender_script = Path(__file__).with_name("blender_make_jug_collision.py")
    args.output_usd.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jug_collision_") as temp_dir:
        temp_path = Path(temp_dir)
        input_objs = []
        for index, component in enumerate(components):
            path = temp_path / f"component_{index}.obj"
            component.export(path)
            input_objs.append(path)
        collision_obj = temp_path / "collision.obj"
        subprocess.run(
            [
                "blender",
                "-b",
                "--python",
                str(blender_script),
                "--",
                *(str(path) for path in input_objs),
                "--output-obj",
                str(collision_obj),
            ],
            check=True,
        )
        collision_mesh = trimesh.load(collision_obj, force="mesh", process=True)
        # OBJ represents corners independently when normals or UVs differ.  Weld
        # those duplicated positions before checking the topology in trimesh.
        collision_mesh.merge_vertices()
        collision_mesh.remove_unreferenced_vertices()
        collision_mesh.fix_normals(multibody=True)

    collision_vertices = np.asarray(collision_mesh.vertices, dtype=np.float32)
    collision_indices = np.asarray(collision_mesh.faces, dtype=np.int32).reshape(-1)
    assert collision_mesh.is_watertight, "Blender 输出的 collision mesh 不水密。"
    tool.write_combined_usd(args.output_usd, collision_vertices, collision_indices)
    print(
        f"[INFO] 已从 {len(mesh_paths)} 个 visual Mesh 生成 collision USD：{args.output_usd}，"
        f"顶点={len(collision_vertices)}，三角形={len(collision_indices) // 3}。"
    )


if __name__ == "__main__":
    main()
