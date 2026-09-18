# Copyright (c) 2026, Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""将 USD 内全部三角网格合并，并烘焙为 Newton 可复用的整体 SDF cache。

示例：

    .venv/bin/python examples/sdf/build_usd_combined_sdf.py \\
        scene/jug_with_cap.usd --cache-dir scene/tmp

输出为 ``{hash}.sdf.npz``。它是 Newton 的标准 sparse-texture SDF cache，内部包含
``coarse_sdf``、窄带子网格和元数据等多个 ``.npy`` 数组；请勿将其改名或拆散后再交给
Newton 读取。相同的合并网格和烘焙参数会命中同一缓存文件。

本工具仅合并 USD 中已摆放好的 Mesh prim；瓶盖必须已位于瓶口。对于开口瓶身与瓶盖，
建议使用 ``--sign-method winding``（默认值）。若两者存在明显缝隙，整体网格仍非封闭体，
应先在 DCC 中修补，或使用后续的体素 CSG 实心化流程。

使用 ``--solidify`` 可执行该实心化流程：表面体素化、闭运算封小缝、填充内部空腔、提取
体素外表面，并基于此代理网格重新烘焙 SDF。该模式适用于“封盖后的水壶应视为实心碰撞体”。
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "scene" / "tmp"


def parse_args() -> argparse.Namespace:
    """Parse USD-to-SDF baking command-line arguments."""
    parser = argparse.ArgumentParser(
        description="合并 USD Mesh 并生成 Newton 整体 SDF cache。"
    )
    parser.add_argument("usd", type=Path, help="输入 USD/USDZ/USDA 文件。")
    parser.add_argument(
        "--output-usd",
        type=Path,
        help="可选：输出合并后的单 Mesh USD（顶点坐标单位为 m）。",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成后启动 Newton SDF 可视化器；未指定 --output-usd 时自动写入 cache 目录。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="输出 Newton SDF cache 的目录。",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=256,
        help="最长轴最大体素数，必须能被 8 整除。",
    )
    parser.add_argument(
        "--narrow-band",
        type=float,
        default=0.04,
        help="SDF 内外窄带距离（m）。",
    )
    parser.add_argument(
        "--margin", type=float, default=0.002, help="SDF AABB 外扩距离（m）。"
    )
    parser.add_argument(
        "--sign-method",
        choices=("winding", "parity", "normal"),
        default="winding",
        help="内部/外部判定方法；含多个部件时推荐 winding。",
    )
    parser.add_argument(
        "--texture-format",
        choices=("uint8", "uint16", "float32"),
        default="uint16",
        help="窄带纹理缓存精度。",
    )
    parser.add_argument(
        "--solidify",
        action="store_true",
        help="把合并网格体素实心化，并用其封闭代理网格重建 SDF。",
    )
    parser.add_argument(
        "--solidify-voxel-size",
        type=float,
        default=0.002,
        help="实心化体素边长（m）；更小更精细但更慢，默认 2 mm。",
    )
    parser.add_argument(
        "--solidify-close-iterations",
        type=int,
        default=1,
        help="实心化前闭运算次数，用于封闭小缝隙。",
    )
    args = parser.parse_args()
    assert args.usd.is_file(), f"USD 文件不存在：{args.usd}"
    assert (
        args.max_resolution > 0 and args.max_resolution % 8 == 0
    ), "--max-resolution 必须为 8 的正整数倍。"
    assert args.narrow_band > 0.0, "--narrow-band 必须大于 0。"
    assert args.margin >= 0.0, "--margin 不能小于 0。"
    assert args.solidify_voxel_size > 0.0, "--solidify-voxel-size 必须大于 0。"
    assert (
        args.solidify_close_iterations >= 0
    ), "--solidify-close-iterations 不能小于 0。"
    return args


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate USD polygon faces into a flattened index array."""
    triangles: list[np.ndarray] = []
    offset = 0
    for count in face_counts:
        count = int(count)
        assert count >= 3, f"USD Mesh 包含少于 3 个顶点的面：{count}"
        face = face_indices[offset : offset + count]
        triangles.extend(
            (face[0], face[index], face[index + 1]) for index in range(1, count - 1)
        )
        offset += count
    assert offset == len(
        face_indices
    ), "USD faceVertexCounts 与 faceVertexIndices 长度不一致。"
    return np.asarray(triangles, dtype=np.int32).reshape(-1)


def load_combined_mesh(usd_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return all USD Mesh prims in stage coordinates converted to metres.

    The returned topology retains every visual Mesh prim, including a separate bottle cap.
    Each prim's local-to-world transform and USD handedness are baked into the result.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"无法打开 USD：{usd_path}"
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertex_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    mesh_paths: list[str] = []
    vertex_offset = 0

    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        face_counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        face_indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        if points is None or face_counts is None or face_indices is None:
            continue

        local_vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(local_vertices) == 0:
            continue
        local_indices = _triangulate(
            np.asarray(face_counts), np.asarray(face_indices, dtype=np.int32)
        )
        assert local_indices.max(initial=-1) < len(
            local_vertices
        ), f"{prim.GetPath()} 的索引超出顶点范围。"

        world_matrix = xform_cache.GetLocalToWorldTransform(prim)
        # GfMatrix4d follows USD's row-vector convention, hence ``points @ matrix``.
        homogeneous_vertices = np.concatenate(
            (local_vertices, np.ones((len(local_vertices), 1), dtype=np.float64)),
            axis=1,
        )
        world_vertices = (
            homogeneous_vertices @ np.asarray(world_matrix, dtype=np.float64)
        )[:, :3]
        world_vertices *= metres_per_unit

        linear = np.asarray(world_matrix, dtype=np.float64)[:3, :3]
        flip_winding = np.linalg.det(linear) < 0.0
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            flip_winding = not flip_winding
        if flip_winding:
            local_indices = local_indices.reshape(-1, 3)[:, (0, 2, 1)].reshape(-1)

        vertex_blocks.append(world_vertices.astype(np.float32))
        index_blocks.append(local_indices + vertex_offset)
        mesh_paths.append(str(prim.GetPath()))
        vertex_offset += len(world_vertices)

    assert vertex_blocks, f"USD 中未找到含三角面的 UsdGeom.Mesh：{usd_path}"
    return np.concatenate(vertex_blocks), np.concatenate(index_blocks), mesh_paths


def _cache_path_for_mesh(mesh, args: argparse.Namespace) -> Path:
    """Return the same deterministic cache path used by Newton's SDF cooker."""
    from newton._src.geometry import _sdf_cache

    cache_hash = _sdf_cache.hash_inputs(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        indices=np.asarray(mesh.indices, dtype=np.int32),
        is_solid=bool(mesh.is_solid),
        narrow_band_range=(-args.narrow_band, args.narrow_band),
        target_voxel_size=None,
        max_resolution=args.max_resolution,
        margin=args.margin,
        texture_format=args.texture_format,
        sign_method_resolved=args.sign_method,
        winding_threshold=0.5,
        scale=None,
    )
    return _sdf_cache.cache_path(args.cache_dir, cache_hash)


def _exposed_voxel_surface(
    occupancy: np.ndarray, voxel_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight triangle mesh from the exposed faces of filled voxels."""
    assert (
        occupancy.ndim == 3 and occupancy.dtype == np.bool_
    ), "occupancy 必须是三维 bool 数组。"
    linear = voxel_transform[:3, :3]
    origin = voxel_transform[:3, 3]
    directions_and_corners = (
        (
            (-1, 0, 0),
            (
                (-0.5, -0.5, -0.5),
                (-0.5, -0.5, 0.5),
                (-0.5, 0.5, 0.5),
                (-0.5, 0.5, -0.5),
            ),
        ),
        (
            (1, 0, 0),
            ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)),
        ),
        (
            (0, -1, 0),
            (
                (-0.5, -0.5, -0.5),
                (0.5, -0.5, -0.5),
                (0.5, -0.5, 0.5),
                (-0.5, -0.5, 0.5),
            ),
        ),
        (
            (0, 1, 0),
            ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)),
        ),
        (
            (0, 0, -1),
            (
                (-0.5, -0.5, -0.5),
                (-0.5, 0.5, -0.5),
                (0.5, 0.5, -0.5),
                (0.5, -0.5, -0.5),
            ),
        ),
        (
            (0, 0, 1),
            ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)),
        ),
    )
    vertex_blocks: list[np.ndarray] = []
    face_count = 0
    for direction, corners in directions_and_corners:
        neighbor = np.zeros_like(occupancy)
        source_slices = []
        target_slices = []
        for axis_shift, axis_size in zip(direction, occupancy.shape, strict=True):
            if axis_shift == -1:
                source_slices.append(slice(1, axis_size))
                target_slices.append(slice(0, axis_size - 1))
            elif axis_shift == 1:
                source_slices.append(slice(0, axis_size - 1))
                target_slices.append(slice(1, axis_size))
            else:
                source_slices.append(slice(None))
                target_slices.append(slice(None))
        neighbor[tuple(source_slices)] = occupancy[tuple(target_slices)]
        cells = np.argwhere(occupancy & ~neighbor).astype(np.float32)
        if len(cells) == 0:
            continue
        corners_array = np.asarray(corners, dtype=np.float32)
        local_vertices = cells[:, None, :] + corners_array[None, :, :]
        vertex_blocks.append(local_vertices.reshape(-1, 3))
        face_count += len(cells) * 2

    assert vertex_blocks, "实心化后没有暴露表面。"
    voxel_vertices = np.concatenate(vertex_blocks)
    vertices = voxel_vertices @ linear.T + origin
    indices = np.arange(len(vertices), dtype=np.int32).reshape(-1, 4)
    triangles = np.concatenate((indices[:, (0, 1, 2)], indices[:, (0, 2, 3)])).reshape(
        -1
    )
    assert len(triangles) // 3 == face_count, "体素表面三角形数不一致。"
    # Share vertices along voxel edges so the exported USD is genuinely watertight,
    # rather than merely a collection of coincident quad faces.
    welded_vertices, welded_indices = np.unique(
        vertices.astype(np.float32), axis=0, return_inverse=True
    )
    return welded_vertices, welded_indices[triangles].astype(np.int32)


def solidify_mesh(
    vertices: np.ndarray, indices: np.ndarray, voxel_size: float, close_iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed collision proxy by voxelizing, sealing, and filling a mesh."""
    from scipy import ndimage
    import trimesh

    source_mesh = trimesh.Trimesh(
        vertices=vertices, faces=indices.reshape(-1, 3), process=False
    )
    voxel_grid = source_mesh.voxelized(pitch=voxel_size)
    # ``VoxelGrid.fill`` first fills regions enclosed by the rasterized surface.
    # This is essential for a jug/cap pair whose source triangles are not welded.
    occupancy = np.asarray(voxel_grid.fill().matrix, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    if close_iterations > 0:
        occupancy = ndimage.binary_closing(
            occupancy, structure=structure, iterations=close_iterations
        )
    occupancy = ndimage.binary_fill_holes(occupancy)
    proxy_vertices, proxy_indices = _exposed_voxel_surface(
        occupancy, np.asarray(voxel_grid.transform)
    )
    print(
        f"[INFO] 实心化：pitch={voxel_size:.6f} m，网格={occupancy.shape}，"
        f"占据体素={int(occupancy.sum())}。"
    )
    return proxy_vertices, proxy_indices


def write_combined_usd(
    output_path: Path, vertices: np.ndarray, indices: np.ndarray
) -> None:
    """Write one world-space ``CombinedMesh`` prim to a standalone USD file."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    assert len(indices) % 3 == 0, "合并网格索引数必须是 3 的整数倍。"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    assert stage is not None, f"无法创建输出 USD：{output_path}"
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/CombinedMesh"))
    mesh.CreatePointsAttr().Set(vertices.tolist())
    mesh.CreateFaceVertexCountsAttr().Set([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr().Set(indices.astype(np.int32).tolist())
    mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()


def main() -> None:
    """Bake the combined USD collision mesh and write its Newton cache plus a manifest."""
    import newton

    args = parse_args()
    vertices, indices, mesh_paths = load_combined_mesh(args.usd)
    if args.solidify:
        vertices, indices = solidify_mesh(
            vertices,
            indices,
            voxel_size=args.solidify_voxel_size,
            close_iterations=args.solidify_close_iterations,
        )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_usd = args.output_usd
    if args.visualize and output_usd is None:
        output_usd = args.cache_dir / f"{args.usd.stem}_combined.usda"
    if output_usd is not None:
        write_combined_usd(output_usd, vertices, indices)

    # ``compute_inertia=False`` accepts open source meshes.  The baked SDF itself is
    # still a single field over all meshes, and ``winding`` robustly classifies nested
    # or overlapping cap/body surfaces better than parity rays.
    mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False)
    cache_path = _cache_path_for_mesh(mesh, args)
    mesh.build_sdf(
        max_resolution=args.max_resolution,
        narrow_band_range=(-args.narrow_band, args.narrow_band),
        margin=args.margin,
        texture_format=args.texture_format,
        sign_method=args.sign_method,
        cache_dir=args.cache_dir,
    )
    assert mesh.sdf is not None, "整体 SDF 构建失败。"
    assert cache_path.is_file(), f"Newton 未写入预期 cache 文件：{cache_path}"

    manifest_path = cache_path.with_suffix(".manifest.json")
    manifest = {
        "usd": str(args.usd.resolve()),
        "mesh_prims": mesh_paths,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(indices) // 3),
        "cache_file": str(cache_path.resolve()),
        "max_resolution": args.max_resolution,
        "narrow_band_range_m": [-args.narrow_band, args.narrow_band],
        "margin_m": args.margin,
        "sign_method": args.sign_method,
        "texture_format": args.texture_format,
        "solidified": args.solidify,
        "solidify_voxel_size_m": args.solidify_voxel_size if args.solidify else None,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[INFO] 合并 {len(mesh_paths)} 个 Mesh：{len(vertices)} 顶点，{len(indices) // 3} 三角形。"
    )
    if output_usd is not None:
        print(f"[INFO] 合并 USD：{output_usd}")
    print(f"[INFO] Newton SDF cache：{cache_path}")
    print(f"[INFO] 清单：{manifest_path}")
    if args.visualize:
        visualizer_path = Path(__file__).with_name("sdf_visualizer.py")
        print(
            "[INFO] 正在启动 Newton SDF 可视化器；在 UI 中勾选 SDF 图层即可查看体素采样。"
        )
        os.execv(
            sys.executable,
            [sys.executable, str(visualizer_path), str(output_usd)],
        )


if __name__ == "__main__":
    main()
