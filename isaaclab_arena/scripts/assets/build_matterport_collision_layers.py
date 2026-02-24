#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build collision-only USDA layers from a cleaned Matterport OBJ mesh.

This script is intended for the recommended collision workflow:

1. Start from a raw Matterport / MP3D mesh, not the already-converted visual USD.
2. Export or convert that raw mesh to a triangulated OBJ.
3. Optionally clean / simplify the OBJ in a mesh tool.
4. Run this script to emit floor-only, obstacle-only, and/or combined collision USDA assets.

The floor split is heuristic and tuned for single-floor indoor scenes:
  - triangle centroid z must lie within ``[floor_min_z, floor_max_z]``
  - triangle normal must point mostly upward
  - tiny triangles can be filtered out via ``min_face_area``

For multi-floor scenes or heavily tilted geometry, treat this as a starting point
and adjust thresholds or manually clean the exported mesh.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _parse_obj(obj_path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Parse a triangulated OBJ file into vertices and triangle faces."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                _, x_str, y_str, z_str = line.strip().split()[:4]
                vertices.append((float(x_str), float(y_str), float(z_str)))
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                face: list[int] = []
                for token in tokens:
                    vertex_index_str = token.split("/")[0]
                    if not vertex_index_str:
                        continue
                    vertex_index = int(vertex_index_str)
                    if vertex_index < 0:
                        vertex_index = len(vertices) + vertex_index
                    else:
                        vertex_index -= 1
                    face.append(vertex_index)
                if len(face) < 3:
                    continue
                for i in range(1, len(face) - 1):
                    faces.append((face[0], face[i], face[i + 1]))

    if not vertices or not faces:
        raise RuntimeError(f"OBJ parse produced no usable mesh data: {obj_path}")

    return vertices, faces


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _triangle_area(
    vertices: list[tuple[float, float, float]], face: tuple[int, int, int]
) -> float:
    p0, p1, p2 = (vertices[idx] for idx in face)
    return 0.5 * _norm(_cross(_sub(p1, p0), _sub(p2, p0)))


def _triangle_centroid_z(
    vertices: list[tuple[float, float, float]], face: tuple[int, int, int]
) -> float:
    p0, p1, p2 = (vertices[idx] for idx in face)
    return (p0[2] + p1[2] + p2[2]) / 3.0


def _triangle_normal_z(
    vertices: list[tuple[float, float, float]], face: tuple[int, int, int]
) -> float:
    p0, p1, p2 = (vertices[idx] for idx in face)
    cross = _cross(_sub(p1, p0), _sub(p2, p0))
    length = _norm(cross)
    if length <= 1e-12:
        return 0.0
    return cross[2] / length


def _split_floor_and_obstacles(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    floor_min_z: float,
    floor_max_z: float,
    floor_normal_min_z: float,
    min_face_area: float,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Split triangles into floor-like and obstacle-like sets."""
    floor_faces: list[tuple[int, int, int]] = []
    obstacle_faces: list[tuple[int, int, int]] = []

    for face in faces:
        area = _triangle_area(vertices, face)
        if area < min_face_area:
            continue

        centroid_z = _triangle_centroid_z(vertices, face)
        normal_z = _triangle_normal_z(vertices, face)
        is_floor_like = (
            floor_min_z <= centroid_z <= floor_max_z and normal_z >= floor_normal_min_z
        )
        if is_floor_like:
            floor_faces.append(face)
        else:
            obstacle_faces.append(face)

    return floor_faces, obstacle_faces


def _compact_mesh(
    vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Compact a mesh selection into a new vertex array and face indices."""
    used_indices: dict[int, int] = {}
    compact_vertices: list[tuple[float, float, float]] = []
    compact_face_indices: list[int] = []
    compact_face_counts: list[int] = []

    for face in faces:
        compact_face_counts.append(len(face))
        for old_idx in face:
            if old_idx not in used_indices:
                used_indices[old_idx] = len(compact_vertices)
                compact_vertices.append(vertices[old_idx])
            compact_face_indices.append(used_indices[old_idx])

    return compact_vertices, compact_face_counts, compact_face_indices


def _format_int_array(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _format_points(vertices: list[tuple[float, float, float]]) -> str:
    return "[" + ", ".join(f"({x:.8f}, {y:.8f}, {z:.8f})" for x, y, z in vertices) + "]"


def _write_usda(
    usd_path: Path,
    vertices: list[tuple[float, float, float]],
    face_vertex_counts: list[int],
    face_vertex_indices: list[int],
    mesh_prim_path: str,
) -> None:
    """Write a simple USDA stage containing a single mesh."""
    prim_parts = [part for part in mesh_prim_path.split("/") if part]
    if len(prim_parts) < 2 or prim_parts[0] != "World":
        raise RuntimeError("mesh_prim_path must start with /World and include a mesh name.")

    mesh_name = prim_parts[-1]
    xform_parts = prim_parts[:-1]

    lines: list[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        ")",
        "",
    ]

    indent = ""
    for index, prim_name in enumerate(xform_parts):
        lines.append(f'{indent}def Xform "{prim_name}"')
        lines.append(f"{indent}" + "{")
        indent += "    "
        if index == len(xform_parts) - 1:
            lines.append(f'{indent}def Mesh "{mesh_name}"')
            lines.append(f"{indent}" + "{")
            lines.append(f"{indent}    int[] faceVertexCounts = {_format_int_array(face_vertex_counts)}")
            lines.append(f"{indent}    int[] faceVertexIndices = {_format_int_array(face_vertex_indices)}")
            lines.append(f"{indent}    point3f[] points = {_format_points(vertices)}")
            lines.append(f'{indent}    uniform token subdivisionScheme = "none"')
            lines.append(f"{indent}" + "}")

    for close_idx in range(len(xform_parts)):
        close_indent = "    " * (len(xform_parts) - close_idx - 1)
        lines.append(f"{close_indent}" + "}")

    usd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selection(
    *,
    output_path: Path | None,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    mesh_prim_path: str,
    label: str,
) -> None:
    if output_path is None:
        return
    if not faces:
        print(f"[build_matterport_collision_layers] skipping {label}: no faces selected")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_vertices, face_vertex_counts, face_vertex_indices = _compact_mesh(vertices, faces)
    _write_usda(
        usd_path=output_path,
        vertices=compact_vertices,
        face_vertex_counts=face_vertex_counts,
        face_vertex_indices=face_vertex_indices,
        mesh_prim_path=mesh_prim_path,
    )
    print(
        "[build_matterport_collision_layers] wrote "
        f"{label} -> {output_path} "
        f"(vertices={len(compact_vertices)} faces={len(face_vertex_counts)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build floor / obstacle / combined collision USDA files from a cleaned OBJ mesh."
    )
    parser.add_argument("--input_obj", type=Path, required=True, help="Input triangulated OBJ mesh path.")
    parser.add_argument(
        "--output_combined_usd",
        type=Path,
        default=None,
        help="Optional path for a combined collision USDA output.",
    )
    parser.add_argument(
        "--output_floor_usd",
        type=Path,
        default=None,
        help="Optional path for a floor-only collision USDA output.",
    )
    parser.add_argument(
        "--output_obstacle_usd",
        type=Path,
        default=None,
        help="Optional path for an obstacle-only collision USDA output.",
    )
    parser.add_argument(
        "--mesh_prim_prefix",
        type=str,
        default="/World/CollisionProxy",
        help="Prefix used for generated mesh prims.",
    )
    parser.add_argument(
        "--floor_min_z",
        type=float,
        default=-0.25,
        help="Minimum triangle centroid z to be considered floor-like.",
    )
    parser.add_argument(
        "--floor_max_z",
        type=float,
        default=0.35,
        help="Maximum triangle centroid z to be considered floor-like.",
    )
    parser.add_argument(
        "--floor_normal_min_z",
        type=float,
        default=0.9,
        help="Minimum triangle normal z-component for floor classification.",
    )
    parser.add_argument(
        "--min_face_area",
        type=float,
        default=1e-6,
        help="Drop tiny triangles below this area threshold before classification.",
    )
    args = parser.parse_args()

    if not (args.output_combined_usd or args.output_floor_usd or args.output_obstacle_usd):
        raise RuntimeError("At least one output path must be provided.")

    input_obj = args.input_obj.expanduser().resolve()
    vertices, faces = _parse_obj(input_obj)
    floor_faces, obstacle_faces = _split_floor_and_obstacles(
        vertices,
        faces,
        floor_min_z=args.floor_min_z,
        floor_max_z=args.floor_max_z,
        floor_normal_min_z=args.floor_normal_min_z,
        min_face_area=args.min_face_area,
    )

    print(
        "[build_matterport_collision_layers] input="
        f"{input_obj} faces={len(faces)} floor_faces={len(floor_faces)} obstacle_faces={len(obstacle_faces)}"
    )

    prim_prefix = args.mesh_prim_prefix.rstrip("/")
    _write_selection(
        output_path=args.output_combined_usd.expanduser().resolve() if args.output_combined_usd else None,
        vertices=vertices,
        faces=faces,
        mesh_prim_path=f"{prim_prefix}/combined",
        label="combined",
    )
    _write_selection(
        output_path=args.output_floor_usd.expanduser().resolve() if args.output_floor_usd else None,
        vertices=vertices,
        faces=floor_faces,
        mesh_prim_path=f"{prim_prefix}/floor",
        label="floor",
    )
    _write_selection(
        output_path=args.output_obstacle_usd.expanduser().resolve() if args.output_obstacle_usd else None,
        vertices=vertices,
        faces=obstacle_faces,
        mesh_prim_path=f"{prim_prefix}/obstacle",
        label="obstacle",
    )


if __name__ == "__main__":
    main()
