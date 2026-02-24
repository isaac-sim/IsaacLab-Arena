# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a triangulated OBJ mesh into a simple USD mesh stage.

This utility is intended for collision-proxy workflows where we already have an
OBJ export from the Matterport visual USD and want a lightweight USD mesh back
for ``--matterport_collision_usd_path``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

def _parse_obj(obj_path: Path) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Parse vertex and face data from a simple OBJ file."""
    vertices: list[tuple[float, float, float]] = []
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []

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
                face_vertex_counts.append(len(face))
                face_vertex_indices.extend(face)

    if not vertices or not face_vertex_counts:
        raise RuntimeError(f"OBJ parse produced no usable mesh data: {obj_path}")

    return vertices, face_vertex_counts, face_vertex_indices


def _format_int_array(values: list[int]) -> str:
    """Format a USD int array."""
    return "[" + ", ".join(str(value) for value in values) + "]"


def _format_points(vertices: list[tuple[float, float, float]]) -> str:
    """Format a USD point3f array."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OBJ mesh to a simple USD mesh.")
    parser.add_argument("--input_obj", type=Path, required=True, help="Input OBJ mesh path.")
    parser.add_argument("--output_usd", type=Path, required=True, help="Output USD path.")
    parser.add_argument(
        "--mesh_prim_path",
        type=str,
        default="/World/CollisionProxy/mesh",
        help="Prim path for the generated mesh.",
    )
    args = parser.parse_args()

    input_obj = args.input_obj.expanduser().resolve()
    output_usd = args.output_usd.expanduser().resolve()
    output_usd.parent.mkdir(parents=True, exist_ok=True)

    vertices, face_vertex_counts, face_vertex_indices = _parse_obj(input_obj)
    _write_usda(
        usd_path=output_usd,
        vertices=vertices,
        face_vertex_counts=face_vertex_counts,
        face_vertex_indices=face_vertex_indices,
        mesh_prim_path=args.mesh_prim_path,
    )
    print(
        f"[obj_to_usd_mesh] wrote {output_usd} "
        f"(vertices={len(vertices)} faces={len(face_vertex_counts)})"
    )


if __name__ == "__main__":
    main()
