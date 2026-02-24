#!/usr/bin/env python3

"""Filter simple collision USDA meshes and optionally add viewport colors."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MeshData:
    mesh_name: str
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, ...]]


@dataclass
class ComponentStats:
    face_indices: list[int]
    min_z: float
    max_z: float
    diag: float


def _parse_usda_mesh(path: Path) -> MeshData:
    text = path.read_text(encoding="utf-8")
    mesh_name_match = re.search(r'def Mesh "([^"]+)"', text)
    points_match = re.search(r"point3f\[\] points = \[(.*?)\]\n", text, re.S)
    counts_match = re.search(r"int\[\] faceVertexCounts = \[(.*?)\]\n", text, re.S)
    indices_match = re.search(r"int\[\] faceVertexIndices = \[(.*?)\]\n", text, re.S)
    if not (mesh_name_match and points_match and counts_match and indices_match):
        raise RuntimeError(f"Failed to parse simple collision USDA mesh: {path}")

    vertices = [
        tuple(map(float, match))
        for match in re.findall(r"\(([-0-9.eE]+),\s*([-0-9.eE]+),\s*([-0-9.eE]+)\)", points_match.group(1))
    ]
    face_counts = [int(value) for value in re.findall(r"-?\d+", counts_match.group(1))]
    face_indices_flat = [int(value) for value in re.findall(r"-?\d+", indices_match.group(1))]

    faces: list[tuple[int, ...]] = []
    cursor = 0
    for face_count in face_counts:
        faces.append(tuple(face_indices_flat[cursor : cursor + face_count]))
        cursor += face_count

    return MeshData(mesh_name=mesh_name_match.group(1), vertices=vertices, faces=faces)


def _iter_connected_components(mesh: MeshData) -> list[ComponentStats]:
    triangle_faces = [tuple(face) for face in mesh.faces if len(face) == 3]
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(triangle_faces):
        for vertex_index in face:
            vertex_to_faces[vertex_index].append(face_index)

    seen = [False] * len(triangle_faces)
    components: list[ComponentStats] = []
    for start_index in range(len(triangle_faces)):
        if seen[start_index]:
            continue

        queue = deque([start_index])
        seen[start_index] = True
        component_faces: list[int] = []
        component_vertices: set[int] = set()
        while queue:
            face_index = queue.popleft()
            component_faces.append(face_index)
            for vertex_index in triangle_faces[face_index]:
                component_vertices.add(vertex_index)
                for neighbor_face_index in vertex_to_faces[vertex_index]:
                    if not seen[neighbor_face_index]:
                        seen[neighbor_face_index] = True
                        queue.append(neighbor_face_index)

        xs = [mesh.vertices[index][0] for index in component_vertices]
        ys = [mesh.vertices[index][1] for index in component_vertices]
        zs = [mesh.vertices[index][2] for index in component_vertices]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)
        components.append(
            ComponentStats(
                face_indices=component_faces,
                min_z=min(zs),
                max_z=max(zs),
                diag=math.sqrt(dx * dx + dy * dy + dz * dz),
            )
        )

    return components


def _should_remove_ground_component(component: ComponentStats, args: argparse.Namespace) -> bool:
    return (
        args.ground_max_faces > 0
        and len(component.face_indices) <= args.ground_max_faces
        and component.max_z <= args.ground_max_top_z
        and component.diag <= args.ground_max_diag
    )


def _should_remove_floating_component(component: ComponentStats, args: argparse.Namespace) -> bool:
    return (
        args.floating_max_faces > 0
        and len(component.face_indices) <= args.floating_max_faces
        and component.min_z >= args.floating_min_bottom_z
        and (args.floating_max_top_z <= 0 or component.max_z <= args.floating_max_top_z)
        and component.diag <= args.floating_max_diag
    )


def _compact_mesh(
    vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    used_indices: dict[int, int] = {}
    compact_vertices: list[tuple[float, float, float]] = []
    compact_face_counts: list[int] = []
    compact_face_indices: list[int] = []

    for face in faces:
        compact_face_counts.append(len(face))
        for old_index in face:
            if old_index not in used_indices:
                used_indices[old_index] = len(compact_vertices)
                compact_vertices.append(vertices[old_index])
            compact_face_indices.append(used_indices[old_index])

    return compact_vertices, compact_face_counts, compact_face_indices


def _format_points(vertices: list[tuple[float, float, float]]) -> str:
    return "[" + ", ".join(f"({x:.8f}, {y:.8f}, {z:.8f})" for x, y, z in vertices) + "]"


def _format_int_array(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _build_output_text(
    mesh_name: str,
    vertices: list[tuple[float, float, float]],
    face_counts: list[int],
    face_indices: list[int],
    display_color: tuple[float, float, float] | None,
    display_opacity: float | None,
) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "CollisionProxy"',
        "    {",
        f'        def Mesh "{mesh_name}"',
        "        {",
        f"            int[] faceVertexCounts = {_format_int_array(face_counts)}",
        f"            int[] faceVertexIndices = {_format_int_array(face_indices)}",
        f"            point3f[] points = {_format_points(vertices)}",
    ]
    if display_color is not None:
        r, g, b = display_color
        lines.append(f"            color3f[] primvars:displayColor = [({r:.4f}, {g:.4f}, {b:.4f})]")
        lines.append('            uniform token primvars:displayColor:interpolation = "constant"')
    if display_opacity is not None:
        lines.append(f"            float[] primvars:displayOpacity = [{display_opacity:.4f}]")
        lines.append('            uniform token primvars:displayOpacity:interpolation = "constant"')
    lines.extend(
        [
            '            uniform token subdivisionScheme = "none"',
            "        }",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter simple collision USDA meshes and optionally add display color.")
    parser.add_argument("--input_usd", type=Path, required=True, help="Input USDA produced by the collision pipeline.")
    parser.add_argument("--output_usd", type=Path, required=True, help="Output USDA path.")
    parser.add_argument(
        "--display_color",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional constant displayColor for viewport visualization.",
    )
    parser.add_argument(
        "--display_opacity",
        type=float,
        default=None,
        help="Optional constant displayOpacity for viewport visualization.",
    )
    parser.add_argument(
        "--ground_max_faces",
        type=int,
        default=0,
        help="Remove connected components at ground height with face counts at or below this value (0 disables).",
    )
    parser.add_argument(
        "--ground_max_top_z",
        type=float,
        default=0.0,
        help="Maximum component top z for ground-fragment removal.",
    )
    parser.add_argument(
        "--ground_max_diag",
        type=float,
        default=0.0,
        help="Maximum component diagonal length for ground-fragment removal.",
    )
    parser.add_argument(
        "--floating_max_faces",
        type=int,
        default=0,
        help="Remove floating connected components with face counts at or below this value (0 disables).",
    )
    parser.add_argument(
        "--floating_min_bottom_z",
        type=float,
        default=0.0,
        help="Minimum component bottom z for floating-fragment removal.",
    )
    parser.add_argument(
        "--floating_max_diag",
        type=float,
        default=0.0,
        help="Maximum component diagonal length for floating-fragment removal.",
    )
    parser.add_argument(
        "--floating_max_top_z",
        type=float,
        default=0.0,
        help="Optional maximum component top z for floating-fragment removal (0 disables).",
    )
    args = parser.parse_args()

    input_usd = args.input_usd.expanduser().resolve()
    output_usd = args.output_usd.expanduser().resolve()
    mesh = _parse_usda_mesh(input_usd)
    components = _iter_connected_components(mesh)

    removed_face_indices: set[int] = set()
    ground_removed = 0
    floating_removed = 0
    for component in components:
        if _should_remove_ground_component(component, args):
            removed_face_indices.update(component.face_indices)
            ground_removed += 1
            continue
        if _should_remove_floating_component(component, args):
            removed_face_indices.update(component.face_indices)
            floating_removed += 1

    triangle_faces = [tuple(face) for face in mesh.faces if len(face) == 3]
    kept_faces = [triangle_faces[index] for index in range(len(triangle_faces)) if index not in removed_face_indices]
    compact_vertices, face_counts, face_indices = _compact_mesh(mesh.vertices, kept_faces)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    output_usd.write_text(
        _build_output_text(
            mesh_name=mesh.mesh_name,
            vertices=compact_vertices,
            face_counts=face_counts,
            face_indices=face_indices,
            display_color=tuple(args.display_color) if args.display_color is not None else None,
            display_opacity=args.display_opacity,
        ),
        encoding="utf-8",
    )

    print(
        "[filter_collision_usda] "
        f"input={input_usd} output={output_usd} "
        f"kept_faces={len(face_counts)} removed_faces={len(removed_face_indices)} "
        f"removed_ground_components={ground_removed} removed_floating_components={floating_removed}"
    )


if __name__ == "__main__":
    main()
