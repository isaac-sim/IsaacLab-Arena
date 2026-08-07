# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline generator for the pre-tetrahedralized deformable-object USD assets.

Arena's runtime image does not ship ``pytetwild``/``pyvista``, so volume deformables cannot be
tetrahedralized on the fly from primitive meshes. Instead we tetrahedralize once, offline, and commit
the resulting ``UsdGeom.TetMesh`` assets next to this script. A ``UsdFileCfg`` referencing one of these
files spawns a deformable without needing ``pytetwild`` at runtime (Isaac Lab detects the existing
``TetMesh`` and derives the visual surface from it).

Run inside the Arena container with ``pytetwild`` + ``pyvista`` installed::

    /isaac-sim/python.sh -m pip install pytetwild pyvista
    /isaac-sim/python.sh isaaclab_arena/assets/usd/generate_deformable_tet_meshes.py

The tetrahedralization parameters mirror Isaac Lab's automatic path in
``isaaclab.sim.schemas.define_deformable_body_properties`` so the offline result matches on-the-fly
generation. The output files are checked in; re-run only when a shape or size changes.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pathlib import Path

from pxr import Usd, UsdGeom, Vt
from pytetwild import tetrahedralize

_OUT_DIR = Path(__file__).resolve().parent


def _tetrahedralize(surface: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a surface mesh and fix tet winding to positive signed volume."""
    points, tets = tetrahedralize(
        np.asarray(surface.vertices, dtype=np.float32),
        np.asarray(surface.faces, dtype=np.int32).reshape(-1, 3),
        edge_length_fac=0.1,
        simplify=False,
        epsilon=1e-2,
        coarsen=True,
    )
    points = np.asarray(points, dtype=np.float32)
    tets = np.asarray(tets, dtype=np.int32).reshape(-1, 4)
    # UsdGeom.TetMesh / ComputeSurfaceFaces require positive signed volume; flip inverted tets.
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    signed_vol = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))
    inverted = signed_vol < 0.0
    tets[inverted] = tets[inverted][:, [0, 1, 3, 2]]
    return points, tets


def _write_tet_usd(points: np.ndarray, tets: np.ndarray, out_path: Path, prim_name: str) -> None:
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform = UsdGeom.Xform.Define(stage, f"/{prim_name}")
    stage.SetDefaultPrim(xform.GetPrim())
    tet_mesh = UsdGeom.TetMesh.Define(stage, f"/{prim_name}/sim_mesh")
    tet_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    tet_mesh.CreateTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tets))
    stage.GetRootLayer().Save()
    print(f"wrote {out_path.name}: {len(points)} vertices, {len(tets)} tetrahedra")


def _structured_box_tets(
    length: float, half_width: float, half_height: float, num_segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a low-resolution tetrahedral prism without runtime tetrahedralization."""
    points = []
    for x in np.linspace(-length * 0.5, length * 0.5, num_segments + 1):
        for y in (-half_width, half_width):
            for z in (-half_height, half_height):
                points.append((x, y, z))

    def vid(i: int, y_id: int, z_id: int) -> int:
        return i * 4 + y_id * 2 + z_id

    cell_tets = ((0, 1, 7, 3), (0, 3, 7, 2), (0, 2, 7, 6), (0, 6, 7, 4), (0, 4, 7, 5), (0, 5, 7, 1))
    tets = []
    for i in range(num_segments):
        local = [
            vid(i, 0, 0),
            vid(i, 0, 1),
            vid(i, 1, 0),
            vid(i, 1, 1),
            vid(i + 1, 0, 0),
            vid(i + 1, 0, 1),
            vid(i + 1, 1, 0),
            vid(i + 1, 1, 1),
        ]
        tets.extend(tuple(local[index] for index in tet) for tet in cell_tets)
    return np.asarray(points, dtype=np.float32), np.asarray(tets, dtype=np.int32)


def main() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
    pts, tets = _tetrahedralize(sphere)
    _write_tet_usd(pts, tets, _OUT_DIR / "procedural_deformable_sphere_tet.usda", "DeformableSphere")

    cube = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
    pts, tets = _tetrahedralize(cube)
    _write_tet_usd(pts, tets, _OUT_DIR / "procedural_deformable_cube_tet.usda", "DeformableCube")

    pts, tets = _structured_box_tets(length=0.08, half_width=0.02, half_height=0.02, num_segments=4)
    _write_tet_usd(pts, tets, _OUT_DIR / "procedural_deformable_volume_block_tet.usda", "DeformableVolumeBlock")

    pts, tets = _structured_box_tets(length=0.4, half_width=0.012, half_height=0.012, num_segments=8)
    _write_tet_usd(pts, tets, _OUT_DIR / "procedural_deformable_cable_tet.usda", "DeformableCable")


if __name__ == "__main__":
    main()
