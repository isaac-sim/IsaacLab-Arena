"""Convert user-supplied SRT IGES geometry; create an explicitly approximate FEM mesh.

Run in the existing Docker with /isaac-sim/python.sh. CAD wheels are isolated
under ~/.local/srt-cad. No Kit startup, host Python, URDF or rigid finger joints.
The open IGES shells do NOT specify pneumatic cavities or material parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/examples_teleop/SFG-FNC3-N4049"
OUTPUT = ROOT / "examples/examples_teleop/assets/abb_irb1200_sfg_n4049"
# Measured from the assembly's central mounting hub; original IGES is in mm.
CAD_ORIGIN = np.array([-17.32277057884106, 48.9342604550781, 198.736983336])


def cad_imports():
    """Use isolated Docker dependencies without replacing Isaac Sim packages."""
    sys.path.insert(0, str(Path.home() / ".local/srt-cad"))
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    return locals()


def read_shells(path, deflection=0.35):
    """Sew matching edges and tessellate every shell, retaining open surfaces."""
    c = cad_imports()
    reader = c["IGESControl_Reader"]()
    assert int(reader.ReadFile(str(path))) == 1, path
    reader.TransferRoots()
    sew = c["BRepBuilderAPI_Sewing"](0.01)
    sew.Add(reader.OneShape())
    sew.Perform()
    shape = sew.SewedShape()
    c["BRepMesh_IncrementalMesh"](shape, deflection, False, 0.25, True).Perform()
    shells = []
    explorer = c["TopExp_Explorer"](shape, c["TopAbs_SHELL"])
    while explorer.More():
        shell = explorer.Current()
        faces = c["TopExp_Explorer"](shell, c["TopAbs_FACE"])
        points, triangles = [], []
        while faces.More():
            face = c["TopoDS"].Face_s(faces.Current())
            location = c["TopLoc_Location"]()
            tri = c["BRep_Tool"].Triangulation_s(face, location)
            assert tri is not None, "CAD face could not be tessellated"
            offset = len(points)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(location.Transformation())
                points.append([p.X(), p.Y(), p.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                ids = list(tri.Triangle(i).Get())
                if face.Orientation() == c["TopAbs_REVERSED"]:
                    ids[1], ids[2] = ids[2], ids[1]
                triangles.append([offset + j - 1 for j in ids])
            faces.Next()
        shells.append((np.asarray(points), np.asarray(triangles, dtype=np.int32), shell.Closed()))
        explorer.Next()
    assert shells, "No tessellated CAD shells"
    return shells


def to_mount(points):
    p = (points - CAD_ORIGIN) * 0.001
    return p[:, [0, 2, 1]] * [1, 1, -1]


def define_mesh(stage, path, points, faces, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int32).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))]))
    return mesh


def new_stage(path, root_name):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/" + root_name).GetPrim())
    return stage


def envelope_tets(points, pitch):
    """Voxelize the convex outer envelope, NOT the undisclosed air chambers."""
    hull = ConvexHull(points)
    low = np.floor(points.min(0) / pitch) * pitch
    count = np.ceil((points.max(0) - low) / pitch).astype(int)
    cells = np.stack(np.meshgrid(*[np.arange(n) for n in count], indexing="ij"), axis=-1).reshape(-1, 3)
    centers = low + (cells + 0.5) * pitch
    inside = np.ones(len(cells), dtype=bool)
    for eq in hull.equations:
        inside &= centers @ eq[:3] + eq[3] <= 0
    cells = cells[inside]
    # Six positively oriented, conforming tetrahedra around the 0--7 diagonal.
    offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    pattern = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                        [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    keys, inverse = np.unique((cells[:, None, :] + offsets).reshape(-1, 3), axis=0, return_inverse=True)
    vertices = low + keys * pitch
    tets = inverse.reshape(-1, 8)[:, pattern].reshape(-1, 4)
    xyz = vertices[tets]
    det = np.linalg.det(np.stack([xyz[:, i] - xyz[:, 0] for i in (1, 2, 3)], axis=-1))
    assert np.all(det > 0), "Inverted reference tetrahedra"
    return vertices, tets.astype(np.int32)


def compose(abb, gripper, output):
    assembly = new_stage(output / "assembly.usda", "abb_irb1200_sfg_n4049")
    assembly.GetDefaultPrim().GetReferences().AddReference(os.path.relpath(abb, output))
    flange = next(p for p in assembly.Traverse() if p.GetName() == "link_6" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    mount = UsdGeom.Xform.Define(assembly, flange.GetPath().AppendChild("sfg_mount"))
    mount.GetPrim().GetReferences().AddReference(os.path.relpath(gripper, output))
    mount.ClearXformOpOrder()
    mount.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.sqrt(.5), 0, math.sqrt(.5), 0))
    assembly.GetRootLayer().Save()
    target = output / "abb_irb1200_sfg_n4049.usda"
    assembly.Flatten().Export(str(target))
    verify = Usd.Stage.Open(str(target))
    assert len([p for p in verify.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]) == 6
    assert not [p for p in verify.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--pitch", type=float, default=.004, help="Approximate FEM voxel edge, metres")
    parser.add_argument("--abb-usd", type=Path, default=ROOT / "isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.usda")
    args = parser.parse_args()
    assert .002 <= args.pitch <= .01
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "User-downloaded SRT CAD; redistribution rights not inferred", "files": {},
                "units": "IGES mm -> USD m", "cad_mount_origin_mm": CAD_ORIGIN.tolist(),
                "approximation": "Convex-envelope voxel FEM; no internal gas cavities, no calibrated silicone or pressure law",
                "pitch_m": args.pitch, "parts": [], "soft_parts": []}
    all_fem = {}
    for name in ("N4049", "SFG-FNC3-N4049"):
        src = args.source_dir / (name + ".IGS")
        manifest["files"][src.name] = {"sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "bytes": src.stat().st_size}
        shells = read_shells(src)
        standalone = name == "N4049"
        stage = new_stage(out / ("single_finger_cad.usda" if standalone else "sfg_n4049_cad.usda"), "SFG")
        soft_index = 0
        for i, (raw, faces, closed) in enumerate(shells):
            points = raw * .001 if standalone else to_mount(raw)
            dims = np.ptp(points, axis=0)
            # Three silicone shells are the only parts reaching >140mm from
            # the mounting plane, with >55mm axial span (verified CAD bounds).
            soft = not standalone and points[:, 2].max() > .14 and dims[2] > .055
            label = f"soft_finger_{soft_index}" if soft else f"cad_part_{i:02d}"
            mesh = define_mesh(stage, f"/SFG/{label}", points, faces, (.22, .62, .78) if soft else (.52, .55, .58))
            if not standalone:
                manifest["parts"].append({"index": i, "name": label, "closed_shell": bool(closed),
                                          "bbox_m": [points.min(0).tolist(), points.max(0).tolist()], "triangles": len(faces)})
            if soft:
                mesh.GetPrim().CreateAttribute("srt:deformableVisual", Sdf.ValueTypeNames.Bool).Set(True)
                vertices, tets = envelope_tets(points, args.pitch)
                all_fem[f"points_{soft_index}"] = vertices.astype(np.float32)
                all_fem[f"tets_{soft_index}"] = tets
                root_ids = np.flatnonzero(vertices[:, 2] <= vertices[:, 2].min() + args.pitch * 1.1)
                all_fem[f"roots_{soft_index}"] = root_ids.astype(np.int32)
                tet = stage.DefinePrim(f"/SFG/FEM/finger_{soft_index}", "TetMesh")
                tet.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                tet.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(Vt.Vec4iArray.FromNumpy(tets))
                tet.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
                manifest["soft_parts"].append({"cad_part": i, "particles": len(vertices), "tets": len(tets), "root_particles": len(root_ids)})
                soft_index += 1
        if not standalone:
            assert soft_index == 3, manifest["soft_parts"]
        stage.GetRootLayer().Save()
        print(f"[CAD] {name}: {len(shells)} shells -> {stage.GetRootLayer().identifier}", flush=True)
    np.savez_compressed(out / "soft_fingers.npz", **all_fem)
    combined = compose(args.abb_usd.resolve(), out / "sfg_n4049_cad.usda", out)
    manifest["combined_usd"] = combined.name
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[USD] {combined}\n[FEM] {manifest['soft_parts']}", flush=True)


if __name__ == "__main__":
    main()