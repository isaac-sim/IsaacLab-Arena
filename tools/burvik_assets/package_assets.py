# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create a portable assembled USD and audit the per-part schemas; run in Blender."""

import json
import os
from pathlib import Path

from mathutils import Vector
from pxr import Gf, Usd, UsdGeom, UsdPhysics

root = Path(os.environ.get("BURVIK_ASSET_DIR", Path(__file__).resolve().parents[2] / "artifacts/ikea_burvik"))
spec = json.loads((root / "specification.json").read_text())
stage = Usd.Stage.CreateNew(str(root / "assets" / "burvik_assembled.usda"))
default = UsdGeom.Xform.Define(stage, "/Burvik")
stage.SetDefaultPrim(default.GetPrim())
UsdGeom.SetStageMetersPerUnit(stage, 1)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
audit = {}
poses = {"tray": ((0, 0, 0.42), (1, 0, 0, 0)), "handle": ((0, 0, 0.42), (1, 0, 0, 0))}
for i, socket in enumerate(spec["sockets"]):
    pos = Vector(socket["seat"]) + Vector((0, 0, 0.42))
    q = Vector(socket["axis"]).to_track_quat("Z", "Y")
    poses[f"leg_{i + 1}"] = (tuple(pos), tuple(q))
for name, (p, q) in poses.items():
    path = "/Burvik/Parts/" + name
    part = UsdGeom.Xform.Define(stage, path)
    part.GetPrim().GetReferences().AddReference("./" + name + ".usd")
    part.AddTranslateOp().Set(Gf.Vec3d(*p))
    part.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    source = Usd.Stage.Open(str(root / "assets" / f"{name}.usd"))
    colliders = [prim for prim in source.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    bodies = [prim for prim in source.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 1
    assert len(colliders) == spec["collider_counts"][name]
    assert source.GetDefaultPrim().GetPath().pathString == "/Asset"
    assert UsdGeom.GetStageMetersPerUnit(source) == 1
    for collider in colliders:
        assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == "convexHull"
        mesh = UsdGeom.Mesh(collider)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        assert sum(counts) == len(indices)
        assert max(indices) < len(mesh.GetPointsAttr().Get())
    audit[name] = {
        "rigid_bodies": len(bodies),
        "convex_colliders": len(colliders),
        "mass_kg": UsdPhysics.MassAPI(bodies[0]).GetMassAttr().Get(),
        "meters_per_unit": 1,
        "up_axis": "Z",
        "passed": True,
    }
    if name != "tray":
        joint = UsdPhysics.FixedJoint.Define(stage, "/Burvik/Connections/" + name)
        joint.CreateBody0Rel().SetTargets(["/Burvik/Parts/tray"])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(p[0], p[1], p[2] - 0.42))
        joint.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateCollisionEnabledAttr(False)
stage.GetRootLayer().Save()
(root / "asset_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit))
