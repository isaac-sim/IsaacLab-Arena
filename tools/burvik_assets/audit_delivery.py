# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit delivered USD composition and the recorded final support contacts in Isaac Sim."""

import argparse
import json
import os
from pathlib import Path

from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

app = get_app_launcher(argparse.Namespace(headless=True, enable_cameras=False, visualizer=[], livestream=-1)).app

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

root = Path(os.environ.get("BURVIK_ASSET_DIR", Path(__file__).resolve().parents[2] / "artifacts/ikea_burvik"))
spec = json.loads((root / "specification.json").read_text())
report = json.loads((root / "results" / "validation.json").read_text())
assert report["passed"] and report["all_bodies_dynamic"] and report["world_fixed_joints"] == 0
final = report["stability"]["after_disturbance"]["parts"]
audit = {"parts": {}, "composition": {}, "support": {}}
total_mass = 0
weighted_com = np.zeros(3)
foot_points = []
for name, mass in spec["masses_kg"].items():
    stage = Usd.Stage.Open(str(root / "assets" / f"{name}.usd"))
    body = stage.GetDefaultPrim()
    assert body and body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    rigid_bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(rigid_bodies) == 1
    assert len(collisions) == spec["collider_counts"][name]
    visual_triangles = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
            assert max(counts) == 3, str(prim.GetPath())
            visual_triangles += len(counts)
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root / "assets" / f"{name}.usd")))
    assert not unresolved, unresolved
    state = final[name]["final_pose"]
    p = np.array(state[:3])
    r = Rotation.from_quat([*state[4:7], state[3]])
    com = np.array(UsdPhysics.MassAPI(body).GetCenterOfMassAttr().Get())
    weighted_com += mass * (p + r.apply(com))
    total_mass += mass
    item = {
        "rigid_bodies": 1,
        "convex_colliders": len(collisions),
        "visual_triangles": visual_triangles,
        "mass_kg": mass,
        "unresolved_assets": [],
    }
    if name.startswith("leg"):
        foot = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Collisions/foot"))
        points = np.array(foot.GetPointsAttr().Get())
        world_points = p + r.apply(points)
        gap = float(world_points[:, 2].min() - 0.75)
        item["foot_gap_m"] = gap
        assert abs(gap) < 0.001, (name, gap)
        foot_points.append(world_points.mean(axis=0)[:2])
    audit["parts"][name] = item

center = weighted_com / total_mass
hull = ConvexHull(np.array(foot_points))
signed = hull.equations[:, :2] @ center[:2] + hull.equations[:, 2]
margin = float(-max(signed))
assert margin > 0.05, margin
normal_force = sum(v["mean_contact_force_N"][2] for v in final.values())
force_ratio = normal_force / (total_mass * 9.81)
assert 0.98 < force_ratio < 1.02, force_ratio
audit["support"] = {
    "total_mass_kg": total_mass,
    "com_world_m": center.tolist(),
    "support_polygon_margin_m": margin,
    "vertical_contact_force_N": normal_force,
    "weight_N": total_mass * 9.81,
    "force_to_weight_ratio": force_ratio,
}
for relative in ["assets/burvik_assembled.usda", "results/assembled_scene.usda"]:
    filename = root / relative
    stage = Usd.Stage.Open(str(filename))
    assert stage and stage.GetDefaultPrim()
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [UsdPhysics.FixedJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
    assert len(bodies) == 6 and len(joints) == 5
    for joint in joints:
        assert joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets()
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(filename)))
    assert not unresolved, unresolved
    audit["composition"][relative] = {
        "rigid_bodies": len(bodies),
        "part_to_part_joints": len(joints),
        "unresolved_assets": [],
    }
audit["passed"] = True
(root / "delivery_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)
app.close()
