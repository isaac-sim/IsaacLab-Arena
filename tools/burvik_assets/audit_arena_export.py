# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the exported Arena scene against its recorded final dynamic state."""

import argparse
import json
from pathlib import Path

from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording", nargs="?", type=Path, default=Path("artifacts/ikea_burvik/arena_verified"))
args = parser.parse_args()

launcher = get_app_launcher(
    argparse.Namespace(headless=True, enable_cameras=False, visualizer=[], livestream=-1, device="cpu")
)
app = launcher.app
exit_code = 0
try:
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics

    output = args.recording
    stage = Usd.Stage.Open(str(output / "arena_assembled.usda"))
    assert stage and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    final = np.load(output / "trajectory.npz", allow_pickle=True)["frames"][-1]["poses"]
    report = {"meters_per_unit": 1.0, "up_axis": "Z", "parts": {}, "connections": {}}
    root = "/World/envs/env_0"
    for name, pose in final.items():
        prim = stage.GetPrimAtPath(f"{root}/{name}")
        body = UsdPhysics.RigidBodyAPI(prim)
        assert body and body.GetRigidBodyEnabledAttr().Get() and not body.GetKinematicEnabledAttr().Get()
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position_error = float(np.linalg.norm(np.array(transform.ExtractTranslation()) - pose[:3]))
        assert position_error < 1e-5, (name, position_error)
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert colliders
        report["parts"][name] = {
            "dynamic": True,
            "collider_count": len(colliders),
            "final_pose_error_m": position_error,
        }
        if name != "tray":
            joint = UsdPhysics.FixedJoint.Get(stage, f"{root}/BurvikConnections/{name}")
            assert joint and joint.GetJointEnabledAttr().Get()
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            assert list(map(str, body0)) == [f"{root}/tray"]
            assert list(map(str, body1)) == [f"{root}/{name}"]
            report["connections"][name] = {"enabled": True, "body0": str(body0[0]), "body1": str(body1[0])}
    validation = json.loads((output / "validation.json").read_text())
    assert validation["passed"] and validation["all_parts_dynamic"]
    assert validation["final_subtask_completion"] == [[True] * 5]
    assert len(validation["progress_events"]) == 5
    assert validation["stability"]["assembled_free"]["passed"]
    assert not validation["stability"]["assembled_free"]["part_guidance_active"]
    report["passed"] = True
    (output / "export_audit.json").write_text(json.dumps(report, indent=2))
    print("ARENA_EXPORTED_SCENE_AUDIT_PASSED", flush=True)
except Exception:
    exit_code = 1
    raise
finally:
    app.close(exit_code=exit_code)
