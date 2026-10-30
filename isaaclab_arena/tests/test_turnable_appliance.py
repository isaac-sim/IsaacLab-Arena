# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check appliance wrappers preserve source files and expose their rotary controls."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_turnable_appliance(simulation_app):
    import math
    import tempfile
    from pathlib import Path

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.affordances.turnable import Turnable
    from isaaclab_arena.assets.turnable_appliance import TurnableAppliance

    with tempfile.TemporaryDirectory() as directory:
        for has_root in (False, True):
            path = Path(directory) / f"appliance_{has_root}.usda"
            stage = Usd.Stage.CreateNew(str(path))
            root = UsdGeom.Xform.Define(stage, "/Appliance").GetPrim()
            stage.SetDefaultPrim(root)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            for name in ("base", "knob"):
                body = UsdGeom.Xform.Define(stage, f"/Appliance/{name}").GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(body)
                shape = UsdGeom.Cube.Define(stage, f"/Appliance/{name}/shape")
                shape.CreateSizeAttr(0.05)
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            if has_root:
                UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/Appliance/base"))
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/Appliance/speed")
            joint.CreateBody0Rel().SetTargets(["/Appliance/base/shape"])
            joint.CreateBody1Rel().SetTargets(["/Appliance/knob"])
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(180.0)
            stage.GetRootLayer().Save()
            original = path.read_bytes()
            appliance = TurnableAppliance(
                usd_path=str(path),
                turnable_joint_name="speed",
                min_level_angle_deg=30,
                max_level_angle_deg=150,
                num_levels=5,
                knob_damping=0.05,
                initial_joint_positions={"speed": 0.1},
            )
            assert isinstance(appliance, Turnable)
            assert appliance.object_cfg.init_state.joint_pos == {"speed": 0.1}
            wrapped = Usd.Stage.Open(appliance.usd_path)
            roots = [p for p in wrapped.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            assert len(roots) == 1
            assert roots[0].GetPath() == "/Asset/base"
            assert roots[0].HasAPI(UsdPhysics.RigidBodyAPI)
            wrapped_joint = UsdPhysics.RevoluteJoint(wrapped.GetPrimAtPath("/Asset/speed"))
            assert wrapped_joint.GetUpperLimitAttr().Get() == 180.0
            assert wrapped_joint.GetBody1Rel().GetTargets()[0] == "/Asset/knob"
            drive = UsdPhysics.DriveAPI(wrapped_joint.GetPrim(), "angular")
            assert abs(drive.GetDampingAttr().Get() * 180 / math.pi - 0.05) < 1e-6
            assert drive.GetStiffnessAttr().Get() == 0
            assert drive.GetTargetVelocityAttr().Get() == 0
            assert path.read_bytes() == original
    return True


def test_turnable_appliance():
    assert run_function_with_persistent_simulation_app(_test_turnable_appliance)
