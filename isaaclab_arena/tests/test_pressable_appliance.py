# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check appliance wrappers preserve source files and expose their button controls."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_pressable_appliance(simulation_app):
    import tempfile
    from pathlib import Path

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.affordances.pressable import Pressable
    from isaaclab_arena.assets.pressable_appliance import PressableAppliance

    with tempfile.TemporaryDirectory() as directory:
        for has_root in (False, True):
            path = Path(directory) / f"appliance_{has_root}.usda"
            stage = Usd.Stage.CreateNew(str(path))
            root = UsdGeom.Xform.Define(stage, "/Appliance").GetPrim()
            stage.SetDefaultPrim(root)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            for name in ("base", "button"):
                body = UsdGeom.Xform.Define(stage, f"/Appliance/{name}").GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(body)
                shape = UsdGeom.Cube.Define(stage, f"/Appliance/{name}/shape")
                shape.CreateSizeAttr(0.05)
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            if has_root:
                UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/Appliance/base"))
            joint = UsdPhysics.PrismaticJoint.Define(stage, "/Appliance/press")
            joint.CreateBody0Rel().SetTargets(["/Appliance/base/shape"])
            joint.CreateBody1Rel().SetTargets(["/Appliance/button"])
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(0.004)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateStiffnessAttr(20.0)
            drive.CreateDampingAttr(0.2)
            drive.CreateTargetPositionAttr(0.0)
            stage.GetRootLayer().Save()
            original = path.read_bytes()
            appliance = PressableAppliance(
                usd_path=str(path),
                pressable_joint_name="press",
                initial_joint_positions={"press": 0.001},
            )
            assert isinstance(appliance, Pressable)
            assert appliance.object_cfg.init_state.joint_pos == {"press": 0.001}
            wrapped = Usd.Stage.Open(appliance.usd_path)
            roots = [p for p in wrapped.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            assert len(roots) == 1
            assert roots[0].GetPath() == "/Asset/base"
            assert roots[0].HasAPI(UsdPhysics.RigidBodyAPI)
            wrapped_joint = UsdPhysics.PrismaticJoint(wrapped.GetPrimAtPath("/Asset/press"))
            assert abs(wrapped_joint.GetUpperLimitAttr().Get() - 0.004) < 1e-8
            assert wrapped_joint.GetBody1Rel().GetTargets()[0] == "/Asset/button"
            drive = UsdPhysics.DriveAPI(wrapped_joint.GetPrim(), "linear")
            assert drive.GetStiffnessAttr().Get() == 20.0
            assert abs(drive.GetDampingAttr().Get() - 0.2) < 1e-7
            assert drive.GetTargetPositionAttr().Get() == 0.0
            assert path.read_bytes() == original
    return True


def test_pressable_appliance():
    assert run_function_with_persistent_simulation_app(_test_pressable_appliance)
