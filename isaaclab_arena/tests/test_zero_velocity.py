# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_zero_velocity(simulation_app) -> bool:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.physics_config import ZeroVelocityCfg

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    body = UsdGeom.Xform.Define(stage, "/World/table").GetPrim()
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(body)
    rigid_body.CreateVelocityAttr(Gf.Vec3f(-0.003, 0.0006, -0.0003))
    rigid_body.CreateAngularVelocityAttr(Gf.Vec3f(-0.05, -0.22, 0.02))

    ZeroVelocityCfg().apply(body, root)

    assert rigid_body.GetVelocityAttr().Get() == Gf.Vec3f(0.0, 0.0, 0.0), "Linear velocity must be zeroed"
    assert rigid_body.GetAngularVelocityAttr().Get() == Gf.Vec3f(0.0, 0.0, 0.0), "Angular velocity must be zeroed"
    return True


def test_zero_velocity():
    result = run_function_with_persistent_simulation_app(_test_zero_velocity)
    assert result
