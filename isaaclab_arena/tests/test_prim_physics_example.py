# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the documented custom physics config with the library's RedCube asset."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_red_cube_physics_example(_simulation_app):
    from isaaclab.sim.utils import create_new_stage
    from pxr import UsdGeom

    stage = create_new_stage()

    # [start-red-cube-physics-example]
    import math

    from isaaclab.utils.configclass import configclass
    from pxr import UsdPhysics, UsdShade

    from isaaclab_arena.assets.object_library import RedCube
    from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg

    @configclass
    class ColliderFrictionCfg(UsdPrimSpawnPhysicsCfg):
        """Bind an instance-local contact material to a selected collider."""

        friction: float = 0.8
        """Static and dynamic friction coefficient."""

        def validate_target(self, prim, root):
            """Require a collider and a finite nonnegative friction coefficient."""
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert math.isfinite(self.friction) and self.friction >= 0
            assert not prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild("ContactMaterial"))

        def apply(self, prim, root):
            """Author and bind a material within the spawned asset."""
            # A local material preserves shared source materials and remaps during cloning.
            material = UsdShade.Material.Define(prim.GetStage(), prim.GetPath().AppendChild("ContactMaterial"))
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(self.friction)
            physics.CreateDynamicFrictionAttr(self.friction)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

    class HighFrictionRedCube(RedCube):
        spawn_cfg_addon = {
            "prim_physics": {"Cube": ColliderFrictionCfg(friction=0.8)},
        }

    red_cube = HighFrictionRedCube()
    # [end-red-cube-physics-example]

    # Spawn the actual library USD through Object's addon integration, including cloning.
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    spawn = red_cube.object_cfg.spawn
    spawn.func("/World/env_.*/RedCube", spawn)
    for index in range(2):
        root_path = f"/World/env_{index}/RedCube"
        collider = stage.GetPrimAtPath(f"{root_path}/Cube")
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        assert str(material.GetPath()) == f"{root_path}/Cube/ContactMaterial"
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.8)
        assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.8)

    # A plain library cube must still use its original material after the tuned cubes spawn.
    plain_spawn = RedCube().object_cfg.spawn
    plain_root = plain_spawn.func("/World/PlainCube", plain_spawn)
    assert not plain_root.GetChild("Cube").GetChild("ContactMaterial")
    assert spawn.usd_path == plain_spawn.usd_path
    assert spawn.scale == plain_spawn.scale

    # Instance tuning cannot leak into the example's class defaults or another object.
    spawn.prim_physics["Cube"].friction = 0.4
    red_cube.spawn_cfg_addon["prim_physics"]["Cube"].friction = 0.2
    assert HighFrictionRedCube.spawn_cfg_addon["prim_physics"]["Cube"].friction == 0.8
    assert HighFrictionRedCube().spawn_cfg_addon["prim_physics"]["Cube"].friction == 0.8
    assert HighFrictionRedCube().object_cfg.spawn.prim_physics["Cube"].friction == 0.8
    return True


def test_red_cube_physics_example():
    assert run_function_with_persistent_simulation_app(_test_red_cube_physics_example)
