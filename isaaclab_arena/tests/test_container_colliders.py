# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_containers_collide_with_boxes_only(_simulation_app):
    from isaaclab.sim.utils import create_new_stage
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.assets.object_library import GreenContainer, RedContainer

    create_new_stage()
    for container in (GreenContainer(), RedContainer()):
        spawn = container.object_cfg.spawn
        root = spawn.func(f"/World/{container.name}", spawn)
        colliders = {
            prim.GetName(): UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        }
        assert colliders == {
            "Container_H20_01": False,
            "Cube": True,
            "Cube_01": True,
            "Cube_02": True,
            "Cube_03": True,
            "Cube_04": True,
        }, f"{container.name}: {colliders}"
    return True


def test_containers_collide_with_boxes_only():
    assert run_function_with_persistent_simulation_app(_test_containers_collide_with_boxes_only)
