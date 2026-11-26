# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check kitchen button appliances expose a spring-returned prismatic button on a fixed articulation."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_kitchen_button_appliances(simulation_app):
    import re

    from pxr import UsdPhysics

    from isaaclab_arena.affordances.pressable import Pressable
    from isaaclab_arena.assets import object_library
    from isaaclab_arena.assets.object_library import KitchenButtonAppliance

    appliance_classes = [
        cls
        for cls in vars(object_library).values()
        if isinstance(cls, type) and issubclass(cls, KitchenButtonAppliance) and cls is not KitchenButtonAppliance
    ]
    assert len(appliance_classes) == 8
    for appliance_class in appliance_classes:
        appliance = appliance_class()
        assert isinstance(appliance, Pressable)
        assert appliance.object_cfg.spawn.fix_root_link
        # The button returns through its authored USD drive, so no actuator may override its gains.
        for actuator in appliance.object_cfg.actuators.values():
            for joint_expr in actuator.joint_names_expr:
                assert not re.fullmatch(joint_expr, appliance.pressable_joint_name)

        stage = appliance.open_spawned_usd_stage()
        joints = [p for p in stage.Traverse() if p.GetName() == appliance.pressable_joint_name]
        assert len(joints) == 1, f"{appliance_class.__name__} needs one {appliance.pressable_joint_name} joint"
        joint = UsdPhysics.PrismaticJoint(joints[0])
        assert joint, f"{appliance_class.__name__} button joint must be prismatic"
        assert joint.GetJointEnabledAttr().Get()
        assert joint.GetLowerLimitAttr().Get() < joint.GetUpperLimitAttr().Get()
        # The Palatial espresso machine authors its button spring under the angular drive instance.
        assert joints[0].HasAPI(UsdPhysics.DriveAPI, "linear") or joints[0].HasAPI(UsdPhysics.DriveAPI, "angular")
        assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    return True


def test_kitchen_button_appliances():
    assert run_function_with_persistent_simulation_app(_test_kitchen_button_appliances)
