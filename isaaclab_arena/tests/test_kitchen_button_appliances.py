# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check kitchen button appliances expose a spring-returned prismatic button on a fixed articulation."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _get_button_stiffness(appliance, joint_prim) -> float:
    """Return the stiffness of the button's linear drive, taking a drive set at spawn over the USD's own."""
    import re

    from pxr import UsdPhysics

    spawn_drives = appliance.object_cfg.spawn.joint_drive_props or {}
    for joint_path_pattern, drives in spawn_drives.items():
        if re.fullmatch(joint_path_pattern, joint_prim.GetPath().pathString):
            return drives[0].stiffness
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
    return drive.GetStiffnessAttr().Get() if drive else 0.0


def _test_kitchen_button_appliances(simulation_app):
    import re

    from pxr import UsdPhysics

    from isaaclab_arena.affordances.pressable import Pressable
    from isaaclab_arena.assets import object_library

    appliance_classes = [
        cls
        for cls in vars(object_library).values()
        if isinstance(cls, type) and issubclass(cls, Pressable) and "kitchen" in getattr(cls, "tags", [])
    ]
    assert len(appliance_classes) == 8
    for appliance_class in appliance_classes:
        appliance = appliance_class()
        assert appliance.object_cfg.spawn.fix_root_link
        # The button returns through its own drive, so no actuator may override its gains.
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
        assert _get_button_stiffness(appliance, joints[0]) > 0.0, f"{appliance_class.__name__} button has no spring"
        assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    return True


def test_kitchen_button_appliances():
    assert run_function_with_persistent_simulation_app(_test_kitchen_button_appliances)
