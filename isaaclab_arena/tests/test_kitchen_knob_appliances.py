# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check kitchen knob appliances configure a passive dial and expose a dial joint covering their levels."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_kitchen_knob_appliances(simulation_app):
    from pxr import UsdPhysics

    from isaaclab_arena.affordances.turnable import Turnable
    from isaaclab_arena.assets import object_library
    from isaaclab_arena.assets.object_library import KitchenKnobAppliance

    appliance_classes = [
        cls
        for cls in vars(object_library).values()
        if isinstance(cls, type) and issubclass(cls, KitchenKnobAppliance) and cls is not KitchenKnobAppliance
    ]
    assert len(appliance_classes) == 8
    for appliance_class in appliance_classes:
        appliance = appliance_class()
        assert isinstance(appliance, Turnable)
        assert appliance.object_cfg.spawn.fix_root_link == appliance.spawn_cfg_addon["fix_root_link"]
        knob = appliance.object_cfg.actuators["knob"]
        assert knob.joint_names_expr == [appliance.turnable_joint_name]
        assert knob.stiffness == 0.0 and knob.damping == 0.0
        assert knob.armature == appliance.knob_armature
        assert knob.friction == knob.dynamic_friction == appliance.knob_friction

        stage = appliance.open_spawned_usd_stage()
        joints = [p for p in stage.Traverse() if p.GetName() == appliance.turnable_joint_name]
        assert len(joints) == 1, f"{appliance_class.__name__} needs one {appliance.turnable_joint_name} joint"
        joint = UsdPhysics.RevoluteJoint(joints[0])
        assert joint, f"{appliance_class.__name__} dial joint must be revolute"
        assert joint.GetJointEnabledAttr().Get()
        lower_limit_deg = joint.GetLowerLimitAttr().Get()
        upper_limit_deg = joint.GetUpperLimitAttr().Get()
        # Unset limits are unbounded; the dial must reach both the zero reset angle and the last level.
        assert lower_limit_deg is None or lower_limit_deg <= 1e-3, f"{appliance_class.__name__} lower limit"
        assert (
            upper_limit_deg is None or upper_limit_deg >= appliance.max_level_angle_deg - 1e-3
        ), f"{appliance_class.__name__} upper limit"
        assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert "/.*/door_joint" in object_library.LightwheelToasterOven001().object_cfg.spawn.joint_drive_props
    return True


def test_kitchen_knob_appliances():
    assert run_function_with_persistent_simulation_app(_test_kitchen_knob_appliances)
