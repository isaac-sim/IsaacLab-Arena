# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID device configuration and gripper conventions at the teleop boundary."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_droid_teleop_device_commands(simulation_app):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab.devices import Se3Keyboard, Se3SpaceMouse

    from isaaclab_arena.assets.registries import DeviceRegistry
    from isaaclab_arena.embodiments.droid.teleop import DroidSe3Keyboard, DroidSe3SpaceMouse

    registry = DeviceRegistry()

    def initialize_buffers(device, cfg):
        device.gripper_term = cfg.gripper_term
        device._sim_device = "cpu"
        device._close_gripper = False
        device._delta_pos = np.array([0.1, -0.2, 0.3])
        device._delta_rot = np.array([0.04, -0.05, 0.06])

    for name, parent_class, droid_class in (
        ("keyboard", Se3Keyboard, DroidSe3Keyboard),
        ("spacemouse", Se3SpaceMouse, DroidSe3SpaceMouse),
    ):
        device = registry.get_device_by_name(name)()
        droid_cfg = registry.get_teleop_device_cfg(device, SimpleNamespace(name="droid_differential_ik"))
        assert droid_cfg.class_type is droid_class
        franka_cfg = registry.get_teleop_device_cfg(device, SimpleNamespace(name="franka_ik"))
        assert franka_cfg.class_type is not droid_class

        # Exercise the real command generation without opening a GUI or HID device.
        with (
            patch.object(parent_class, "__init__", initialize_buffers),
            patch.object(parent_class, "__del__", lambda self: None),
        ):
            controller = droid_cfg.class_type(droid_cfg)
            for close_gripper in (False, True, False):
                controller._close_gripper = close_gripper
                lab_action = parent_class.advance(controller)
                droid_action = controller.advance()
                assert torch.equal(droid_action[:6], lab_action[:6]), "Arm controls must preserve Lab's convention"
                assert droid_action[-1].item() == float(close_gripper), "DROID uses one to close and zero to open"
                assert lab_action[-1].item() == (-1.0 if close_gripper else 1.0)
            controller.gripper_term = False
            assert torch.equal(controller.advance(), parent_class.advance(controller))
            del controller
    return True


def test_droid_teleop_device_commands():
    assert run_function_with_persistent_simulation_app(_test_droid_teleop_device_commands)
