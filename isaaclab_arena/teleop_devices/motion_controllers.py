# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Teleop device for motion controllers (VR controllers like Quest controllers)."""

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import (
    G1LowerBodyStandingMotionControllerRetargeterCfg,
    G1TriHandUpperBodyMotionControllerGripperRetargeterCfg,
)
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class MotionControllersTeleopDevice(TeleopDeviceBase):
    """
    Teleop device for VR motion controllers (e.g., Quest controllers).
    
    This device uses motion controllers instead of hand tracking for teleoperation.
    It's useful when you have VR controllers but not hand tracking capability.
    Currently supports G1 humanoid robot with gripper control via trigger buttons.
    """

    name = "motion_controllers"

    def __init__(
        self,
        sim_device: str | None = None,
    ):
        """Initialize motion controllers teleop device.
        
        Args:
            sim_device: The simulation device (e.g., "cuda:0").
        """
        super().__init__(sim_device=sim_device)

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        """Get the teleop device configuration.
        
        Args:
            embodiment: The embodiment to use for the teleop device configuration.
            
        Returns:
            DevicesCfg: The device configuration for motion controllers.
        """
        xr_cfg = embodiment.get_xr_cfg()
        xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        
        return DevicesCfg(
            devices={
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyMotionControllerGripperRetargeterCfg(
                            sim_device=self.sim_device,
                        ),
                        G1LowerBodyStandingMotionControllerRetargeterCfg(
                            sim_device=self.sim_device,
                        ),
                    ],
                    sim_device=self.sim_device,
                    xr_cfg=xr_cfg,
                ),
            }
        )
