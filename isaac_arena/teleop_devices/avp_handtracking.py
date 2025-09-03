#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GR1T2RetargeterCfg

from isaac_arena.assets.register import register_device
from isaac_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class HandTrackingTeleopDevice(TeleopDeviceBase):
    """
    Teleop device for hand tracking.
    """

    NUM_OPENXR_HAND_JOINTS = 26

    name = "avp_handtracking"

    def __init__(self):
        super().__init__()

    def build_cfg(self, *, sim_device: str | None = None, actions: object | None = None, xr_cfg: object | None = None):
        return DevicesCfg(
            devices={
                "avp_handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            # number of joints in both hands
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=sim_device,
                            hand_joint_names=actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=sim_device,
                    xr_cfg=xr_cfg,
                ),
            }
        )
