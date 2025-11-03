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

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class HandTrackingTeleopDevice(TeleopDeviceBase):
    """
    Teleop device for hand tracking.
    """

    name = "avp_handtracking"

    def __init__(
        self, sim_device: str | None = None, num_open_xr_hand_joints: int = 52, enable_visualization: bool = True
    ):
        super().__init__(sim_device=sim_device)
        self.num_open_xr_hand_joints = num_open_xr_hand_joints
        self.enable_visualization = enable_visualization

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return DevicesCfg(
            devices={
                "avp_handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=self.enable_visualization,
                            # number of joints in both hands
                            num_open_xr_hand_joints=self.num_open_xr_hand_joints,
                            sim_device=self.sim_device,
                            hand_joint_names=embodiment.get_action_cfg().upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim_device,
                    xr_cfg=embodiment.get_xr_cfg(),
                ),
            }
        )
