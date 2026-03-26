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
from isaaclab.devices.keyboard import Se3KeyboardCfg

from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg

import numpy as np
import torch
from dataclasses import dataclass
from typing import Final

import isaaclab.sim as sim_utils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


import isaaclab.utils.math as PoseUtils


# Register FiiRetargeter in the factory's RETARGETER_MAP
def _register_fii_retargeter():
    """Register FiiRetargeter with the teleop device factory."""
    try:
        from isaaclab.devices.teleop_device_factory import RETARGETER_MAP
        # Forward declare to avoid circular import
        RETARGETER_MAP[type('FiiRetargeterCfg', (), {})] = None  # Placeholder
    except ImportError:
        pass  # Factory might not be loaded yet


@register_device
class HandTrackingTeleopDevice(TeleopDeviceBase):
    """
    Teleop device for hand tracking.
    """

    name = "handtracking"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)
        self.xr = XrCfg(anchor_pos=(0., 0., 0.25), anchor_rot=(1.0, 0.0, 0.0, 0.0))

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        FiiRetargeterCfg(
                            sim_device=self.sim_device,
                            num_open_xr_hand_joints=2 * 26,
                            enable_visualization=True
                        )
                    ],
                    sim_device=self.sim_device,
                    xr_cfg=self.xr,
                ),
            }
        )

@dataclass
class FiiRetargeterCfg(RetargeterCfg):
    enable_visualization: bool = True
    num_open_xr_hand_joints: int = 100


class FiiRetargeter(RetargeterBase):

    def __init__(self, cfg: FiiRetargeterCfg):
        """Initialize the retargeter."""
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints

        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/g1_hand_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:

        base_vel = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self._sim_device)
        base_height = torch.tensor([0.7], dtype=torch.float32, device=self._sim_device)

        left_eef_pose = torch.tensor([-0.3, 0.3, 0.72648, 1.0, 0., 0., 0.], dtype=torch.float32, device=self._sim_device)
        right_eef_pose = torch.tensor([-0.3, 0.3, 0.72648, 1.0, 0., 0., 0.], dtype=torch.float32, device=self._sim_device)


        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]
        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))

        if left_wrist is not None:
            left_eef_pose = torch.tensor(
                self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self._sim_device
            )
        if right_wrist is not None:
            right_eef_pose = torch.tensor(
                self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self._sim_device
            )

        # left_wrist_tensor = torch.tensor(
        #     self._retarget_abs(left_wrist, is_left=True), dtype=torch.float32, device=self._sim_device
        # )
        # right_wrist_tensor = torch.tensor(
        #     self._retarget_abs(right_wrist, is_left=False), dtype=torch.float32, device=self._sim_device
        # )

        gripper_value_left = self._hand_data_to_gripper_values(data[OpenXRDevice.TrackingTarget.HAND_LEFT])
        gripper_value_right = self._hand_data_to_gripper_values(data[OpenXRDevice.TrackingTarget.HAND_RIGHT])

        return torch.cat([left_eef_pose, right_eef_pose, gripper_value_left, gripper_value_right, base_vel, base_height])
    
    def _hand_data_to_gripper_values(self, hand_data):
        thumb_tip = hand_data["thumb_tip"]
        index_tip = hand_data["index_tip"]

        distance = np.linalg.norm(thumb_tip[:3] - index_tip[:3])

        finger_dist_closed = 0.00
        finger_dist_open = 0.06

        gripper_value_closed = 0.06
        gripper_value_open = 0.00

        t = np.clip((distance - finger_dist_closed) / (finger_dist_open - finger_dist_closed), 0, 1)
        # t = 1 -> open
        # t = 0 -> closed
        gripper_joint_value = (1.0 - t) * gripper_value_closed + t * gripper_value_open

        return torch.tensor([gripper_joint_value, gripper_joint_value], dtype=torch.float32, device=self._sim_device)

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            wrist: Wrist pose data from OpenXR.
            is_left: True for the left hand, False for the right hand.

        Returns:
            Retargeted wrist pose in USD control frame.
        """
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)


        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))

        if is_left:
            # Corresponds to a rotation of (0, 90, 90) in euler angles (x,y,z)
            combined_quat = torch.tensor([0, 0.7071, 0, 0.7071], dtype=torch.float32)
            # combined_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
            # combined_quat = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
            # combined_quat = torch.tensor([0., 1., 0., 0.], dtype=torch.float32)
            # combined_quat = torch.tensor([0, -0.7071, 0, 0.7071], dtype=torch.float32)
        else:
            # Corresponds to a rotation of (0, -90, -90) in euler angles (x,y,z)
            combined_quat = torch.tensor([0, -0.7071, 0, 0.7071], dtype=torch.float32)
            # combined_quat = torch.tensor([0, 0, 0, -1], dtype=torch.float32)
            # combined_quat = torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=torch.float32)
            # combined_quat = torch.tensor([0., 1., 0., 0.], dtype=torch.float32)
            # combined_quat = torch.tensor([0, 0.7071, 0, 0.7071], dtype=torch.float32)

        offset = torch.tensor([0.0, 0.25, -0.15])
        transform_pose = PoseUtils.make_pose(offset, PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])


# Register the FII retargeter with the factory after class definition
try:
    from isaaclab.devices.teleop_device_factory import RETARGETER_MAP
    RETARGETER_MAP[FiiRetargeterCfg] = FiiRetargeter
except ImportError:
    pass  # Factory not available yet