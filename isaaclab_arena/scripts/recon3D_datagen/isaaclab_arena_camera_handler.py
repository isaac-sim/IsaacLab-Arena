# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix


class IsaacLabArenaCameraHandler:
    """Handles a static camera sensor in an IsaacLab Arena environment.

    Wraps an IsaacLab Camera sensor and provides methods to extract
    RGB, depth, intrinsic matrix, and camera-to-world extrinsic matrix.

    Args:
        camera: An isaaclab.sensors.Camera instance.
        camera_name: Identifier used for file naming when writing data.
    """

    def __init__(self, camera: Any, camera_name: str = "static_camera"):
        self._camera = camera
        self._camera_name = camera_name

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def _get_camera_output(self) -> Dict[str, torch.Tensor]:
        from isaaclab.utils import convert_dict_to_backend

        return convert_dict_to_backend(self._camera.data.output, backend="torch")

    def update(self, dt: float) -> None:
        """Read the latest render data from the camera sensor."""
        self._camera.update(dt)

    def get_rgb(self) -> torch.Tensor:
        """Returns RGB image as a (H, W, 3) uint8 tensor."""
        rgb = self._get_camera_output()["rgb"].clone()
        assert rgb.shape[0] == 1, "Expected single environment"
        assert rgb.shape[-1] == 3, "Expected 3-channel RGB"
        return rgb.squeeze(0)

    def get_depth(self) -> torch.Tensor:
        """Returns depth map as a (H, W) float32 tensor (distance to image plane)."""
        depth = self._get_camera_output()["distance_to_image_plane"].clone()
        assert depth.shape[0] == 1, "Expected single environment"
        return depth.squeeze()

    def get_intrinsics(self) -> torch.Tensor:
        """Returns the 3x3 camera intrinsic matrix."""
        assert self._camera.data.intrinsic_matrices.shape[0] == 1, "Expected single environment"
        return self._camera.data.intrinsic_matrices.data[0].clone()

    def get_extrinsics(self) -> torch.Tensor:
        """Returns the 4x4 camera-to-world homogeneous transformation matrix."""
        assert self._camera.data.pos_w.shape[0] == 1, "Expected single environment"
        translation = self._camera.data.pos_w.data[0].clone()
        rotation_quat = self._camera.data.quat_w_ros.data[0].clone()

        T = torch.eye(4, dtype=torch.float32, device=rotation_quat.device)
        quat_flat = rotation_quat.reshape(4)
        T[:3, :3] = matrix_from_quat(quat_flat)
        T[:3, 3] = translation
        return T


def create_static_camera(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float],
    width: int = 640,
    height: int = 480,
    focal_length: float = 24.0,
    prim_path: str = "/World/StaticCamera",
) -> IsaacLabArenaCameraHandler:
    """Create and initialise a static camera sensor looking from *position* at *target*.

    Must be called **after** the simulation app has been launched and the
    environment has been reset so that the USD stage is ready.

    Args:
        position: Camera position in world frame (x, y, z).
        target: Look-at point in world frame (x, y, z).
        width: Image width in pixels.
        height: Image height in pixels.
        focal_length: Focal length in mm.
        prim_path: USD prim path where the camera will be spawned.

    Returns:
        An initialised :class:`IsaacLabArenaCameraHandler`.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    quat_wxyz = _look_at_quaternion_ros(position, target)

    cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=quat_wxyz,
            convention="ros",
        ),
    )

    camera = Camera(cfg)
    # The Camera only gets _is_initialized when the timeline PLAY callback runs. We create this
    # camera after env.reset(), so PLAY already fired. Trigger initialization manually.
    camera._initialize_callback(None)
    camera.reset([0])
    return IsaacLabArenaCameraHandler(camera)


def _look_at_quaternion_ros(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[float, float, float, float]:
    """Compute a quaternion (w, x, y, z) for a ROS-convention camera at *eye* looking at *target*.

    ROS camera convention: x = right, y = down, z = forward (optical axis).
    """
    eye_arr = np.array(eye, dtype=np.float64)
    target_arr = np.array(target, dtype=np.float64)
    up_arr = np.array(up, dtype=np.float64)

    forward = target_arr - eye_arr
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_arr)
    right /= np.linalg.norm(right)

    # In ROS convention y points down; recompute to ensure orthogonality
    down = np.cross(forward, right)

    # Columns are camera axes expressed in world frame
    R_cam_to_world = np.column_stack([right, down, forward])
    quat_tensor = quat_from_matrix(torch.from_numpy(R_cam_to_world).float())
    return tuple(quat_tensor.tolist())
