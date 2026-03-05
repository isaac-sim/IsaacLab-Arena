# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

ALL_DATA_TYPES = [
    "rgb",
    "distance_to_image_plane",
    "normals",
    "motion_vectors",
    "semantic_segmentation",
]


class IsaacLabArenaCameraHandler:
    """Handles a static camera sensor in an IsaacLab Arena environment.

    Wraps an IsaacLab Camera sensor and provides methods to extract
    RGB, depth, normals, optical flow, semantic segmentation,
    intrinsic matrix, and camera-to-world extrinsic matrix.

    Args:
        camera: An isaaclab.sensors.Camera instance.
        camera_name: Identifier used for file naming when writing data.
    """

    def __init__(self, camera: Any, camera_name: str = "static_camera"):
        self._camera = camera
        self._camera_name = camera_name
        # Persistent RGBA → object_id mapping for temporal consistency across frames
        self._color_to_object_id: Dict[tuple, int] = {}
        self._next_object_id: int = 0

    @property
    def camera_name(self) -> str:
        return self._camera_name

    def _get_camera_output(self) -> Dict[str, torch.Tensor]:
        from isaaclab.utils import convert_dict_to_backend

        return convert_dict_to_backend(self._camera.data.output, backend="torch")

    def update(self, dt: float) -> None:
        """Read the latest render data from the camera sensor."""
        self._camera.update(dt)

    # ------------------------------------------------------------------
    # Core outputs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def get_normals(self) -> torch.Tensor:
        """Returns surface normals as a (H, W, 3) float32 tensor (x, y, z)."""
        normals = self._get_camera_output()["normals"].clone()
        assert normals.shape[0] == 1, "Expected single environment"
        assert normals.shape[-1] == 3, "Expected 3-channel normals"
        return normals.squeeze(0)

    # ------------------------------------------------------------------
    # Optical flow (motion vectors)
    # ------------------------------------------------------------------

    def get_optical_flow(self) -> torch.Tensor:
        """Returns dense optical flow as a (H, W, 2) float32 tensor (dx, dy in pixels)."""
        mv = self._get_camera_output()["motion_vectors"].clone()
        assert mv.shape[0] == 1, "Expected single environment"
        assert mv.shape[-1] == 2, "Expected 2-channel motion vectors"
        return mv.squeeze(0)

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    def get_semantic_segmentation(self) -> Tuple[torch.Tensor, Dict]:
        """Returns semantic segmentation data.

        Returns:
            Tuple of:
                - (H, W, 4) uint8 tensor with RGBA segmentation colors.
                - dict mapping RGBA id-strings to label dicts (``{"class": ...}``).
        """
        seg = self._get_camera_output()["semantic_segmentation"].clone()
        id_to_labels = self._camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        assert seg.shape[0] == 1, "Expected single environment"
        assert seg.shape[-1] == 4, "Expected 4-channel segmentation (RGBA)"
        return seg.squeeze(0).to(torch.uint8), id_to_labels

    # ------------------------------------------------------------------
    # Per-object semantic info
    # ------------------------------------------------------------------

    def _get_object_id(self, rgba: tuple) -> int:
        """Return a stable integer object ID for a given RGBA colour.

        The same colour always maps to the same ID across frames, so objects
        can be tracked temporally even when no semantic label is available.
        """
        if rgba not in self._color_to_object_id:
            self._color_to_object_id[rgba] = self._next_object_id
            self._next_object_id += 1
        return self._color_to_object_id[rgba]

    def get_semantic_info(self) -> List[Dict[str, Any]]:
        """Returns per-object metadata for every semantic class visible in this frame.

        For each distinct RGBA colour present in the segmentation image, reports:
        - ``object_id``: temporally consistent integer ID (same object → same ID
          across frames).
        - ``object_name``: stable name, e.g. ``"obj_0"``, ``"obj_1"``, ...
          If the USD prim has a ``semanticLabel`` the label is appended:
          ``"obj_0_cracker_box"``.
        - ``rgba``: the RGBA colour tuple.
        - ``class_name``: raw semantic class from USD (may be ``""`` or
          ``"BACKGROUND"``).
        - ``pixel_count``: number of pixels belonging to this object.

        Returns:
            List of dicts, one per visible object, sorted by descending pixel count.
        """
        seg_data, id_to_labels = self.get_semantic_segmentation()

        rgba_to_class: Dict[tuple, str] = {}
        for key, val in id_to_labels.items():
            rgba_tuple = tuple(eval(key)) if isinstance(key, str) else key
            rgba_to_class[rgba_tuple] = val.get("class", "")

        seg_np = seg_data.cpu().numpy()
        flat_seg = seg_np.reshape(-1, 4)
        unique_colors = np.unique(flat_seg, axis=0)

        results: List[Dict[str, Any]] = []
        for color in unique_colors:
            color_tuple = tuple(int(c) for c in color)
            mask = (seg_np == color.reshape(1, 1, 4)).all(axis=-1)
            pixel_count = int(mask.sum())
            if pixel_count == 0:
                continue

            object_id = self._get_object_id(color_tuple)
            class_name = rgba_to_class.get(color_tuple, "")

            # Build a human-readable, temporally consistent name
            if class_name and class_name.upper() not in ("UNKNOWN", "UNLABELLED", "BACKGROUND"):
                object_name = f"obj_{object_id}_{class_name}"
            else:
                object_name = f"obj_{object_id}"

            results.append({
                "object_id": object_id,
                "object_name": object_name,
                "rgba": color_tuple,
                "class_name": class_name,
                "pixel_count": pixel_count,
            })

        results.sort(key=lambda x: x["pixel_count"], reverse=True)
        return results


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


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

    The camera is configured with all sensor data types needed for 3D
    reconstruction data generation: RGB, depth, normals, motion vectors
    (optical flow), and semantic segmentation.

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
        data_types=ALL_DATA_TYPES,
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

    down = np.cross(forward, right)

    R_cam_to_world = np.column_stack([right, down, forward])
    quat_tensor = quat_from_matrix(torch.from_numpy(R_cam_to_world).float())
    return tuple(quat_tensor.tolist())
