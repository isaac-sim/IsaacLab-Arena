# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
from PIL import Image


class IsaacLabArenaWriter:
    """Writes per-frame camera data (RGB, depth, intrinsics, extrinsics) to disk.

    Files are saved with the naming pattern::

        {frame_index:04d}.{camera_name}_{data_type}.{ext}

    Args:
        output_dir: Directory where all frame data will be stored.
    """

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

    def write_rgb(self, rgb: torch.Tensor, camera_name: str, frame_index: int) -> None:
        """Save an RGB image as a PNG file.

        Args:
            rgb: (H, W, 3) uint8 tensor.
            camera_name: Camera identifier for the filename.
            frame_index: Frame number.
        """
        path = os.path.join(self._output_dir, f"{frame_index:04d}.{camera_name}_rgb.png")
        Image.fromarray(rgb.cpu().numpy()).save(path)

    def write_depth(self, depth: torch.Tensor, camera_name: str, frame_index: int) -> None:
        """Save a depth map as a float32 ``.npy`` file.

        Args:
            depth: (H, W) float32 tensor (metres, distance-to-image-plane).
            camera_name: Camera identifier for the filename.
            frame_index: Frame number.
        """
        path = os.path.join(self._output_dir, f"{frame_index:04d}.{camera_name}_depth.npy")
        np.save(path, depth.cpu().numpy().astype(np.float32))

    def write_intrinsics(self, intrinsics: torch.Tensor, camera_name: str, frame_index: int) -> None:
        """Save the 3x3 intrinsic matrix as a ``.npy`` file.

        Args:
            intrinsics: (3, 3) float tensor.
            camera_name: Camera identifier for the filename.
            frame_index: Frame number.
        """
        path = os.path.join(self._output_dir, f"{frame_index:04d}.{camera_name}_intrinsics.npy")
        np.save(path, intrinsics.cpu().numpy())

    def write_extrinsics(self, extrinsics: torch.Tensor, camera_name: str, frame_index: int) -> None:
        """Save the 4x4 camera-to-world matrix as a ``.npy`` file.

        Args:
            extrinsics: (4, 4) float tensor.
            camera_name: Camera identifier for the filename.
            frame_index: Frame number.
        """
        path = os.path.join(self._output_dir, f"{frame_index:04d}.{camera_name}_extrinsics.npy")
        np.save(path, extrinsics.cpu().numpy())

    def write_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        camera_name: str,
        frame_index: int,
    ) -> None:
        """Convenience method that writes all four data types for a single frame."""
        self.write_rgb(rgb, camera_name, frame_index)
        self.write_depth(depth, camera_name, frame_index)
        self.write_intrinsics(intrinsics, camera_name, frame_index)
        self.write_extrinsics(extrinsics, camera_name, frame_index)
