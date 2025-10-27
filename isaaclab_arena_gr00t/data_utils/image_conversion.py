# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch

import cv2


def resize_frames_with_padding(
    frames: torch.Tensor | np.ndarray, target_image_size: tuple, bgr_conversion: bool = False, pad_img: bool = True
) -> np.ndarray:
    """Process batch of frames with padding and resizing vectorized
    Args:
        frames: np.ndarray of shape [N, 256, 160, 3]
        target_image_size: target size (height, width)
        bgr_conversion: whether to convert BGR to RGB
        pad_img: whether to resize images
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    elif not isinstance(frames, np.ndarray):
        raise ValueError(f"Invalid frame type: {type(frames)}")

    if bgr_conversion:
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    if pad_img:
        top_padding = (frames.shape[2] - frames.shape[1]) // 2
        bottom_padding = top_padding

        # Add padding to all frames at once
        frames = np.pad(
            frames,
            pad_width=((0, 0), (top_padding, bottom_padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # Resize all frames at once
    # frames.shape is (N, height, width, channels)
    # target_image_size is (height, width)
    # cv2.resize expects (width, height)
    if frames.shape[1:3] != target_image_size:
        target_size_cv2 = (target_image_size[1], target_image_size[0])  # Convert to (width, height)
        frames = np.stack([cv2.resize(f, target_size_cv2) for f in frames])

    return frames
