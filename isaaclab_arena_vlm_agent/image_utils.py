# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image encoding and camera calibration utilities."""

import base64
import io
import numpy as np

from PIL import Image


def extract_rgb_images_from_camera_observations(
    camera_observations: dict, camera_keys: tuple[str, ...], env_id: int
) -> dict[str, np.ndarray]:
    """Extract named camera images for one environment as HWC numpy arrays.

    Args:
        camera_observations: Camera tensors shaped (num_envs, height, width, channels).
        camera_keys: Camera names in request order.
        env_id: Environment index within each camera batch.

    Returns:
        Images indexed by camera name, in the requested order.
    """
    assert camera_keys, "Configure at least one camera observation key"
    images = {}
    for key in camera_keys:
        batch = camera_observations[key]
        assert batch.ndim == 4, "Expected batched HWC images"
        images[key] = batch[env_id].detach().cpu().numpy()
    return images


def image_to_jpeg_data_url(image: np.ndarray, max_width_or_height: int) -> tuple[str, tuple[int, int]]:
    """Encode an RGB or RGBA image as JPEG, preserving aspect ratio without upscaling.

    Args:
        image: HWC uint8 pixels; the alpha channel, if present, is discarded.
        max_width_or_height: Maximum encoded image width or height, in pixels.

    Returns:
        A JPEG data URL and its actual (width, height) in pixels.
    """
    assert max_width_or_height > 0, "max_width_or_height must be positive"
    assert image.ndim == 3 and image.shape[-1] in (3, 4), f"Unexpected image shape {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8 pixels, got {image.dtype}"
    frame = Image.fromarray(image[..., :3])
    frame.thumbnail((max_width_or_height, max_width_or_height))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), frame.size


def resize_camera_geometry(geometry: dict, original_size: tuple[int, int], resized_size: tuple[int, int]) -> None:
    """Update camera intrinsics and image dimensions in place after resizing.

    Args:
        geometry: Camera calibration containing an intrinsic_matrix and image_size_px.
        original_size: Original image (width, height) in pixels.
        resized_size: Actual resized image (width, height) in pixels.
    """
    intrinsics = np.array(geometry["intrinsic_matrix"], dtype=float)
    intrinsics[0] *= resized_size[0] / original_size[0]
    intrinsics[1] *= resized_size[1] / original_size[1]
    geometry["intrinsic_matrix"] = intrinsics.tolist()
    geometry["image_size_px"] = list(resized_size)
