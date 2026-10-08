# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare DROID RGB inputs with calibration matching the resized images."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

DEFAULT_CAMERA_KEYS = ("external_camera_rgb", "wrist_camera_rgb")


@dataclass(frozen=True)
class DroidCameraInput:
    """Store one camera image and optional calibration in the robot-root frame."""

    rgb: np.ndarray
    """Copied HWC uint8 RGB image, resized without cropping, padding, or upscaling."""

    intrinsics: np.ndarray | None = None
    """Pinhole matrix scaled to rgb's actual width and height, shaped (3, 3)."""

    T_B_C: np.ndarray | None = None
    """Optical camera C in robot root B, meters and XYZW; C uses +X right, +Y down, +Z forward."""


def enable_droid_camera_pose_updates(camera_cfg: Any, camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS) -> None:
    """Enable per-step calibration before building the environment.

    Args:
        camera_cfg: DROID embodiment's camera_config to update in place.
        camera_keys: RGB observation keys identifying the cameras to configure.
    """
    for key in camera_keys:
        assert key.endswith("_rgb"), "Camera keys must name RGB observations"
        cfg = getattr(camera_cfg, key.removesuffix("_rgb"))
        cfg.update_latest_camera_pose = True
        cfg.update_period = 0.0


def extract_droid_camera_inputs(
    env: Any,
    observation: dict,
    env_id: int,
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
    image_max_edge: int = 384,
    include_calibration: bool = True,
) -> dict[str, DroidCameraInput]:
    """Copy current DROID camera observations with optional robot-root calibration.

    Args:
        env: Arena environment, optionally gym-wrapped; do not step between observation and extraction.
        observation: Current observations with unnormalized uint8 camera_obs terms.
        env_id: Environment whose images and calibration to extract.
        camera_keys: RGB observation keys; external_camera_2_rgb can also be selected.
        image_max_edge: Maximum resized width or height, preserving aspect ratio.
        include_calibration: Require per-step camera pose updates enabled before environment construction.

    Returns:
        Inputs keyed by observation name. Calibration uses camera data from the
        same update as the images. Intrinsics alone do not determine scene depth.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    arena_env = env.unwrapped
    assert 0 <= env_id < arena_env.num_envs, "env_id is out of range"
    assert isinstance(image_max_edge, int) and image_max_edge > 0, "image_max_edge must be a positive integer"
    assert len(set(camera_keys)) == len(camera_keys), "Camera keys must be unique"
    result = {}
    for key in camera_keys:
        assert key.endswith("_rgb"), "Camera keys must name RGB observations"
        images = observation["camera_obs"][key]
        assert images.ndim == 4 and images.shape[0] == arena_env.num_envs, "Expected batched HWC camera images"
        image = images[env_id].detach()
        assert image.dtype == torch.uint8 and image.shape[-1] in (3, 4), "Expected unnormalized uint8 RGB or RGBA"
        source_height, source_width = image.shape[:2]
        ratio = min(1.0, image_max_edge / max(source_height, source_width))
        height, width = max(1, round(source_height * ratio)), max(1, round(source_width * ratio))
        rgb = image[..., :3]
        if (height, width) != (source_height, source_width):
            pixels = rgb.permute(2, 0, 1).unsqueeze(0).float()
            pixels = torch.nn.functional.interpolate(
                pixels, size=(height, width), mode="bilinear", align_corners=False, antialias=True
            )
            rgb = pixels[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        rgb_array = rgb.cpu().numpy().copy()
        intrinsics, T_B_C_array = None, None
        if include_calibration:
            camera = arena_env.scene[key.removesuffix("_rgb")]
            assert camera.cfg.update_latest_camera_pose and camera.cfg.update_period == 0, (
                "Call enable_droid_camera_pose_updates(embodiment.camera_config, camera_keys) before building the"
                " environment"
            )
            data = camera.data
            assert tuple(data.image_shape) == (
                source_height,
                source_width,
            ), "Image dimensions do not match camera calibration"
            intrinsics = data.intrinsic_matrices.torch[env_id].detach().cpu().numpy().copy()
            intrinsics[0, :] *= width / source_width
            intrinsics[1, :] *= height / source_height
            T_W_B = arena_env.scene["robot"].data.root_pose_w.torch[env_id : env_id + 1]
            t_W_C = data.pos_w.torch[env_id : env_id + 1]
            q_W_C = data.quat_w_ros.torch[env_id : env_id + 1]
            t_B_C, q_B_C = subtract_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], t_W_C, q_W_C)
            T_B_C_array = torch.cat((t_B_C, q_B_C), dim=-1)[0].detach().cpu().numpy().copy()
        result[key] = DroidCameraInput(rgb=rgb_array, intrinsics=intrinsics, T_B_C=T_B_C_array)
    return result
