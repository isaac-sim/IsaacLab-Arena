# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare RGB inputs with calibration in a caller-selected reference frame."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraInput:
    """Store one camera image and optional calibration in a reference frame."""

    rgb: np.ndarray
    """Copied HWC uint8 RGB image, resized without cropping, padding, or upscaling."""

    intrinsics: np.ndarray | None = None
    """Pinhole matrix scaled to rgb's actual width and height, shaped (3, 3)."""

    T_R_C: np.ndarray | None = None
    """Optical camera C in reference R, meters and XYZW; C uses +X right, +Y down, +Z forward."""


def enable_camera_pose_updates(camera_cfg: Any, camera_keys: tuple[str, ...]) -> None:
    """Enable per-step calibration before building the environment.

    Args:
        camera_cfg: Arena camera configuration to update in place.
        camera_keys: RGB observation keys identifying the cameras to configure.
    """
    for key in camera_keys:
        assert key.endswith("_rgb"), "Camera keys must name RGB observations"
        cfg = getattr(camera_cfg, key.removesuffix("_rgb"))
        cfg.update_latest_camera_pose = True
        cfg.update_period = 0.0


def extract_camera_inputs(
    env: Any,
    observation: dict,
    env_id: int,
    camera_keys: tuple[str, ...],
    *,
    T_W_R: torch.Tensor | None = None,
    image_max_edge: int = 384,
    include_calibration: bool = True,
) -> dict[str, CameraInput]:
    """Copy current camera observations with optional reference-frame calibration.

    Args:
        env: Arena environment, optionally gym-wrapped; do not step between observation and extraction.
        observation: Current observations with unnormalized uint8 camera_obs terms.
        env_id: Environment whose images and calibration to extract.
        camera_keys: RGB observation keys of the form <scene camera name>_rgb.
        T_W_R: Reference R in simulation world W for each environment, shaped (N, 7),
            meters and unit XYZW quaternions. Required with calibration. Pass a robot
            root pose for robot-relative calibration, or identity poses for world-relative
            calibration. Must correspond to the same simulation step as observation.
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
    if include_calibration:
        assert T_W_R is not None, "T_W_R is required for calibration"
        assert T_W_R.shape == (arena_env.num_envs, 7), "Expected one reference pose per environment"
        assert torch.isfinite(T_W_R).all(), "Reference poses must be finite"
        quaternion_norm = torch.linalg.vector_norm(T_W_R[:, 3:], dim=-1)
        assert torch.allclose(
            quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-4
        ), "Expected unit XYZW reference quaternions"
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
        intrinsics, T_R_C_array = None, None
        if include_calibration:
            camera = arena_env.scene[key.removesuffix("_rgb")]
            assert (
                camera.cfg.update_latest_camera_pose and camera.cfg.update_period == 0
            ), "Call enable_camera_pose_updates(camera_cfg, camera_keys) before building the environment"
            data = camera.data
            assert tuple(data.image_shape) == (
                source_height,
                source_width,
            ), "Image dimensions do not match camera calibration"
            intrinsics = data.intrinsic_matrices.torch[env_id].detach().cpu().numpy().copy()
            intrinsics[0, :] *= width / source_width
            intrinsics[1, :] *= height / source_height
            t_W_C = data.pos_w.torch[env_id : env_id + 1]
            q_W_C = data.quat_w_ros.torch[env_id : env_id + 1]
            reference_pose = T_W_R[env_id : env_id + 1].to(device=t_W_C.device, dtype=t_W_C.dtype)
            t_R_C, q_R_C = subtract_frame_transforms(reference_pose[:, :3], reference_pose[:, 3:], t_W_C, q_W_C)
            T_R_C_array = torch.cat((t_R_C, q_R_C), dim=-1)[0].detach().cpu().numpy().copy()
        result[key] = CameraInput(rgb=rgb_array, intrinsics=intrinsics, T_R_C=T_R_C_array)
    return result
