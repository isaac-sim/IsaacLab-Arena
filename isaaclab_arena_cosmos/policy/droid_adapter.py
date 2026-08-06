# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any

from openpi_client import image_tools

from isaaclab_arena.policy.remote_policy_base import EmbodimentAdapter


@dataclass(frozen=True)
class DroidObservation:
    """Per-env images (already padded/resized to the tile size) and proprio for a request."""

    wrist_image: np.ndarray  # (IMAGE_H, IMAGE_W, 3) uint8
    left_image: np.ndarray  # (IMAGE_H, IMAGE_W, 3) uint8, left / exterior_1 (over-shoulder-left)
    right_image: np.ndarray  # (IMAGE_H, IMAGE_W, 3) uint8, right / exterior_2 (over-shoulder-right)
    joint_position: np.ndarray  # (7,) float32
    gripper_position: np.ndarray  # (1,) float32


class CosmosDroidAdapter(EmbodimentAdapter):
    """Wire format for the released Cosmos DROID policies (Nano/Edge), joint_pos action space."""

    # Fixed by the released DROID joint_pos policies: 7 arm joints + 1 gripper command.
    action_dim = 8

    # Per-view tile size matches Robolab client.
    # The intention is when the images are combined, we end up with a 540x640 image.
    # Reference to robolab client: https://github.com/NVlabs/RoboLab/blob/main/policies/cosmos3/client.py
    IMAGE_H = 360
    IMAGE_W = 640

    # Arena DROID camera keys (see isaaclab_arena/embodiments/droid/droid.py). external_camera is
    # the over-shoulder-left view and external_camera_2 the over-shoulder-right view.
    arena_wrist_camera_key = "wrist_camera_rgb"
    arena_exterior_camera_1_key = "external_camera_rgb"
    arena_exterior_camera_2_key = "external_camera_2_rgb"

    # Top-level keys on the arena gym observation dict this adapter consumes; both are standard
    # Arena conventions (camera group from make_camera_observation_cfg, and the embodiment's
    # `policy` PolicyCfg group).
    arena_camera_obs_group = "camera_obs"
    arena_policy_obs_group = "policy"

    def extract(self, observation: dict[str, Any], env_id: int) -> DroidObservation:
        cam = observation[self.arena_camera_obs_group]
        proprio = observation[self.arena_policy_obs_group]
        return DroidObservation(
            wrist_image=self._resize_with_pad(cam[self.arena_wrist_camera_key][env_id]),
            left_image=self._resize_with_pad(cam[self.arena_exterior_camera_1_key][env_id]),
            right_image=self._resize_with_pad(cam[self.arena_exterior_camera_2_key][env_id]),
            joint_position=proprio["joint_pos"][env_id].detach().cpu().numpy(),
            gripper_position=proprio["gripper_pos"][env_id].detach().cpu().numpy(),
        )

    def pack_request(self, extracted: DroidObservation, language_instruction: str) -> dict[str, Any]:
        return {
            "observation/image": _compose_concat_view(
                extracted.wrist_image, extracted.left_image, extracted.right_image
            ),
            "observation/joint_position": extracted.joint_position,
            "observation/gripper_position": extracted.gripper_position,
            "prompt": language_instruction,
        }

    def _resize_with_pad(self, image: Any) -> np.ndarray:
        """Aspect-preserving resize + pad of an Arena camera tensor to (IMAGE_H, IMAGE_W)."""
        array = np.ascontiguousarray(image.detach().cpu().numpy().astype(np.uint8, copy=False))
        return image_tools.resize_with_pad(array, self.IMAGE_H, self.IMAGE_W)


def _compose_concat_view(wrist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Build the concat_view: wrist on top, the two exterior views half-size side-by-side below.

    Matches RoboLab ``policies/cosmos3/client.py``: the exterior views are downsized to half the
    tile size and concatenated, then stacked under the full-size wrist view.
    """
    half = (wrist.shape[0] // 2, wrist.shape[1] // 2)
    left_small = _resize_bilinear(left, half)
    right_small = _resize_bilinear(right, half)
    return np.concatenate((wrist, np.concatenate((left_small, right_small), axis=1)))


def _resize_bilinear(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize an (H, W, 3) uint8 image to ``size``, preserving dtype."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=size, mode="bilinear")
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)
