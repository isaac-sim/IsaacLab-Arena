# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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

from isaaclab_arena_cosmos3.policy.cosmos3_remote_policy import Cosmos3EmbodimentAdapter


@dataclass(frozen=True)
class Cosmos3DroidObservation:
    """Per-env tensors needed to assemble a cosmos3 DROID request."""

    left_image: np.ndarray  # (H, W, 3) uint8 — over-shoulder left / external_camera
    right_image: np.ndarray  # (H, W, 3) uint8 — over-shoulder right / external_camera_2
    wrist_image: np.ndarray  # (H, W, 3) uint8 — wrist camera
    joint_position: np.ndarray  # (7,) float32
    gripper_position: np.ndarray  # (1,) float32


class Cosmos3DroidAdapter(Cosmos3EmbodimentAdapter):
    """Wire format for upstream cosmos3 DROID checkpoint.

    Cosmos3 composes three camera views into a single 720×640 image
    and sends it under the ``observation/image`` key (unlike pi0 which
    sends separate per-camera keys).
    """

    # Fixed by upstream cosmos3 droid checkpoint: 7 panda joints + 1 gripper command.
    action_dim = 8

    # Cosmos3 predicts 32 action steps per inference call.
    open_loop_horizon = 32

    # Image dimensions for cosmos3's composed input.
    image_w = 640
    image_h = 360

    # Arena observation keys — follow the same convention as Pi0DroidAdapter:
    #   camera_obs group  — from isaaclab_arena.utils.cameras.make_camera_observation_cfg
    #   policy group      — standard Isaac Lab ObservationsCfg field name
    arena_camera_obs_group = "camera_obs"
    arena_policy_obs_group = "policy"

    # Camera field names in the arena observation dict.
    # The "_rgb" suffix is appended by make_camera_observation_cfg.
    arena_left_camera_key = "external_camera_rgb"
    arena_right_camera_key = "external_camera_2_rgb"
    arena_wrist_camera_key = "wrist_camera_rgb"

    def extract(self, observation: dict[str, Any], env_id: int) -> Cosmos3DroidObservation:
        cam = observation[self.arena_camera_obs_group]
        proprio = observation[self.arena_policy_obs_group]
        return Cosmos3DroidObservation(
            left_image=cam[self.arena_left_camera_key][env_id].detach().cpu().numpy(),
            right_image=cam[self.arena_right_camera_key][env_id].detach().cpu().numpy(),
            wrist_image=cam[self.arena_wrist_camera_key][env_id].detach().cpu().numpy(),
            joint_position=proprio["joint_pos"][env_id].detach().cpu().numpy(),
            gripper_position=proprio["gripper_pos"][env_id].detach().cpu().numpy(),
        )

    def pack_request(self, extracted: Cosmos3DroidObservation, language_instruction: str) -> dict[str, Any]:
        """Build the cosmos3 wire-format request.

        Composes a single 720×640 image by vertically stacking:
        - wrist image resized to (360, 640)
        - left + right images each downsampled to (180, 320) and stacked horizontally
        """
        # Resize wrist to target dimensions using openpi's resize_with_pad.
        wrist = image_tools.resize_with_pad(extracted.wrist_image, self.image_h, self.image_w)

        # Resize left/right to full size first, then bilinear downsample.
        half_size = (self.image_h // 2, self.image_w // 2)  # (180, 320)

        left = image_tools.resize_with_pad(extracted.left_image, self.image_h, self.image_w)
        left_t = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float()
        left_t = F.interpolate(left_t, size=half_size, mode="bilinear", align_corners=False)
        left = left_t.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)

        right = image_tools.resize_with_pad(extracted.right_image, self.image_h, self.image_w)
        right_t = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float()
        right_t = F.interpolate(right_t, size=half_size, mode="bilinear", align_corners=False)
        right = right_t.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)

        # Compose: vstack(wrist, hstack(left, right)) → (720, 640, 3)
        image = np.concatenate((wrist, np.concatenate((left, right), axis=1)), axis=0)
        assert image.shape == (self.image_h + self.image_h // 2, self.image_w, 3), (
            f"Expected composed image shape (720, 640, 3); got {image.shape}"
        )

        return {
            "observation/image": image,
            "observation/joint_position": extracted.joint_position,
            "observation/gripper_position": extracted.gripper_position,
            "prompt": language_instruction,
        }
