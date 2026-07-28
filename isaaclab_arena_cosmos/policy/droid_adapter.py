# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from isaaclab_arena_cosmos.policy.cosmos_remote_policy import CosmosEmbodimentAdapter


@dataclass(frozen=True)
class DroidObservation:
    """Per-env tensors needed to assemble a Cosmos RoboLab DROID request."""

    wrist_image: np.ndarray  # (H, W, 3) uint8
    exterior_image_1: np.ndarray  # (H, W, 3) uint8
    exterior_image_2: np.ndarray  # (H, W, 3) uint8
    joint_position: np.ndarray  # (7,) float32
    gripper_position: np.ndarray  # (1,) float32


class CosmosDroidAdapter(CosmosEmbodimentAdapter):
    """Wire format for the released Cosmos DROID policies (Nano/Edge), joint_pos action space.

    The RoboLab server composes its ``concat_view`` (wrist on top, two third-person
    views on the bottom) from the wrist + two exterior images, so this adapter forwards
    all three of Arena's DROID cameras. Images are sent at their native resolution; the
    server resizes them to the model's input size.
    """

    # Fixed by the released DROID joint_pos policies: 7 arm joints + 1 gripper command.
    action_dim = 8

    # Arena DROID camera keys (see isaaclab_arena/embodiments/droid/droid.py).
    arena_wrist_camera_key = "wrist_camera_rgb"
    arena_exterior_camera_1_key = "external_camera_rgb"
    arena_exterior_camera_2_key = "external_camera_2_rgb"

    # Top-level keys on the arena gym observation dict this adapter consumes; both are
    # standard Arena conventions (camera group from make_camera_observation_cfg, and the
    # embodiment's `policy` PolicyCfg group).
    arena_camera_obs_group = "camera_obs"
    arena_policy_obs_group = "policy"

    def extract(self, observation: dict[str, Any], env_id: int) -> DroidObservation:
        cam = observation[self.arena_camera_obs_group]
        proprio = observation[self.arena_policy_obs_group]
        return DroidObservation(
            wrist_image=_to_uint8_hwc(cam[self.arena_wrist_camera_key][env_id]),
            exterior_image_1=_to_uint8_hwc(cam[self.arena_exterior_camera_1_key][env_id]),
            exterior_image_2=_to_uint8_hwc(cam[self.arena_exterior_camera_2_key][env_id]),
            joint_position=proprio["joint_pos"][env_id].detach().cpu().numpy(),
            gripper_position=proprio["gripper_pos"][env_id].detach().cpu().numpy(),
        )

    def pack_request(self, extracted: DroidObservation, language_instruction: str) -> dict[str, Any]:
        return {
            "observation/wrist_image_left": extracted.wrist_image,
            "observation/exterior_image_1_left": extracted.exterior_image_1,
            "observation/exterior_image_2_left": extracted.exterior_image_2,
            "observation/joint_position": extracted.joint_position,
            "observation/gripper_position": extracted.gripper_position,
            "prompt": language_instruction,
        }


def _to_uint8_hwc(image: Any) -> np.ndarray:
    """Return a contiguous (H, W, 3) uint8 array from an Arena camera tensor."""
    array = image.detach().cpu().numpy()
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))
