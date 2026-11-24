# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from openpi_client import image_tools

from isaaclab_arena_openpi.policy.pi0_remote_policy import Pi0EmbodimentAdapter


@dataclass(frozen=True)
class DroidObservation:
    """Per-env tensors needed to assemble an openpi DROID request."""

    exterior_image: np.ndarray  # (H, W, 3) uint8
    wrist_image: np.ndarray  # (H, W, 3) uint8
    joint_position: np.ndarray  # (7,) float32
    gripper_position: np.ndarray  # (1,) float32


class Pi0DroidAdapter(Pi0EmbodimentAdapter):
    """Wire format for upstream openpi DROID checkpoints (pi05/pi0)."""

    # Fixed by upstream pi0 droid checkpoints: 7 panda joints + 1 gripper command.
    action_dim = 8

    # How many actions to replay before refetching a new chunk from the server.
    open_loop_horizon_by_variant = {
        "pi05": 15,
        "pi0": 10,
    }

    target_image_size = (224, 224)
    arena_external_camera_key = "external_camera_rgb"
    arena_wrist_camera_key = "wrist_camera_rgb"

    # Top-level keys on the arena gym observation dict that this adapter
    # consumes. Both are conventions:
    #   arena_camera_obs_group  - set by isaaclab_arena.utils.cameras.make_camera_observation_cfg;
    #                             every arena embodiment using that helper exposes cameras here.
    #   arena_policy_obs_group  - standard Isaac Lab ObservationsCfg field name; every arena
    #                             embodiment (droid, franka, galbot, agibot, ...) defines a
    #                             `policy: PolicyCfg` group.
    arena_camera_obs_group = "camera_obs"
    arena_policy_obs_group = "policy"

    def extract(self, observation: dict[str, Any], env_id: int) -> DroidObservation:
        cam = observation[self.arena_camera_obs_group]
        proprio = observation[self.arena_policy_obs_group]
        return DroidObservation(
            exterior_image=cam[self.arena_external_camera_key][env_id].detach().cpu().numpy(),
            wrist_image=cam[self.arena_wrist_camera_key][env_id].detach().cpu().numpy(),
            joint_position=proprio["joint_pos"][env_id].detach().cpu().numpy(),
            gripper_position=proprio["gripper_pos"][env_id].detach().cpu().numpy(),
        )

    def pack_request(self, extracted: DroidObservation, language_instruction: str) -> dict[str, Any]:
        target_height, target_width = self.target_image_size
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                extracted.exterior_image, target_height, target_width
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                extracted.wrist_image, target_height, target_width
            ),
            "observation/joint_position": extracted.joint_position,
            "observation/gripper_position": extracted.gripper_position,
            "prompt": language_instruction,
        }
