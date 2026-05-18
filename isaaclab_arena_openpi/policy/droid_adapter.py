# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from typing import Any

from openpi_client import image_tools

from isaaclab_arena_openpi.policy.pi0_remote_policy import Pi0EmbodimentAdapter


class Pi0DroidAdapter(Pi0EmbodimentAdapter):
    """Wire format for upstream openpi DROID checkpoints (pi05/pi0/pi0_fast)."""

    # Fixed by upstream pi0 droid checkpoints: 7 panda joints + 1 gripper command.
    action_dim = 8

    # How many actions to replay before refetching a new chunk from the server.
    open_loop_horizon_by_variant = {
        "pi05": 15,
        "pi0": 10,
        "pi0_fast": 10,
    }

    target_image_size = (224, 224)
    arena_external_camera_key = "external_camera_rgb"
    arena_wrist_camera_key = "wrist_camera_rgb"

    def extract(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        cam = observation["camera_obs"]
        proprio = observation["policy"]
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in {
                "exterior_image": cam[self.arena_external_camera_key][0],
                "wrist_image": cam[self.arena_wrist_camera_key][0],
                "joint_position": proprio["joint_pos"][0],
                "gripper_position": proprio["gripper_pos"][0],
            }.items()
        }

    def pack_request(self, extracted: dict[str, np.ndarray], language_instruction: str) -> dict[str, Any]:
        target_height, target_width = self.target_image_size
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                extracted["exterior_image"], target_height, target_width
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                extracted["wrist_image"], target_height, target_width
            ),
            "observation/joint_position": extracted["joint_position"],
            "observation/gripper_position": extracted["gripper_position"],
            "prompt": language_instruction,
        }
