# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration constants, variant table, and CLI args for the openpi DROID policy.

Holds everything that is *not* execution logic: image / action-layout
facts fixed by upstream pi0 droid checkpoints, Arena's DROID observation
key names, the per-variant open-loop horizon table, and the runtime
``Pi0DroidRemotePolicyArgs`` dataclass consumed by the policy.
"""

from dataclasses import dataclass

# --- Pi0 droid checkpoint facts (fixed by upstream) ---
# DROID action layout: 7 panda arm joints + 1 gripper command.
ACTION_DIM = 8
# Image format the trained models expect (after letterbox padding).
TARGET_IMAGE_SIZE = (224, 224)
# Per-variant open-loop horizon: how many actions to replay before refetching.
OPEN_LOOP_HORIZON_BY_VARIANT: dict[str, int] = {
    "pi05": 15,
    "pi0": 10,
    "pi0_fast": 10,
}
DEFAULT_VARIANT = "pi05"

# --- Arena DROID embodiment vocabulary ---
ARENA_EXTERNAL_CAMERA_KEY = "external_camera_rgb"
ARENA_WRIST_CAMERA_KEY = "wrist_camera_rgb"

# --- Network knob ---
MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class Pi0DroidRemotePolicyArgs:
    """Connection + runtime config for ``Pi0DroidRemotePolicy``."""

    policy_variant: str = DEFAULT_VARIANT
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 8000
