# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

# Fixed by upstream pi0 droid checkpoints: 7 panda joints + 1 gripper command.
ACTION_DIM = 8
TARGET_IMAGE_SIZE = (224, 224)

# How many actions to replay before refetching a new chunk from the server.
OPEN_LOOP_HORIZON_BY_VARIANT: dict[str, int] = {
    "pi05": 15,
    "pi0": 10,
    "pi0_fast": 10,
}
DEFAULT_VARIANT = "pi05"

ARENA_EXTERNAL_CAMERA_KEY = "external_camera_rgb"
ARENA_WRIST_CAMERA_KEY = "wrist_camera_rgb"

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class Pi0DroidRemotePolicyArgs:
    """Connection + runtime config for ``Pi0DroidRemotePolicy``."""

    policy_variant: str = DEFAULT_VARIANT
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 8000
