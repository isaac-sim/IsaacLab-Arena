# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, get_args

MAX_RECONNECT_ATTEMPTS = 3

Cam2Source = Literal["black", "right", "head", "duplicate"]
"""Source for the second exterior camera slot (observation/exterior_image_1_left):
- "black": fill with a zero (black) canvas; use when no second camera is mounted.
- "right": pull from the Arena camera named by cam_exterior_right.
- "head": pull from the Arena camera named by cam_head.
- "duplicate": copy the first exterior camera's image (cam_exterior_left).
"""

# TODO(tstuyck): The DreamZero checkpoint is trained on DROID only, and
# num_arm_joints/robot_joint_pos below assume the sim's articulation joint
# order already matches what the checkpoint expects (true for DROID by
# construction). Supporting another embodiment (e.g. G1, GR1T2) will need an
# explicit sim->policy joint remap, like isaaclab_arena_gr00t's
# joints_conversion.remap_sim_joints_to_policy_joints_from_np.
_SUPPORTED_EMBODIMENTS = ("droid",)


@dataclass
class DreamZeroRemotePolicyConfig:
    """Connection and runtime configuration for DreamZeroRemotePolicy."""

    remote_host: str = "localhost"
    """Hostname of the DreamZero inference server."""

    remote_port: int = 5000
    """Port the DreamZero inference server listens on."""

    open_loop_horizon: int = 24
    """Number of action steps to execute per server inference call."""

    num_arm_joints: int = 7
    """Number of arm DOF; used to split robot_joint_pos into arm joints and gripper."""

    embodiment: str = "droid"
    """Embodiment the checkpoint was trained on. Only 'droid' is currently supported."""

    cam_exterior_left: str = "external_camera_rgb"
    """Arena camera key that maps to observation/exterior_image_0_left."""

    cam2_source: Cam2Source = "black"
    """Source for the second exterior camera slot; see Cam2Source for what each value does."""

    cam_exterior_right: str = "external_camera_2_rgb"
    """Arena camera key used when cam2_source='right'."""

    cam_head: str = "head_camera"
    """Arena camera key used when cam2_source='head'."""

    cam_wrist: str = "wrist_camera_rgb"
    """Arena camera key that maps to observation/wrist_image_left."""

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    def __post_init__(self) -> None:
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"
        assert self.num_arm_joints > 0, "num_arm_joints must be positive"
        assert self.embodiment in _SUPPORTED_EMBODIMENTS, (
            f"DreamZeroRemotePolicy only supports {_SUPPORTED_EMBODIMENTS} embodiments"
            f" (checkpoint is DROID-only), got {self.embodiment!r}."
        )
        valid_cam2_sources = get_args(Cam2Source)
        assert (
            self.cam2_source in valid_cam2_sources
        ), f"cam2_source must be one of {valid_cam2_sources}, got {self.cam2_source!r}"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> DreamZeroRemotePolicyConfig:
        """Build config from parsed CLI arguments."""
        return cls(
            remote_host=args.dreamzero_host,
            remote_port=args.dreamzero_port,
            open_loop_horizon=args.dreamzero_open_loop_horizon,
            num_arm_joints=args.dreamzero_num_arm_joints,
            embodiment=args.dreamzero_embodiment,
            cam_exterior_left=args.dreamzero_cam_exterior_left,
            cam2_source=args.dreamzero_cam2_source,
            cam_exterior_right=args.dreamzero_cam_exterior_right,
            cam_head=args.dreamzero_cam_head,
            cam_wrist=args.dreamzero_cam_wrist,
            policy_device=args.policy_device,
        )
