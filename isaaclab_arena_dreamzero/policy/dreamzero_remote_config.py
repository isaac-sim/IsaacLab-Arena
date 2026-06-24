# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass

MAX_RECONNECT_ATTEMPTS = 3

_VALID_CAM2_SOURCES = ("black", "right", "head", "duplicate")


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

    cam_exterior_left: str = "over_shoulder_left_camera"
    """Arena camera key that maps to observation/exterior_image_0_left."""

    cam2_source: str = "black"
    """Source for the second exterior camera slot. One of: black, right, head, duplicate."""

    cam_exterior_right: str = "over_shoulder_right_camera"
    """Arena camera key used when cam2_source='right'."""

    cam_head: str = "head_camera"
    """Arena camera key used when cam2_source='head'."""

    cam_wrist: str = "wrist_cam"
    """Arena camera key that maps to observation/wrist_image_left."""

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    def __post_init__(self) -> None:
        assert self.open_loop_horizon > 0, "open_loop_horizon must be positive"
        assert self.num_arm_joints > 0, "num_arm_joints must be positive"
        assert (
            self.cam2_source in _VALID_CAM2_SOURCES
        ), f"cam2_source must be one of {_VALID_CAM2_SOURCES}, got {self.cam2_source!r}"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> DreamZeroRemotePolicyConfig:
        """Build config from parsed CLI arguments."""
        return cls(
            remote_host=args.dreamzero_host,
            remote_port=args.dreamzero_port,
            open_loop_horizon=args.dreamzero_open_loop_horizon,
            num_arm_joints=args.dreamzero_num_arm_joints,
            cam_exterior_left=args.dreamzero_cam_exterior_left,
            cam2_source=args.dreamzero_cam2_source,
            cam_exterior_right=args.dreamzero_cam_exterior_right,
            cam_head=args.dreamzero_cam_head,
            cam_wrist=args.dreamzero_cam_wrist,
            policy_device=args.policy_device,
        )
