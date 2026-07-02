# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Literal, get_args

from isaaclab_arena_dreamzero.policy.dreamzero_remote_policy import DreamZeroEmbodimentAdapter
from isaaclab_arena_dreamzero.policy.image_utils import TARGET_H, TARGET_W, resize_with_pad

Cam2Source = Literal["black", "right", "head", "duplicate"]
"""Source for the second exterior camera slot (observation/exterior_image_1_left):

- "black": fill with a zero (black) canvas; use when no second camera is mounted.
- "right": pull from the Arena camera named by cam_exterior_right.
- "head": pull from the Arena camera named by cam_head.
- "duplicate": copy the first exterior camera's image (cam_exterior_left).
"""


@dataclass
class DroidAdapterConfig:
    """DROID-specific observation wire-format configuration for DroidAdapter."""

    num_arm_joints: int = 7
    """Number of arm DOF; used to split robot_joint_pos into arm joints and gripper."""

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

    target_image_height: int = TARGET_H
    """Height images are letterbox-resized to before being sent to the DreamZero-DROID checkpoint."""

    target_image_width: int = TARGET_W
    """Width images are letterbox-resized to before being sent to the DreamZero-DROID checkpoint."""

    def __post_init__(self) -> None:
        assert self.num_arm_joints > 0, "num_arm_joints must be positive"
        valid_cam2_sources = get_args(Cam2Source)
        assert (
            self.cam2_source in valid_cam2_sources
        ), f"cam2_source must be one of {valid_cam2_sources}, got {self.cam2_source!r}"
        assert self.target_image_height > 0, "target_image_height must be positive"
        assert self.target_image_width > 0, "target_image_width must be positive"

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add DROID adapter CLI arguments to parser."""
        group = parser.add_argument_group(
            "DreamZero DROID Adapter",
            "Arguments for the DROID observation/action wire format.",
        )
        group.add_argument(
            "--dreamzero_num_arm_joints",
            type=int,
            default=7,
            help="Number of arm DOF in robot_joint_pos (remainder is gripper).",
        )
        group.add_argument(
            "--dreamzero_cam_exterior_left",
            type=str,
            default="external_camera_rgb",
            help="Arena camera key for the primary exterior (left shoulder) camera.",
        )
        group.add_argument(
            "--dreamzero_cam2_source",
            type=str,
            default="black",
            choices=list(get_args(Cam2Source)),
            help="Source for the second exterior camera slot.",
        )
        group.add_argument(
            "--dreamzero_cam_exterior_right",
            type=str,
            default="external_camera_2_rgb",
            help="Arena camera key for the right shoulder camera (used when cam2_source='right').",
        )
        group.add_argument(
            "--dreamzero_cam_head",
            type=str,
            default="head_camera",
            help="Arena camera key for the head camera (used when cam2_source='head').",
        )
        group.add_argument(
            "--dreamzero_cam_wrist",
            type=str,
            default="wrist_camera_rgb",
            help="Arena camera key for the wrist camera.",
        )
        group.add_argument(
            "--dreamzero_target_image_height",
            type=int,
            default=TARGET_H,
            help="Height images are letterbox-resized to before being sent to the checkpoint.",
        )
        group.add_argument(
            "--dreamzero_target_image_width",
            type=int,
            default=TARGET_W,
            help="Width images are letterbox-resized to before being sent to the checkpoint.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> DroidAdapterConfig:
        """Build config from parsed CLI arguments."""
        return cls(
            num_arm_joints=args.dreamzero_num_arm_joints,
            cam_exterior_left=args.dreamzero_cam_exterior_left,
            cam2_source=args.dreamzero_cam2_source,
            cam_exterior_right=args.dreamzero_cam_exterior_right,
            cam_head=args.dreamzero_cam_head,
            cam_wrist=args.dreamzero_cam_wrist,
            target_image_height=args.dreamzero_target_image_height,
            target_image_width=args.dreamzero_target_image_width,
        )


@dataclass(frozen=True)
class DroidObservation:
    """Per-env tensors needed to assemble a DreamZero DROID request."""

    exterior_image_0: np.ndarray  # (target_image_height, target_image_width, 3) uint8
    exterior_image_1: np.ndarray  # (target_image_height, target_image_width, 3) uint8
    wrist_image: np.ndarray  # (target_image_height, target_image_width, 3) uint8
    joint_position: np.ndarray  # (num_arm_joints,) float32
    gripper_position: np.ndarray  # (1,) float32


class DroidAdapter(DreamZeroEmbodimentAdapter):
    """Wire format for the DreamZero-DROID checkpoint."""

    # Fixed by the DreamZero-DROID checkpoint: 7 arm joints + 1 gripper command.
    action_dim = 8

    def __init__(self, config: DroidAdapterConfig) -> None:
        self.config = config

    def extract(self, observation: dict[str, Any], env_id: int) -> DroidObservation:
        """Pull one env's camera and joint tensors out of the Arena observation dict."""
        cam_obs = observation["camera_obs"]
        joint_pos_full = observation["policy"]["robot_joint_pos"][env_id].detach().cpu().numpy().astype(np.float32)
        n = self.config.num_arm_joints

        img0 = self._extract_image(cam_obs, self.config.cam_exterior_left, env_id)
        img1 = self._resolve_cam2(cam_obs, env_id, img0)
        img_wrist = self._extract_image(cam_obs, self.config.cam_wrist, env_id)

        return DroidObservation(
            exterior_image_0=img0,
            exterior_image_1=img1,
            wrist_image=img_wrist,
            joint_position=joint_pos_full[:n],
            gripper_position=joint_pos_full[n : n + 1],
        )

    def pack_request(self, extracted: DroidObservation) -> dict[str, Any]:
        """Build the observation/* portion of the DreamZero wire-format request."""
        return {
            "observation/exterior_image_0_left": extracted.exterior_image_0,
            "observation/exterior_image_1_left": extracted.exterior_image_1,
            "observation/wrist_image_left": extracted.wrist_image,
            "observation/joint_position": extracted.joint_position,
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": extracted.gripper_position,
        }

    def _extract_image(self, cam_obs: dict[str, Any], cam_name: str, env_id: int) -> np.ndarray:
        """Slice one camera frame, move to CPU, and resize to (target_image_height, target_image_width, 3).

        Args:
            cam_obs: Arena camera_obs sub-dict.
            cam_name: Key into cam_obs.
            env_id: Batch index.

        Returns:
            uint8 ndarray of shape (target_image_height, target_image_width, 3).
        """
        assert (
            cam_name in cam_obs
        ), f"Camera {cam_name!r} not found in observation. Available cameras: {sorted(cam_obs.keys())}"
        tensor = cam_obs[cam_name][env_id].detach().cpu()
        assert tensor.dtype == torch.uint8, (
            f"Camera {cam_name!r} returned {tensor.dtype} tensor; expected uint8."
            " If Isaac Sim is configured with float images, convert to uint8 in your scene config."
        )
        return resize_with_pad(tensor.numpy(), self.config.target_image_height, self.config.target_image_width)

    def _resolve_cam2(self, cam_obs: dict[str, Any], env_id: int, img0: np.ndarray) -> np.ndarray:
        """Return the image for the second exterior camera slot based on cam2_source.

        Args:
            cam_obs: Arena camera_obs sub-dict.
            env_id: Batch index.
            img0: Already-extracted exterior_left image (used for 'duplicate' mode).

        Returns:
            uint8 ndarray of shape (target_image_height, target_image_width, 3).
        """
        src = self.config.cam2_source
        if src == "black":
            return np.zeros((self.config.target_image_height, self.config.target_image_width, 3), dtype=np.uint8)
        if src == "duplicate":
            return img0.copy()
        if src == "right":
            return self._extract_image(cam_obs, self.config.cam_exterior_right, env_id)
        if src == "head":
            return self._extract_image(cam_obs, self.config.cam_head, env_id)
        raise ValueError(f"Unreachable: unknown cam2_source {src!r}")
