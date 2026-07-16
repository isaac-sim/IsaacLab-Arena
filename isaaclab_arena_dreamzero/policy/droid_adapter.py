# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Literal, get_args

from isaaclab_arena_dreamzero.policy.dreamzero_remote_policy import DreamZeroEmbodimentAdapter
from isaaclab_arena_dreamzero.policy.image_utils import TARGET_H, TARGET_W, resize_with_pad

Cam2Source = Literal["black", "right", "duplicate"]
"""Source for the second exterior camera slot (observation/exterior_image_1_left):

- "right" (default): pull from the Arena camera named by DroidAdapter.cam_exterior_right;
  the droid embodiment mounts this second exterior camera, matching DROID training data.
- "black": fill with a zero (black) canvas; use when no second camera is mounted.
- "duplicate": copy the first exterior camera's image (DroidAdapter.cam_exterior_left).
"""


@dataclass(frozen=True)
class DroidObservation:
    """Per-env tensors needed to assemble a DreamZero DROID request."""

    exterior_image_0: np.ndarray  # (target_image_size, 3) uint8
    exterior_image_1: np.ndarray  # (target_image_size, 3) uint8
    wrist_image: np.ndarray  # (target_image_size, 3) uint8
    joint_position: np.ndarray  # (num_arm_joints,) float32
    gripper_position: np.ndarray  # (1,) float32


class DroidAdapter(DreamZeroEmbodimentAdapter):
    """Wire format for the DreamZero-DROID checkpoint.

    Values fixed by the droid embodiment (camera keys, arm DOF) and by the
    DreamZero-DROID checkpoint (action layout, target image size) are class
    constants, mirroring the openpi DROID adapter. Only cam2_source is a
    per-eval choice, so it is the sole constructor argument.
    """

    # Fixed by the DreamZero-DROID checkpoint: 7 arm joints + 1 gripper command.
    action_dim = 8

    # Arm DOF count; the remainder of robot_joint_pos is treated as the gripper.
    num_arm_joints = 7

    # (height, width) images are letterbox-resized to before being sent to the checkpoint.
    target_image_size = (TARGET_H, TARGET_W)

    # Arena camera_obs keys published by the droid embodiment (see DroidCameraCfg).
    cam_exterior_left = "external_camera_rgb"
    cam_exterior_right = "external_camera_2_rgb"
    cam_wrist = "wrist_camera_rgb"

    def __init__(self, cam2_source: Cam2Source = "right") -> None:
        valid_cam2_sources = get_args(Cam2Source)
        assert (
            cam2_source in valid_cam2_sources
        ), f"cam2_source must be one of {valid_cam2_sources}, got {cam2_source!r}"
        self.cam2_source: Cam2Source = cam2_source

    def extract(self, observation: dict[str, Any], env_id: int) -> DroidObservation:
        """Pull one env's camera and joint tensors out of the Arena observation dict."""
        cam_obs = observation["camera_obs"]
        joint_pos_full = observation["policy"]["robot_joint_pos"][env_id].detach().cpu().numpy().astype(np.float32)
        n = self.num_arm_joints

        img0 = self._extract_image(cam_obs, self.cam_exterior_left, env_id)
        img1 = self._resolve_cam2(cam_obs, env_id, img0)
        img_wrist = self._extract_image(cam_obs, self.cam_wrist, env_id)

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

    def parse_actions(self, response: dict[str, Any] | np.ndarray) -> np.ndarray:
        """Decode a DreamZero-DROID response into a float32 (num_steps, action_dim) chunk.

        Accepts either a bare ndarray or a dict with an 'actions' key. The
        DreamZero-DROID checkpoint may return action_dim - 1 columns (7 arm joints,
        no gripper command); a zero (open) gripper column is appended in that case.
        """
        raw = response.get("actions", response) if isinstance(response, dict) else response
        chunk = np.asarray(raw, dtype=np.float32)
        assert chunk.ndim == 2, f"Expected 2-D action chunk from server, got shape {chunk.shape}"
        assert chunk.shape[1] in (
            self.action_dim - 1,
            self.action_dim,
        ), f"Expected {self.action_dim - 1} or {self.action_dim} action dims, got {chunk.shape[1]}"
        if chunk.shape[1] == self.action_dim - 1:
            chunk = np.concatenate([chunk, np.zeros((len(chunk), 1), dtype=np.float32)], axis=1)
        return chunk

    def _extract_image(self, cam_obs: dict[str, Any], cam_name: str, env_id: int) -> np.ndarray:
        """Slice one camera frame, move to CPU, and resize to (*target_image_size, 3).

        Args:
            cam_obs: Arena camera_obs sub-dict.
            cam_name: Key into cam_obs.
            env_id: Batch index.

        Returns:
            uint8 ndarray of shape (*target_image_size, 3).
        """
        assert (
            cam_name in cam_obs
        ), f"Camera {cam_name!r} not found in observation. Available cameras: {sorted(cam_obs.keys())}"
        tensor = cam_obs[cam_name][env_id].detach().cpu()
        assert tensor.dtype == torch.uint8, (
            f"Camera {cam_name!r} returned {tensor.dtype} tensor; expected uint8."
            " If Isaac Sim is configured with float images, convert to uint8 in your scene config."
        )
        height, width = self.target_image_size
        return resize_with_pad(tensor.numpy(), height, width)

    def _resolve_cam2(self, cam_obs: dict[str, Any], env_id: int, img0: np.ndarray) -> np.ndarray:
        """Return the image for the second exterior camera slot based on cam2_source.

        Args:
            cam_obs: Arena camera_obs sub-dict.
            env_id: Batch index.
            img0: Already-extracted exterior_left image (used for 'duplicate' mode).

        Returns:
            uint8 ndarray of shape (*target_image_size, 3).
        """
        src = self.cam2_source
        if src == "black":
            height, width = self.target_image_size
            return np.zeros((height, width, 3), dtype=np.uint8)
        if src == "duplicate":
            return img0.copy()
        if src == "right":
            return self._extract_image(cam_obs, self.cam_exterior_right, env_id)
        raise ValueError(f"Unreachable: unknown cam2_source {src!r}")
