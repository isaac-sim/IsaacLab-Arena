# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy core: config, camera conversion, and policy observation building.

This module is the single source of truth (SSOT) for the embodiment-independent
parts of GR00T closed-loop policy logic. Embodiment-specific state and action
translation lives in ``Gr00tEmbodimentAdapter`` subclasses. It is structured so that:

- **Torch/numpy conversion is isolated at boundaries**: camera observations
  convert to numpy via ``extract_rgb_numpy_from_torch`` and robot state via the
  adapter's ``extract_state_np``; core logic is numpy-only; actions convert back
  to torch only in ``to_tensor``.
- **Core logic is modular and numpy-only**: ``resize_rgb_for_policy`` and
  ``build_gr00t_policy_observations`` do not use torch.

Pipeline: env (torch) -> ``extract_rgb_numpy_from_torch`` + adapter
``extract_state_np`` -> numpy core -> adapter ``build_action_np`` ->
``to_tensor`` -> tensor.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.utils.io_utils import to_numpy

# --------------------------------------------------------------------------- #
# Base config (shared by local and remote policies)
# --------------------------------------------------------------------------- #


@dataclass
class Gr00tBasePolicyCfg(PolicyCfg):
    """Configure settings shared by local and remote GR00T policies.

    Subclasses (e.g. for local vs remote) may add fields such as ``num_envs``.
    """

    policy_config_yaml_path: str
    """Path to the GR00T closed-loop policy YAML configuration."""

    policy_device: str = "cuda"
    """Device used for Arena-side policy operations."""


# --------------------------------------------------------------------------- #
# Observation format conversion (pipeline boundaries)
# --------------------------------------------------------------------------- #


def _extract_rgb_from_nested_obs(
    nested_obs: dict[str, np.ndarray] | dict[str, torch.Tensor],
    camera_names: list[str],
    group_key: str = "camera_obs",
    convert_to_numpy: bool = True,
) -> list[np.ndarray]:
    """Extract ordered list of RGB arrays from nested observation.

    Args:
        nested_obs: Dict with group_key (cam_name -> array or tensor).
        camera_names: Ordered list of camera keys to read.
        group_key: Top-level key for camera data (default "camera_obs").
        convert_to_numpy: If True, convert each value to numpy; if False, return as-is.

    Returns:
        List of (N, H, W, C) arrays one per camera.

    Raises:
        AssertionError: If group_key or a requested camera key is missing.
    """
    assert group_key in nested_obs, f"{group_key} is not in observation"
    rgb_list_np: list[np.ndarray] = []
    for cam in camera_names:
        assert cam in nested_obs[group_key], f"camera {cam} is not in {group_key}"
        val = nested_obs[group_key][cam]
        rgb_list_np.append(to_numpy(val) if convert_to_numpy else val)
    return rgb_list_np


def extract_rgb_numpy_from_torch(
    nested_obs: dict[str, Any],
    camera_names: list[str],
    camera_group_key: str = "camera_obs",
) -> list[np.ndarray]:
    """Convert the named camera observations to numpy for the closed-loop pipeline.

    Args:
        nested_obs: Env observation with camera_group_key (per-camera tensors).
        camera_names: Ordered list of camera keys to read from camera_group_key.
        camera_group_key: Top-level key for camera data in observation (default "camera_obs").

    Returns:
        List of (N, H, W, C) uint8 numpy arrays, one per camera.
    """
    return _extract_rgb_from_nested_obs(
        nested_obs=nested_obs, camera_names=camera_names, group_key=camera_group_key, convert_to_numpy=True
    )


# --------------------------------------------------------------------------- #
# Core SSOT: policy inputs (numpy-only, modality-config driven)
# --------------------------------------------------------------------------- #


def resize_rgb_for_policy(
    rgb_list_np: list[np.ndarray],
    target_image_size: tuple[int, int, int],
) -> list[np.ndarray]:
    """Resize each RGB frame to the policy target size with padding if needed.

    Args:
        rgb_list_np: List of (N, H, W, C) RGB arrays.
        target_image_size: (H, W, C) target dimensions.

    Returns:
        List of (N, H', W', C) arrays, each resized/padded to target H, W.
    """
    processed: list[np.ndarray] = []
    for rgb_np in rgb_list_np:
        if rgb_np.shape[1:3] != tuple(target_image_size[:2]):
            rgb_np = resize_frames_with_padding(
                rgb_np,
                target_image_size=target_image_size,
                bgr_conversion=False,
                pad_img=True,
            )
        processed.append(rgb_np)
    return processed


def build_gr00t_policy_observations(
    rgb_list_np: list[np.ndarray],
    state_np: dict[str, np.ndarray],
    task_description: str,
    policy_config: Gr00tClosedloopPolicyCfg,
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Build GR00T policy observation dict from numpy env observations.

    Resizes RGB, then fills language / video / state keys from modality config.
    No torch; use after :func:`extract_rgb_numpy_from_torch` and the embodiment
    adapter's ``extract_state_np``.

    Args:
        rgb_list_np: List of RGB arrays, one per camera; each shape (N, H, W, C).
        state_np: GR00T state key -> (N, dim) array; keys absent from the state
            modality config are not sent.
        task_description: Language instruction for the policy.
        policy_config: Closed-loop config (target_image_size, etc.).
        modality_configs: Dict with "language", "video", "state" modality configs.

    Returns:
        Nested dict "language" / "video" / "state" with keys from modality
        config and arrays shaped for GR00T (e.g. video: N, 1, H, W, C).
    """
    target_image_size = getattr(policy_config, "target_image_size", None)
    if target_image_size is not None:
        rgb_list_np = resize_rgb_for_policy(rgb_list_np=rgb_list_np, target_image_size=target_image_size)
    num_envs = rgb_list_np[0].shape[0]

    language_keys = modality_configs["language"].modality_keys
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    assert len(video_keys) == len(
        rgb_list_np
    ), f"number of video keys ({len(video_keys)}) and rgb inputs ({len(rgb_list_np)}) must match"

    # TODO(xinejiayao, 2025-12-10): when multi-task with parallel envs feature is enabled,
    # we need to pass in a list of task descriptions.
    policy_observations: dict[str, Any] = {
        "language": {language_keys[0]: [[task_description] for _ in range(num_envs)]},
        "video": {},
        "state": {},
    }
    for i, video_key in enumerate(video_keys):

        policy_observations["video"][video_key] = rgb_list_np[i].reshape(
            num_envs, 1, target_image_size[0], target_image_size[1], target_image_size[2]
        )
    for state_key in state_keys:
        if state_key in state_np:
            arr = state_np[state_key]
            assert (
                arr.shape[0] == num_envs
            ), f"state_np[{state_key}] has shape {arr.shape} but expected ({num_envs}, -1)"
            policy_observations["state"][state_key] = arr.reshape(num_envs, 1, -1)

    return policy_observations
