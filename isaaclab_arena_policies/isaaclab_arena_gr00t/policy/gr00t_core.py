# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T client parts shared by all embodiments: the base policy config, camera image conversion, and request building.

``build_gr00t_policy_observations`` combines the resized camera images, the state arrays, and the language
instruction into the observation dict the GR00T server expects.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import (
    import_gr00t_embodiment_adapter_class,
)
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

    gr00t_embodiment_adapter: str
    """Adapter for the robot and checkpoint: a built-in name (droid, gr1_arms, g1_wbc) or a dotted class path."""

    policy_device: str = "cuda"
    """Device used for Arena-side policy operations."""

    def __post_init__(self):
        # Import the adapter class now so a bad name fails at config load, before any env is built.
        import_gr00t_embodiment_adapter_class(self.gr00t_embodiment_adapter)


# --------------------------------------------------------------------------- #
# Observation format conversion (pipeline boundaries)
# --------------------------------------------------------------------------- #


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
    assert camera_group_key in nested_obs, f"{camera_group_key} is not in observation"
    rgb_list_np: list[np.ndarray] = []
    for cam in camera_names:
        assert cam in nested_obs[camera_group_key], f"camera {cam} is not in {camera_group_key}"
        rgb_list_np.append(to_numpy(nested_obs[camera_group_key][cam]))
    return rgb_list_np


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
    target_image_size: tuple[int, int, int],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Build GR00T policy observation dict from numpy env observations.

    Resizes RGB, then fills language / video / state keys from modality config.
    No torch; use after ``extract_rgb_numpy_from_torch`` and the embodiment
    adapter's ``extract_state_np``.

    Args:
        rgb_list_np: List of RGB arrays, one per camera; each shape (N, H, W, C).
        state_np: GR00T state key -> (N, dim) array. It must contain every state modality key;
            other keys are not sent.
        task_description: Language instruction for the policy.
        target_image_size: (H, W, C) that each RGB frame is padded and resized to.
        modality_configs: Dict with "language", "video", "state" modality configs.

    Returns:
        Nested dict "language" / "video" / "state" with keys from modality
        config and arrays shaped for GR00T (e.g. video: N, 1, H, W, C).
    """
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
        assert state_key in state_np, f"State key {state_key!r} is missing; the adapter returned {sorted(state_np)}"
        arr = state_np[state_key]
        assert arr.shape[0] == num_envs, f"state_np[{state_key}] has shape {arr.shape} but expected ({num_envs}, -1)"
        policy_observations["state"][state_key] = arr.reshape(num_envs, 1, -1)

    return policy_observations
