# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T policy core: config, observation conversion, and action building.

This module is the single source of truth (SSOT) for GR00T closed-loop policy
logic. It is structured so that:

- **Torch/numpy conversion is isolated at boundaries**: observation paths
  convert to numpy once (``extract_obs_numpy_from_torch`` or
  ``extract_obs_numpy_from_packed``); core logic is numpy-only; the action
  path converts back to torch only in ``action_numpy_to_tensor``.
- **Core logic is modular and numpy-only**: ``resize_rgb_for_policy``,
  ``remap_sim_joints_to_policy_joints_from_np`` (in joints_conversion),
  ``build_gr00t_policy_observations``, and ``build_gr00t_action_np`` do not
  use torch.

Pipelines:

- **Local**: env (torch) -> ``extract_obs_numpy_from_torch`` -> numpy core
  -> ``build_gr00t_action_np`` -> ``to_tensor`` -> tensor.
- **Remote**: socket (packed) -> ``extract_obs_numpy_from_packed`` -> same
  numpy core and action path.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml, to_numpy, to_tensor
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints_np,
    remap_sim_joints_to_policy_joints_from_np,
)

# --------------------------------------------------------------------------- #
# Base config (shared by local and remote policies)
# --------------------------------------------------------------------------- #


@dataclass
class Gr00tBasePolicyArgs:
    """Base arguments for GR00T policies, shared by local and remote.

    Subclasses (e.g. for local vs remote) may add fields such as ``num_envs``.
    """

    policy_config_yaml_path: str = field(
        metadata={
            "help": "Path to the Gr00t closedloop policy config YAML file",
            "required": True,
        }
    )

    policy_device: str = field(
        default="cuda",
        metadata={
            "help": "Device to use for policy inference (e.g. 'cuda', 'cpu').",
        },
    )


# --------------------------------------------------------------------------- #
# Config, model, and joint helpers (backend-agnostic)
# --------------------------------------------------------------------------- #


def load_gr00t_policy_from_config(policy_config: Gr00tClosedloopPolicyConfig) -> Gr00tPolicy:
    """Instantiate a GR00T policy from the closed-loop config.

    Args:
        policy_config: Loaded closed-loop config (model path, embodiment, device).

    Returns:
        Loaded ``Gr00tPolicy`` on the configured device.

    Raises:
        AssertionError: If ``policy_config.model_path`` does not exist.
    """
    model_path = policy_config.model_path
    # HuggingFace Hub repo IDs use "owner/repo" format (e.g. "nvidia/GR00T-N1.6-DROID").
    is_hf_id = bool(model_path and "/" in model_path and not model_path.startswith(("/", ".")))
    assert (
        Path(model_path).exists() or is_hf_id
    ), f"Model path {model_path} does not exist and is not a HuggingFace model id"
    return Gr00tPolicy(
        model_path=policy_config.model_path,
        embodiment_tag=EmbodimentTag[policy_config.embodiment_tag],
        device=policy_config.policy_device,
        strict=True,
    )


def load_gr00t_joint_configs(
    policy_config: Gr00tClosedloopPolicyConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load policy, action, and state joint configs from paths in the config.

    Args:
        policy_config: Closed-loop config with paths to joint YAMLs.

    Returns:
        A 3-tuple of (policy_joints_config, robot_action_joints_config,
        robot_state_joints_config), each a joint-name -> config dict.
    """
    policy_joints_config = load_robot_joints_config_from_yaml(policy_config.policy_joints_config_path)
    robot_action_joints_config = load_robot_joints_config_from_yaml(policy_config.action_joints_config_path)
    robot_state_joints_config = load_robot_joints_config_from_yaml(policy_config.state_joints_config_path)
    return policy_joints_config, robot_action_joints_config, robot_state_joints_config


def compute_action_dim(task_mode: TaskMode, robot_action_joints_config: dict[str, Any]) -> int:
    """Compute the total action dimension for the given task mode and joints.

    For locomanipulation, adds WBC command dimensions (navigate, base height,
    torso orientation RPY) to the joint count.

    Args:
        task_mode: Current task mode (e.g. G1_LOCOMANIPULATION).
        robot_action_joints_config: Action joint config (joint names -> config).

    Returns:
        Total action dimension (number of action components per step).
    """
    action_dim = len(robot_action_joints_config)
    if task_mode == TaskMode.G1_LOCOMANIPULATION:
        action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
    return action_dim


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


def _extract_joints_from_nested_obs(
    nested_obs: dict[str, np.ndarray] | dict[str, torch.Tensor],
    group_key: str = "policy",
    joint_pos_name: str = "robot_joint_pos",
    convert_to_numpy: bool = True,
) -> np.ndarray:
    """Extract robot joint positions from nested observation.

    Args:
        nested_obs: Dict with group_key[joint_pos_name] (array or tensor).
        group_key: Top-level key under which joint positions live (default "policy").
        joint_pos_name: Key for the joint position array (default "robot_joint_pos").
        convert_to_numpy: If True, convert to numpy; if False, return as-is.

    Returns:
        (N, num_joints) array in sim joint order.
    """
    assert group_key in nested_obs, f"{group_key} is not in observation"
    assert joint_pos_name in nested_obs[group_key], f"{joint_pos_name} is not in {group_key}"
    val = nested_obs[group_key][joint_pos_name]
    return to_numpy(val) if convert_to_numpy else val


def extract_obs_numpy_from_torch(
    nested_obs: dict[str, Any],
    camera_names: list[str],
    camera_group_key: str = "camera_obs",
    joint_group_key: str = "policy",
    joint_pos_name: str = "robot_joint_pos",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convert torch env observation to numpy for the local pipeline.

    This is the single torch-to-numpy boundary for the local (in-process)
    pipeline. The remote pipeline gets numpy over the socket and uses
    ``extract_obs_numpy_from_packed`` instead; both then share the same
    downstream interface.

    Args:
        observation: Env observation with camera_group_key (per-camera tensors)
            and joint_group_key[joint_pos_key] (torch tensor).
        camera_names: Ordered list of camera keys to read from camera_group_key.
        camera_group_key: Top-level key for camera data in observation (default "camera_obs").
        joint_group_key: Top-level key for joint data in observation (default "policy").
        joint_pos_key: Key for the joint position array (default "robot_joint_pos").

    Returns:
        rgb_list_np: List of (N, H, W, C) uint8 numpy arrays, one per camera.
        joint_pos_sim_np: (N, num_joints) float64 numpy array in sim joint order.

    """

    rgb_list_np = _extract_rgb_from_nested_obs(
        nested_obs=nested_obs, camera_names=camera_names, group_key=camera_group_key, convert_to_numpy=True
    )
    joint_pos_sim_np = _extract_joints_from_nested_obs(
        nested_obs=nested_obs, group_key=joint_group_key, joint_pos_name=joint_pos_name, convert_to_numpy=True
    )
    return rgb_list_np, joint_pos_sim_np


def extract_obs_numpy_from_packed(
    packed_observation: dict[str, Any],
    camera_names: list[str],
    unpack_fn: Callable[[dict[str, Any]], dict[str, Any]],
    camera_group_key: str = "camera_obs",
    joint_group_key: str = "policy",
    joint_pos_name: str = "robot_joint_pos",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract numpy observation from packed dict for the remote pipeline.

    The remote (client-server) pipeline receives observations as serialized
    numpy in a flat dict. This helper unpacks them into the same
    (rgb_list_np, joint_pos_sim_np) tuple used by the core logic so both
    pipelines share one downstream interface.

    Args:
        packed_observation: Flat observation dict (e.g. from socket).
        camera_names: Ordered list of camera keys to read after unpacking.
        unpack_fn: Function that converts packed_observation to a nested dict
            with camera_group_key and joint_group_key[joint_pos_name].
        camera_group_key: Top-level key for camera data in unpacked obs (default "camera_obs").
        joint_group_key: Top-level key for joint data in unpacked obs (default "policy").
        joint_pos_name: Key for the joint position array (default "robot_joint_pos").

    Returns:
        rgb_list_np: List of (N, H, W, C) numpy arrays, one per camera.
        joint_pos_sim_np: (N, num_joints) float numpy array in sim joint order.

    Raises:
        AssertionError: If unpacked observation lacks "camera_obs" or a camera key.
    """
    nested_obs = unpack_fn(packed_observation)
    rgb_list_np = _extract_rgb_from_nested_obs(
        nested_obs=nested_obs, camera_names=camera_names, group_key=camera_group_key, convert_to_numpy=False
    )
    joint_pos_sim_np = _extract_joints_from_nested_obs(
        nested_obs=nested_obs, group_key=joint_group_key, joint_pos_name=joint_pos_name, convert_to_numpy=False
    )
    return rgb_list_np, joint_pos_sim_np


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
    joint_pos_sim_np: np.ndarray,
    task_description: str,
    policy_config: Gr00tClosedloopPolicyConfig,
    robot_state_joints_config: dict[str, Any],
    policy_joints_config: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Build GR00T policy observation dict from numpy env observations.

    Resizes RGB, remaps sim joints to policy order, then fills language / video /
    state keys from modality config. No torch; use after
    :func:`extract_obs_numpy_from_torch` or :func:`extract_obs_numpy_from_packed`.

    Args:
        rgb_list_np: List of RGB arrays, one per camera; each shape (N, H, W, C).
        joint_pos_sim_np: Joint positions in sim order, shape (N, num_joints).
        task_description: Language instruction for the policy.
        policy_config: Closed-loop config (target_image_size, etc.).
        robot_state_joints_config: State joint name->index for sim order.
        policy_joints_config: Policy group name->list of joint names.
        modality_configs: Dict with "language", "video", "state" modality configs.

    Returns:
        Nested dict "language" / "video" / "state" with keys from modality
        config and arrays shaped for GR00T (e.g. video: N, 1, H, W, C).
    """
    target_image_size = getattr(policy_config, "target_image_size", None)
    if target_image_size is not None:
        rgb_list_np = resize_rgb_for_policy(rgb_list_np=rgb_list_np, target_image_size=target_image_size)
    joint_pos_state_policy = remap_sim_joints_to_policy_joints_from_np(
        joint_pos_sim_np, robot_state_joints_config, policy_joints_config
    )
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
        if state_key in joint_pos_state_policy:
            arr = joint_pos_state_policy[state_key]
            assert (
                arr.shape[0] == num_envs
            ), f"joint_pos_state_policy[{state_key}] has shape {arr.shape} but expected ({num_envs}, -1)"
            policy_observations["state"][state_key] = arr.reshape(num_envs, 1, -1)

    return policy_observations


# --------------------------------------------------------------------------- #
# Core SSOT: action building (numpy logic + single torch boundary)
# --------------------------------------------------------------------------- #


def build_gr00t_action_np(
    robot_action_policy: dict[str, Any],
    task_mode: TaskMode,
    policy_joints_config: dict[str, Any],
    robot_action_joints_config: dict[str, Any],
    embodiment_tag: str = "NEW_EMBODIMENT",
) -> np.ndarray:
    """Build the full action vector in numpy from GR00T policy output.

    Remaps policy joint names to sim order and, for G1_LOCOMANIPULATION,
    concatenates WBC commands (navigate, base height, torso RPY). Torso RPY
    is zeroed (GR00T output does not include it). No torch; use
    :func:`action_numpy_to_tensor` to convert to tensor.

    Args:
        robot_action_policy: Policy output dict (joint names, and for
            G1_LOCOMANIPULATION: navigate_command, base_height_command).
        task_mode: Determines whether to add WBC command dimensions.
        policy_joints_config: Policy joint name -> config.
        robot_action_joints_config: Sim action joint config (name->index).
        embodiment_tag: Robot embodiment for joint name mapping.

    Returns:
        (N, horizon, action_dim) float64 numpy array.
    """
    joints_sim_np = remap_policy_joints_to_sim_joints_np(
        robot_action_policy,
        policy_joints_config,
        robot_action_joints_config,
        embodiment_tag=embodiment_tag,
    )

    if task_mode == TaskMode.G1_LOCOMANIPULATION:
        # NOTE(xinjieyao, 2025-09-29): GR00T output does not include
        # torso_orientation_rpy_command; use zeros.
        nav = np.asarray(robot_action_policy["navigate_command"], dtype=np.float64)
        base_h = np.asarray(robot_action_policy["base_height_command"], dtype=np.float64)
        torso_rpy = np.zeros_like(nav, dtype=np.float64)
        return np.concatenate([joints_sim_np, nav, base_h, torso_rpy], axis=2)
    elif task_mode in (TaskMode.GR1_TABLETOP_MANIPULATION, TaskMode.DROID_MANIPULATION):
        return joints_sim_np
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")


def build_gr00t_action_tensor(
    robot_action_policy: dict[str, Any],
    task_mode: TaskMode,
    policy_joints_config: dict[str, Any],
    robot_action_joints_config: dict[str, Any],
    device: str | torch.device,
    embodiment_tag: str = "NEW_EMBODIMENT",
) -> torch.Tensor:
    """Convert GR00T policy output dict to a single torch action tensor.

    Delegates to numpy-based :func:`build_gr00t_action_np` then converts to
    tensor at the single boundary :func:`action_numpy_to_tensor`.

    Args:
        robot_action_policy: Policy output dict (joint names, and for
            G1_LOCOMANIPULATION: navigate_command, base_height_command).
        task_mode: Determines whether to add WBC command dimensions.
        policy_joints_config: Policy joint name -> config.
        robot_action_joints_config: Sim action joint config for remapping.
        device: Target device for the returned tensor.
        embodiment_tag: Robot embodiment for joint name mapping (optional).

    Returns:
        Action tensor of shape (N, horizon, action_dim) on the given device.

    Raises:
        ValueError: If task_mode is not supported.
    """
    action_np = build_gr00t_action_np(
        robot_action_policy,
        task_mode,
        policy_joints_config,
        robot_action_joints_config,
        embodiment_tag=embodiment_tag,
    )
    return to_tensor(action_np, device)
