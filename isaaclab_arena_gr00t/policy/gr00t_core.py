# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
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
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    load_gr00t_modality_config_from_file,
    load_robot_joints_config_from_yaml,
)
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints,
    remap_sim_joints_to_policy_joints,
)
from isaaclab_arena_gr00t.utils.robot_joints import JointsAbsPosition

# --------------------------------------------------------------------------- #
# Base config dataclass (shared by local and remote policies)
# --------------------------------------------------------------------------- #


@dataclass
class Gr00tBasePolicyArgs:
    """Base configuration for GR00T policies (shared by local and remote).

    Child classes can extend this with additional fields like num_envs.
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
            "help": "Device to use for the policy-related operations.",
        },
    )


# --------------------------------------------------------------------------- #
# Config / model / joints helpers (backend-agnostic)
# --------------------------------------------------------------------------- #


def load_gr00t_closedloop_config(args: Gr00tBasePolicyArgs) -> Gr00tClosedloopPolicyConfig:
    """Load the closed-loop policy config from YAML."""
    return create_config_from_yaml(args.policy_config_yaml_path, Gr00tClosedloopPolicyConfig)


def load_gr00t_policy_from_config(policy_config: Gr00tClosedloopPolicyConfig) -> Gr00tPolicy:
    """Load a Gr00tPolicy from the closed-loop config."""
    assert Path(policy_config.model_path).exists(), f"Dataset path {policy_config.dataset_path} does not exist"

    return Gr00tPolicy(
        model_path=policy_config.model_path,
        embodiment_tag=EmbodimentTag[policy_config.embodiment_tag],
        device=policy_config.policy_device,
        strict=True,
    )


def load_gr00t_joint_configs(
    policy_config: Gr00tClosedloopPolicyConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load policy / action / state joint configs."""
    policy_joints_config = load_robot_joints_config_from_yaml(policy_config.policy_joints_config_path)
    robot_action_joints_config = load_robot_joints_config_from_yaml(policy_config.action_joints_config_path)
    robot_state_joints_config = load_robot_joints_config_from_yaml(policy_config.state_joints_config_path)
    return policy_joints_config, robot_action_joints_config, robot_state_joints_config


def compute_action_dim(task_mode: TaskMode, robot_action_joints_config: dict[str, Any]) -> int:
    """Compute action dimension given task_mode and action joints configuration."""
    action_dim = len(robot_action_joints_config)
    if task_mode == TaskMode.G1_LOCOMANIPULATION:
        # WBC commands: navigate_command, base_height_command, torso_orientation_rpy_command
        action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
    return action_dim


def load_gr00t_modality_config(policy_config: Gr00tClosedloopPolicyConfig) -> dict[str, Any]:
    """Load GR00T modality config and return modality configs."""
    return load_gr00t_modality_config_from_file(
        policy_config.modality_config_path,
        policy_config.embodiment_tag,
    )


# --------------------------------------------------------------------------- #
# Explicit data-format conversion helpers
# --------------------------------------------------------------------------- #


def extract_obs_numpy_from_torch(
    observation: dict[str, Any],
    camera_names: list[str],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convert torch environment observation to numpy arrays.

    This is the single explicit torch-to-numpy boundary for the local
    (one-docker) pipeline.  The remote (client-server) pipeline receives
    numpy directly over the socket and does not need this conversion.

    Returns:
        rgb_list_np: List of (N, H, W, C) uint8 numpy arrays, one per camera.
        joint_pos_sim_np: (N, num_joints) float numpy array.
    """
    assert "camera_obs" in observation, "camera_obs is not in observation"

    rgb_list_np: list[np.ndarray] = []
    for cam in camera_names:
        assert cam in observation["camera_obs"], f"camera {cam} is not in camera_obs"
        rgb_list_np.append(observation["camera_obs"][cam].detach().cpu().numpy())

    joint_pos_sim_np: np.ndarray = observation["policy"]["robot_joint_pos"].detach().cpu().numpy()
    return rgb_list_np, joint_pos_sim_np


def extract_obs_numpy_from_packed(
    packed_observation: dict[str, Any],
    camera_names: list[str],
    unpack_fn: Any,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract numpy arrays from a flat/packed observation dict (remote pipeline).

    The remote (client-server) pipeline receives numpy arrays serialised
    over the socket.  This helper unpacks them into the same
    ``(rgb_list_np, joint_pos_sim_np)`` tuple used by the core logic so
    that both pipelines converge on the same interface.

    Returns:
        rgb_list_np: List of (N, H, W, C) numpy arrays, one per camera.
        joint_pos_sim_np: (N, num_joints) float numpy array.
    """
    nested_obs = unpack_fn(packed_observation)
    assert "camera_obs" in nested_obs, "camera_obs is not in observation"

    rgb_list_np: list[np.ndarray] = []
    for cam in camera_names:
        assert cam in nested_obs["camera_obs"], f"camera {cam} is not in camera_obs"
        rgb_list_np.append(nested_obs["camera_obs"][cam])

    joint_pos_sim_np: np.ndarray = nested_obs["policy"]["robot_joint_pos"]
    return rgb_list_np, joint_pos_sim_np


# --------------------------------------------------------------------------- #
# Core SSOT logic (numpy-based, modality-config driven)
# --------------------------------------------------------------------------- #


def build_gr00t_policy_inputs_np(
    rgb_list_np: list[np.ndarray],  # list of (N, H, W, C) arrays, one per camera
    joint_pos_sim_np: np.ndarray,  # (N, num_joints)
    task_description: str,
    policy_config: Gr00tClosedloopPolicyConfig,
    robot_state_joints_config: dict[str, Any],
    policy_joints_config: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Convert numpy observations to numpy GR00T policy inputs using modality config.

    Args:
        rgb_list_np: List of RGB arrays, one per camera. Each has shape (N, H, W, C).
    """
    num_envs = rgb_list_np[0].shape[0]

    # Resize RGB frames if needed
    processed_rgb_list: list[np.ndarray] = []
    for rgb_np in rgb_list_np:
        if rgb_np.shape[1:3] != tuple(policy_config.target_image_size[:2]):
            rgb_np = resize_frames_with_padding(
                rgb_np,
                target_image_size=policy_config.target_image_size,
                bgr_conversion=False,
                pad_img=True,
            )
        processed_rgb_list.append(rgb_np)

    # Use existing JointsAbsPosition / remap helpers by temporarily going through torch
    joint_pos_sim_t = torch.from_numpy(joint_pos_sim_np)
    joint_pos_state_sim = JointsAbsPosition(joint_pos_sim_t, robot_state_joints_config)
    joint_pos_state_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, policy_joints_config)

    # Extract modality keys
    language_keys = modality_configs["language"].modality_keys
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys

    assert len(video_keys) == len(
        processed_rgb_list
    ), f"number of video keys ({len(video_keys)}) and rgb inputs ({len(processed_rgb_list)}) must match"

    # Dynamically construct policy observations using modality config keys
    # TODO(xinejiayao, 2025-12-10): when multi-task with parallel envs feature is enabled,
    # we need to pass in a list of task descriptions.
    policy_observations: dict[str, Any] = {
        "language": {language_keys[0]: [[task_description] for _ in range(num_envs)]},
        "video": {},
        "state": {},
    }

    for i, video_key in enumerate(video_keys):
        policy_observations["video"][video_key] = processed_rgb_list[i].reshape(
            num_envs,
            1,
            policy_config.target_image_size[0],
            policy_config.target_image_size[1],
            policy_config.target_image_size[2],
        )

    # Dynamically populate state keys from modality config
    for state_key in state_keys:
        if state_key in joint_pos_state_policy:
            policy_observations["state"][state_key] = joint_pos_state_policy[state_key].reshape(num_envs, 1, -1)

    return policy_observations


def build_gr00t_action_tensor(
    robot_action_policy: dict[str, Any],
    task_mode: TaskMode,
    policy_joints_config: dict[str, Any],
    robot_action_joints_config: dict[str, Any],
    device: str | torch.device,
) -> torch.Tensor:
    """Convert GR00T outputs to torch action tensor (N, horizon, action_dim)."""
    robot_action_sim = remap_policy_joints_to_sim_joints(
        robot_action_policy,
        policy_joints_config,
        robot_action_joints_config,
        device,
    )

    if task_mode == TaskMode.G1_LOCOMANIPULATION:
        # NOTE(xinjieyao, 2025-09-29): GR00T output dim=32, does not fit the entire action space,
        # including torso_orientation_rpy_command. Manually set it to 0.
        torso_orientation_rpy_command = torch.zeros(
            robot_action_policy["navigate_command"].shape, dtype=torch.float, device=device
        )
        action_tensor = torch.cat(
            [
                robot_action_sim.get_joints_pos(),
                torch.tensor(robot_action_policy["navigate_command"], dtype=torch.float, device=device),
                torch.tensor(robot_action_policy["base_height_command"], dtype=torch.float, device=device),
                torso_orientation_rpy_command,
            ],
            axis=2,
        )
    elif task_mode in (TaskMode.GR1_TABLETOP_MANIPULATION, TaskMode.DROID_MANIPULATION):
        action_tensor = robot_action_sim.get_joints_pos()
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    return action_tensor
