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
# Core SSOT logic (numpy-based, modality-config driven)
# --------------------------------------------------------------------------- #


def build_gr00t_policy_inputs_np(
    rgb_np: np.ndarray,  # (N, H, W, C)
    joint_pos_sim_np: np.ndarray,  # (N, num_joints)
    task_description: str,
    policy_config: Gr00tClosedloopPolicyConfig,
    robot_state_joints_config: dict[str, Any],
    policy_joints_config: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    """Convert numpy observations to numpy GR00T policy inputs using modality config."""
    num_envs = rgb_np.shape[0]

    # Resize RGB frames if needed
    if rgb_np.shape[1:3] != tuple(policy_config.target_image_size[:2]):
        rgb_np = resize_frames_with_padding(
            rgb_np,
            target_image_size=policy_config.target_image_size,
            bgr_conversion=False,
            pad_img=True,
        )

    # Use existing JointsAbsPosition / remap helpers by temporarily going through torch
    joint_pos_sim_t = torch.from_numpy(joint_pos_sim_np)
    joint_pos_state_sim = JointsAbsPosition(joint_pos_sim_t, robot_state_joints_config)
    joint_pos_state_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, policy_joints_config)

    # Extract modality keys
    language_keys = modality_configs["language"].modality_keys
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys

    # Dynamically construct policy observations using modality config keys
    # TODO(xinejiayao, 2025-12-10): when multi-task with parallel envs feature is enabled,
    # we need to pass in a list of task descriptions.
    policy_observations: dict[str, Any] = {
        "language": {language_keys[0]: [[task_description] for _ in range(num_envs)]},
        "video": {
            video_keys[0]: rgb_np.reshape(
                num_envs,
                1,
                policy_config.target_image_size[0],
                policy_config.target_image_size[1],
                policy_config.target_image_size[2],
            )
        },
        "state": {},
    }

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
    elif task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
        action_tensor = robot_action_sim.get_joints_pos()
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    return action_tensor
