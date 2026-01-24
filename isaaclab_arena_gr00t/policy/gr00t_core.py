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

from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config
from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.utils.io_utils import load_robot_joints_config_from_yaml
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints,
    remap_sim_joints_to_policy_joints,
)
from isaaclab_arena_gr00t.utils.robot_joints import JointsAbsPosition


@dataclass
class Gr00tBasePolicyArgs:
    """Base configuration for GR00T policies (shared by local and remote)."""

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
# Config / model helpers (backend-agnostic)
# --------------------------------------------------------------------------- #


def load_gr00t_policy_from_config(policy_config: Gr00tClosedloopPolicyConfig) -> Gr00tPolicy:
    """Load a Gr00tPolicy from the closed-loop config."""
    if policy_config.data_config in DATA_CONFIG_MAP:
        data_config = DATA_CONFIG_MAP[policy_config.data_config]
    elif policy_config.data_config == "unitree_g1_sim_wbc":
        data_config = load_data_config(
            "isaaclab_arena_gr00t.embodiments.g1.g1_sim_wbc_data_config:UnitreeG1SimWBCDataConfig"
        )
    else:
        raise ValueError(f"Invalid data config: {policy_config.data_config}")

    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    model_path = Path(policy_config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    return Gr00tPolicy(
        model_path=str(model_path),
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=policy_config.embodiment_tag,
        denoising_steps=policy_config.denoising_steps,
        device=policy_config.policy_device,
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
        action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
    return action_dim


# --------------------------------------------------------------------------- #
# Core SSOT logic (numpy-based)
# --------------------------------------------------------------------------- #


def build_gr00t_policy_inputs_np(
    rgb_np: np.ndarray,  # (N, H, W, C)
    joint_pos_sim_np: np.ndarray,  # (N, num_joints)
    task_description: str,
    policy_config: Gr00tClosedloopPolicyConfig,
    robot_state_joints_config: dict[str, Any],
    policy_joints_config: dict[str, Any],
) -> dict[str, Any]:
    """Convert numpy observations to numpy GR00T policy inputs."""
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
    joint_pos_state_sim = JointsAbsPosition(joint_pos_sim_np, robot_state_joints_config)
    joint_pos_state_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, policy_joints_config)

    left_arm = joint_pos_state_policy["left_arm"].reshape(num_envs, 1, -1)
    right_arm = joint_pos_state_policy["right_arm"].reshape(num_envs, 1, -1)
    left_hand = joint_pos_state_policy["left_hand"].reshape(num_envs, 1, -1)
    right_hand = joint_pos_state_policy["right_hand"].reshape(num_envs, 1, -1)

    policy_inputs: dict[str, Any] = {
        # TODO(xinejiayao, 2025-12-10): when multi-task with parallel envs feature is enabled, we need to pass in a list of task descriptions.
        "annotation.human.task_description": [task_description] * num_envs,
        "video.ego_view": rgb_np.reshape(
            num_envs,
            1,
            policy_config.target_image_size[0],
            policy_config.target_image_size[1],
            policy_config.target_image_size[2],
        ),
        "state.left_arm": left_arm,
        "state.right_arm": right_arm,
        "state.left_hand": left_hand,
        "state.right_hand": right_hand,
    }
    # NOTE(xinjieyao, 2025-10-07): waist is not used in GR1 tabletop manipulation
    if TaskMode(policy_config.task_mode_name) == TaskMode.G1_LOCOMANIPULATION:
        waist = joint_pos_state_policy["waist"].reshape(num_envs, 1, -1)
        policy_inputs["state.waist"] = waist

    return policy_inputs


def build_gr00t_action_tensor(
    robot_action_policy: dict[str, Any],
    task_mode: TaskMode,
    policy_joints_config: dict[str, Any],
    robot_action_joints_config: dict[str, Any],
    device: str | torch.device,
) -> np.ndarray:
    """Convert numpy GR00T outputs to numpy action tensor (N, horizon, action_dim)."""

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
            robot_action_policy["action.navigate_command"].shape, dtype=torch.float, device=device
        )
        action_tensor = torch.cat(
            [
                robot_action_sim.get_joints_pos(),
                torch.tensor(robot_action_policy["action.navigate_command"], dtype=torch.float, device=device),
                torch.tensor(robot_action_policy["action.base_height_command"], dtype=torch.float, device=device),
                torso_orientation_rpy_command,
            ],
            axis=2,
        )
    elif task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
        action_tensor = robot_action_sim.get_joints_pos()
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    return action_tensor
