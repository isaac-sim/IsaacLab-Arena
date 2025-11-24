# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def normalize_value(value: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Normalize a value to the range [0, 1] given min and max bounds."""
    return (value - min_value) / (max_value - min_value)


def unnormalize_value(value: float, min_value: float, max_value: float) -> float:
    """Unnormalize a value from [0, 1] back to the original range."""
    return min_value + (max_value - min_value) * value


def get_normalized_joint_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the normalized position of a joint (in range [0, 1])."""
    articulation = env.scene.articulations[asset_cfg.name]
    assert len(asset_cfg.joint_names) == 1, "Only one joint name is supported for now."
    joint_index = articulation.data.joint_names.index(asset_cfg.joint_names[0])
    joint_position = articulation.data.joint_pos[:, joint_index]
    joint_position_limits = articulation.data.joint_pos_limits[0, joint_index, :]
    joint_min, joint_max = joint_position_limits[0], joint_position_limits[1]
    normalized_position = normalize_value(joint_position, joint_min, joint_max)
    if joint_min < 0.0:
        normalized_position = 1 - normalized_position
    return normalized_position


def set_normalized_joint_position(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, target_joint_position: float, env_ids: torch.Tensor | None = None
) -> None:
    """Set the position of a joint using a normalized value (in range [0, 1])."""
    articulation = env.scene.articulations[asset_cfg.name]
    assert len(asset_cfg.joint_names) == 1, "Only one joint name is supported for now."
    joint_index = articulation.data.joint_names.index(asset_cfg.joint_names[0])
    joint_position_limits = articulation.data.joint_pos_limits[0, joint_index, :]
    joint_min, joint_max = joint_position_limits[0], joint_position_limits[1]
    if joint_min < 0.0:
        target_joint_position = 1 - target_joint_position
    target_joint_position_unnormlized = unnormalize_value(target_joint_position, joint_min, joint_max)
    articulation.write_joint_position_to_sim(
        torch.tensor([[target_joint_position_unnormlized]]).to(env.device),
        torch.tensor([joint_index]).to(env.device),
        env_ids=env_ids.to(env.device) if env_ids is not None else None,
    )
