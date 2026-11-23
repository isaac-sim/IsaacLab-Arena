# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gear Assembly reward terms with Arena-compatible aliases."""

from __future__ import annotations

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.rewards import (  # noqa: F401
    keypoint_ee_grasp_error,
    keypoint_ee_grasp_error_exp,
    keypoint_entity_error,
    keypoint_entity_error_exp,
)


def _normalize_ee_threshold_param(cfg: RewardTermCfg) -> None:
    if "ee_gear_threshold" in cfg.params and "ee_grasp_threshold" not in cfg.params:
        cfg.params["ee_grasp_threshold"] = cfg.params["ee_gear_threshold"]


class keypoint_ee_gear_error(keypoint_ee_grasp_error):
    """Compatibility alias for the source grasp-corrected EE/gear keypoint penalty."""

    def __init__(self, cfg: RewardTermCfg, env):
        _normalize_ee_threshold_param(cfg)
        super().__init__(cfg, env)

    def __call__(
        self,
        env,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        end_effector_body_name: str = "",
        grasp_rot_offset: list | None = None,
        gear_offsets_grasp: dict | None = None,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
        weight_ramp_start: float = 0.0,
        weight_ramp_steps: int = 1,
        ee_grasp_threshold: float = 0.0,
        ee_gear_threshold: float | None = None,
    ) -> torch.Tensor:
        if self.eef_idx is None:
            return torch.zeros(env.num_envs, device=env.device)

        eef_pos, eef_quat, gear_grasp_pos, gear_quat_grasp = self._get_grasp_corrected_target(env)
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=eef_pos,
            current_quat=eef_quat,
            target_pos=gear_grasp_pos,
            target_quat=gear_quat_grasp,
            keypoint_scale=keypoint_scale,
        )
        mean_kp_error = keypoint_dist_sep.mean(-1)
        threshold = ee_grasp_threshold if ee_gear_threshold is None else ee_gear_threshold
        is_active = (mean_kp_error > threshold).float()
        weight_scale = self._get_weight_scale(env)
        scaled_reward = mean_kp_error * weight_scale * is_active

        _log_extra(env, "ee_grasp_kp_error/mean_keypoint_dist", mean_kp_error.mean().item())
        _log_extra(env, "ee_grasp_kp_error/pct_envs_active", is_active.mean().item())
        _log_extra(env, "ee_grasp_kp_error/weight_scale", weight_scale)
        return scaled_reward


class keypoint_ee_gear_error_exp(keypoint_ee_grasp_error_exp):
    """Compatibility alias for the source grasp-corrected exponential EE/gear reward."""

    def __init__(self, cfg: RewardTermCfg, env):
        _normalize_ee_threshold_param(cfg)
        super().__init__(cfg, env)

    def __call__(
        self,
        env,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        end_effector_body_name: str = "",
        grasp_rot_offset: list | None = None,
        gear_offsets_grasp: dict | None = None,
        kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
        weight_ramp_start: float = 0.0,
        weight_ramp_steps: int = 1,
        ee_grasp_threshold: float = 0.0,
        ee_gear_threshold: float | None = None,
    ) -> torch.Tensor:
        if self.eef_idx is None:
            return torch.zeros(env.num_envs, device=env.device)

        eef_pos, eef_quat, gear_grasp_pos, gear_quat_grasp = self._get_grasp_corrected_target(env)
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=eef_pos,
            current_quat=eef_quat,
            target_pos=gear_grasp_pos,
            target_quat=gear_quat_grasp,
            keypoint_scale=keypoint_scale,
        )
        mean_kp_error = keypoint_dist_sep.mean(-1)
        threshold = ee_grasp_threshold if ee_gear_threshold is None else ee_gear_threshold
        is_active = (mean_kp_error > threshold).float()

        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])
        if kp_use_sum_of_exps:
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (
                    1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
                ).mean(-1)
        else:
            kp_dist_mean = keypoint_dist_sep.mean(-1)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1.0 / (torch.exp(a * kp_dist_mean) + b + torch.exp(-a * kp_dist_mean))

        weight_scale = self._get_weight_scale(env)
        scaled_reward = keypoint_reward_exp * weight_scale * is_active

        _log_extra(env, "ee_grasp_kp_error_exp/mean_keypoint_dist", mean_kp_error.mean().item())
        _log_extra(env, "ee_grasp_kp_error_exp/mean_exp_reward", keypoint_reward_exp.mean().item())
        _log_extra(env, "ee_grasp_kp_error_exp/pct_envs_active", is_active.mean().item())
        _log_extra(env, "ee_grasp_kp_error_exp/weight_scale", weight_scale)
        return scaled_reward


def _log_extra(env, key: str, value: float) -> None:
    if not hasattr(env, "extras"):
        env.extras = {}
    env.extras.setdefault("log", {})[key] = value
