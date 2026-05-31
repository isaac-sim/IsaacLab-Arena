# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSC-specific reward terms for NIST gear insertion.

These terms consume public state from :class:`NistGearInsertionOscAction`, such as
filtered actions, command deltas, smoothed force, and the auxiliary success
prediction channel.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.tasks.nist_gear_insertion import geometry as gear_geometry
from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_action import get_nist_gear_insertion_arm_action


@configclass
class NistGearInsertionOscRewardsCfg:
    """Config group for OSC-specific gear insertion rewards.

    Environments pass scene-specific asset names and insertion geometry so this
    OSC reward group remains reusable across gear-insertion scenes.
    """

    action_magnitude_penalty: RewardTermCfg = MISSING
    action_delta_penalty: RewardTermCfg = MISSING
    contact_penalty: RewardTermCfg = MISSING
    success_prediction_error: RewardTermCfg = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float] | tuple[float, ...],
        held_gear_base_offset: list[float] | tuple[float, ...],
        gear_peg_height: float,
        success_z_fraction: float,
        xy_threshold: float,
    ):
        """Initialize OSC reward terms with shared insertion geometry."""
        geometry_params = {
            "gear_cfg": SceneEntityCfg(gear_name),
            "board_cfg": SceneEntityCfg(board_name),
            "peg_offset": list(peg_offset),
            "held_gear_base_offset": list(held_gear_base_offset),
            "gear_peg_height": gear_peg_height,
            "xy_threshold": xy_threshold,
        }

        self.action_magnitude_penalty = RewardTermCfg(
            func=osc_action_magnitude_penalty,
            weight=-0.0005,
            params={},
        )
        self.action_delta_penalty = RewardTermCfg(
            func=osc_action_delta_penalty,
            weight=-0.01,
            params={},
        )
        self.contact_penalty = RewardTermCfg(
            func=wrist_contact_force_penalty,
            weight=-0.001,
            params={},
        )
        self.success_prediction_error = RewardTermCfg(
            func=success_prediction_error,
            weight=-1.0,
            params={
                **geometry_params,
                "success_z_fraction": success_z_fraction,
                "delay_until_ratio": 0.25,
            },
        )


class osc_action_magnitude_penalty(ManagerTermBase):
    """Penalize large asset-relative position and yaw commands."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = get_nist_gear_insertion_arm_action(env)
        position_thresholds = action_term.position_thresholds.clamp_min(1.0e-6)
        yaw_thresholds = action_term.rotation_thresholds[:, 2].clamp_min(1.0e-6)
        pos_error = torch.norm(action_term.delta_pos / position_thresholds, p=2, dim=-1)
        rot_error = torch.abs(action_term.delta_yaw) / yaw_thresholds
        return pos_error + rot_error


class osc_action_delta_penalty(ManagerTermBase):
    """Penalize jerky actions using smoothed action deltas."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = get_nist_gear_insertion_arm_action(env)
        return torch.norm(
            action_term.smoothed_actions - action_term.previous_smoothed_actions,
            p=2,
            dim=-1,
        )


class wrist_contact_force_penalty(ManagerTermBase):
    """Penalize wrist/contact force magnitude above per-episode threshold."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = get_nist_gear_insertion_arm_action(env)
        force_mag = torch.norm(action_term.smoothed_force, p=2, dim=-1)
        return torch.nn.functional.relu(force_mag - action_term.contact_thresholds)


class success_prediction_error(ManagerTermBase):
    """Penalize incorrect success predictions from the seventh action channel."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prediction_loss_weight = 0.0
        self._held_gear_base_offset_values = tuple(cfg.params["held_gear_base_offset"])
        self._held_gear_base_offset = torch.tensor(
            self._held_gear_base_offset_values, device=env.device, dtype=torch.float32
        )
        self._peg_offset_values = tuple(cfg.params["peg_offset"])
        self._peg_offset = torch.tensor(self._peg_offset_values, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg,
        board_cfg: SceneEntityCfg,
        gear_peg_height: float,
        success_z_fraction: float,
        xy_threshold: float,
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        delay_until_ratio: float = 0.25,
    ) -> torch.Tensor:
        true_success = self._compute_true_success(
            env,
            gear_cfg,
            board_cfg,
            peg_offset,
            held_gear_base_offset,
            gear_peg_height,
            success_z_fraction,
            xy_threshold,
        )
        self._update_prediction_loss_weight(true_success, delay_until_ratio)
        pred = self._read_success_prediction(env)
        return self._compute_prediction_error(true_success, pred)

    def _compute_true_success(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg,
        board_cfg: SceneEntityCfg,
        peg_offset: list[float] | None,
        held_gear_base_offset: list[float] | None,
        gear_peg_height: float,
        success_z_fraction: float,
        xy_threshold: float,
    ) -> torch.Tensor:
        """Return the geometric success label used by the auxiliary prediction loss."""
        return gear_geometry.compute_gear_insertion_success(
            env,
            gear_cfg,
            board_cfg,
            gear_geometry.resolve_offset_tensor(peg_offset, self._peg_offset_values, self._peg_offset, env.device),
            gear_geometry.resolve_offset_tensor(
                held_gear_base_offset,
                self._held_gear_base_offset_values,
                self._held_gear_base_offset,
                env.device,
            ),
            gear_peg_height,
            success_z_fraction,
            xy_threshold,
        )

    def _update_prediction_loss_weight(self, true_success: torch.Tensor, delay_until_ratio: float) -> None:
        """Enable the auxiliary loss once enough environments have reached success."""
        if true_success.float().mean() >= delay_until_ratio:
            self._prediction_loss_weight = 1.0

    def _read_success_prediction(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Read the success-probability prediction from the OSC action term."""
        arm_osc_action = get_nist_gear_insertion_arm_action(env)
        return (arm_osc_action.success_prediction + 1.0) / 2.0

    def _compute_prediction_error(self, true_success: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Return the gated auxiliary success-prediction error."""
        return torch.abs(true_success.float() - pred) * self._prediction_loss_weight
