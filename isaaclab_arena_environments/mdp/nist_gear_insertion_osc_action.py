# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSC torque action term for NIST gear insertion: asset-relative commands, EMA smoothing,
roll/pitch lock, target clipping, and success prediction (7-D policy output).
"""

from __future__ import annotations

import math
import torch
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _wrap_yaw(angle: torch.Tensor) -> torch.Tensor:
    """Map yaw angles so the Franka joint-limit discontinuity (~-135 deg) is avoided."""
    return torch.where(angle > math.radians(235), angle - 2 * math.pi, angle)


def _randomize_gains(
    default_values: torch.Tensor,
    noise_levels: tuple[float, ...],
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    """Multiplicative gain randomization for position/rotation clip thresholds."""
    ndim = default_values.shape[-1]
    noise = torch.rand(num_envs, ndim, device=device) * torch.tensor(noise_levels, device=device)
    multiplier = 1.0 + noise
    decrease = torch.rand(num_envs, ndim, device=device) > 0.5
    return default_values * torch.where(decrease, 1.0 / multiplier, multiplier)


class NistGearInsertionOscAction(OperationalSpaceControllerAction):
    """Operational-space torque control for peg-style insertion with a 7-D policy.

    3 position + 3 rotation + 1 success prediction. Asset-relative position, roll/pitch
    locked, EMA smoothing, target clipping. Layout follows common assembly RL benchmarks.
    """

    cfg: NistGearInsertionOscActionCfg

    def __init__(self, cfg: NistGearInsertionOscActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._smoothed_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self.ema_factor = torch.full((self.num_envs, 1), 0.05, device=self.device)
        self._pos_bounds = torch.tensor(cfg.pos_action_bounds, device=self.device)

        _pt = torch.tensor(cfg.pos_action_threshold, device=self.device)
        _rt = torch.tensor(cfg.rot_action_threshold, device=self.device)
        self._default_pos_thresh = _pt.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._default_rot_thresh = _rt.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._pos_thresh = self._default_pos_thresh.clone()
        self._rot_thresh = self._default_rot_thresh.clone()

        self._peg_offset = torch.tensor(cfg.peg_offset, device=self.device)
        self._fixed_pos_noise_levels = torch.tensor(cfg.fixed_pos_noise_levels, device=self.device)
        self.fixed_pos_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self.contact_thresholds = torch.full((self.num_envs,), 7.5, device=self.device)
        self.force_smooth = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_smoothed_actions = torch.zeros(self.num_envs, 7, device=self.device)

        self.delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.success_pred = torch.full((self.num_envs,), -1.0, device=self.device)
        self._pos_dead_zone = torch.tensor(cfg.pos_dead_zone, device=self.device).unsqueeze(0)
        self._rot_dead_zone = cfg.rot_dead_zone

    def _get_bolt_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Peg tip in env frame from fixed asset pose + local peg offset."""
        origins = self._env.scene.env_origins
        board = self._env.scene[self.cfg.fixed_asset_name]
        pos = wp.to_torch(board.data.root_pos_w) - origins
        quat = wp.to_torch(board.data.root_quat_w)
        offset = self._peg_offset.unsqueeze(0).expand(pos.shape[0], 3)
        peg_pos = pos + math_utils.quat_apply(quat, offset)
        if env_ids is not None:
            return peg_pos[env_ids]
        return peg_pos

    @property
    def action_dim(self) -> int:
        return 7

    def apply_actions(self):
        super().apply_actions()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._prev_smoothed_actions[:] = self._smoothed_actions
        self._smoothed_actions[:] = (
            self.ema_factor * actions + (1.0 - self.ema_factor) * self._smoothed_actions
        )
        self.success_pred[:] = self._smoothed_actions[:, 6]

        self._compute_ee_pose()
        ee_pos_b = self._ee_pose_b[:, :3]
        ee_quat_b = self._ee_pose_b[:, 3:7]

        bolt_pos_b = self._get_bolt_pos() + self.fixed_pos_noise

        pos_actions = self._smoothed_actions[:, :3]
        target_pos = bolt_pos_b + pos_actions * self._pos_bounds

        self.delta_pos[:] = target_pos - ee_pos_b
        clipped_delta = torch.clamp(self.delta_pos, -self._pos_thresh, self._pos_thresh)
        clipped_delta = torch.where(
            torch.abs(clipped_delta) > self._pos_dead_zone,
            clipped_delta,
            torch.zeros_like(clipped_delta),
        )
        final_pos = ee_pos_b + clipped_delta

        rot_actions = self._smoothed_actions[:, 3:6].clone()
        rot_actions[:, 0:2] = 0.0

        yaw_rad = math.radians(-180.0) + math.radians(270.0) * (rot_actions[:, 2] + 1.0) / 2.0
        zero = torch.zeros_like(yaw_rad)
        bolt_quat = math_utils.quat_from_euler_xyz(zero, zero, yaw_rad)
        pi_t = torch.full_like(yaw_rad, math.pi)
        flip_quat = math_utils.quat_from_euler_xyz(pi_t, zero, zero)
        target_quat = math_utils.quat_mul(flip_quat, bolt_quat)

        curr_roll, curr_pitch, curr_yaw = math_utils.euler_xyz_from_quat(ee_quat_b, wrap_to_2pi=True)
        desired_roll, desired_pitch, desired_yaw = math_utils.euler_xyz_from_quat(target_quat, wrap_to_2pi=True)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        curr_yaw = _wrap_yaw(curr_yaw)
        desired_yaw = _wrap_yaw(desired_yaw)

        self.delta_yaw[:] = desired_yaw - curr_yaw
        clipped_yaw = torch.clamp(self.delta_yaw, -self._rot_thresh[:, 2], self._rot_thresh[:, 2])
        clipped_yaw = torch.where(
            torch.abs(clipped_yaw) > self._rot_dead_zone,
            clipped_yaw,
            torch.zeros_like(clipped_yaw),
        )
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * math.pi, desired_roll)
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clamp(delta_roll, -self._rot_thresh[:, 0], self._rot_thresh[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch_w = torch.where(curr_pitch > math.pi, curr_pitch - 2 * math.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * math.pi, desired_pitch)
        desired_pitch_w = torch.where(desired_pitch > math.pi, desired_pitch - 2 * math.pi, desired_pitch)
        delta_pitch = desired_pitch_w - curr_pitch_w
        clipped_pitch = torch.clamp(delta_pitch, -self._rot_thresh[:, 1], self._rot_thresh[:, 1])
        desired_xyz[:, 1] = curr_pitch_w + clipped_pitch

        final_quat = math_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2],
        )

        self._processed_actions[:, :3] = final_pos
        self._processed_actions[:, 3:7] = final_quat

        self._compute_task_frame_pose()
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=self._task_frame_pose_b,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or (hasattr(env_ids, '__len__') and len(env_ids) == 0):
            return

        n = len(env_ids)

        lo, hi = self.cfg.ema_factor_range
        self.ema_factor[env_ids] = lo + torch.rand(n, 1, device=self.device) * (hi - lo)

        self._pos_thresh[env_ids] = _randomize_gains(
            self._default_pos_thresh[env_ids], self.cfg.pos_threshold_noise_level, n, self.device,
        )
        self._rot_thresh[env_ids] = _randomize_gains(
            self._default_rot_thresh[env_ids], self.cfg.rot_threshold_noise_level, n, self.device,
        )

        self.fixed_pos_noise[env_ids] = (
            torch.randn(n, 3, device=self.device) * self._fixed_pos_noise_levels
        )

        ct_lo, ct_hi = self.cfg.contact_threshold_range
        self.contact_thresholds[env_ids] = ct_lo + torch.rand(n, device=self.device) * (ct_hi - ct_lo)

        self.force_smooth[env_ids] = 0.0

        self._compute_ee_pose()
        ee_pos = self._ee_pose_b[env_ids, :3]
        ee_quat = self._ee_pose_b[env_ids, 3:7]

        bolt_pos = self._get_bolt_pos(env_ids) + self.fixed_pos_noise[env_ids]

        pos_actions = (ee_pos - bolt_pos) / self._pos_bounds
        self._smoothed_actions[env_ids, 0:3] = pos_actions

        unrot_pi = math_utils.quat_from_euler_xyz(
            torch.full((n,), -math.pi, device=self.device),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
        )
        quat_rel_bolt = math_utils.quat_mul(unrot_pi, ee_quat)
        yaw_bolt = math_utils.euler_xyz_from_quat(quat_rel_bolt, wrap_to_2pi=True)[2]
        yaw_bolt = torch.where(yaw_bolt > math.pi / 2, yaw_bolt - 2 * math.pi, yaw_bolt)
        yaw_bolt = torch.where(yaw_bolt < -math.pi, yaw_bolt + 2 * math.pi, yaw_bolt)
        yaw_action = (yaw_bolt + math.radians(180.0)) / math.radians(270.0) * 2.0 - 1.0

        self._smoothed_actions[env_ids, 3:5] = 0.0
        self._smoothed_actions[env_ids, 5] = yaw_action
        self._smoothed_actions[env_ids, 6] = -1.0

        self._prev_smoothed_actions[env_ids] = self._smoothed_actions[env_ids]
        self.delta_pos[env_ids] = 0.0
        self.delta_yaw[env_ids] = 0.0
        self.success_pred[env_ids] = -1.0


@configclass
class NistGearInsertionOscActionCfg(OperationalSpaceControllerActionCfg):
    """Config for :class:`NistGearInsertionOscAction`."""

    class_type: type[ActionTerm] = NistGearInsertionOscAction

    fixed_asset_name: str = "gears_and_base"
    peg_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_pos_noise_levels: tuple[float, float, float] = (0.001, 0.001, 0.001)
    pos_action_bounds: tuple[float, float, float] = (0.05, 0.05, 0.05)
    pos_action_threshold: tuple[float, float, float] = (0.02, 0.02, 0.02)
    rot_action_threshold: tuple[float, float, float] = (0.097, 0.097, 0.097)
    pos_threshold_noise_level: tuple[float, float, float] = (0.25, 0.25, 0.25)
    rot_threshold_noise_level: tuple[float, float, float] = (0.29, 0.29, 0.29)
    ema_factor_range: tuple[float, float] = (0.05, 0.2)
    contact_threshold_range: tuple[float, float] = (5.0, 10.0)
    # Dead zone: zero out small commanded deltas on the task wrench.
    pos_dead_zone: tuple[float, float, float] = (0.0005, 0.0005, 0.0005)  # m, ~0.5 mm
    rot_dead_zone: float = 0.001  # rad
