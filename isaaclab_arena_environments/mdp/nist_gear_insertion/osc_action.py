# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Operational-space action term for Franka gear insertion.

This module converts normalized policy commands into end-effector pose targets
for the OSC controller. The policy targets the insertion peg directly, while
this term applies the filtering, command limits, force smoothing, and reset-time
randomization used by the Isaac Lab Forge gear-insertion setup.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab_tasks.direct.factory.factory_utils import wrap_yaw
from isaaclab_tasks.direct.forge.forge_utils import get_random_prop_gains

from isaaclab_arena.tasks.nist_gear_insertion.geometry import compute_asset_local_offset_pos

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


ACTION_DIM = 7
POS_SLICE = slice(0, 3)
QUAT_SLICE = slice(3, 7)
ROLL_IDX = 3
YAW_IDX = 5
SUCCESS_IDX = 6
ROLL_PITCH_SLICE = slice(ROLL_IDX, YAW_IDX)
ROT_ROLL_IDX = 0
ROT_PITCH_IDX = 1
ROT_YAW_IDX = 2
# The trained policy uses a gripper-down orientation and controls only yaw.
# The interval avoids the wrist-frame discontinuity behind the end effector.
YAW_RANGE_RAD = math.radians(270.0)
YAW_MIN_RAD = -math.pi
YAW_MAX_RAD = math.pi / 2.0


def _action_to_target_yaw(action: torch.Tensor) -> torch.Tensor:
    """Map normalized policy action to the commanded yaw interval."""
    return YAW_MIN_RAD + YAW_RANGE_RAD * (action + 1.0) / 2.0


def _target_yaw_to_action(yaw: torch.Tensor) -> torch.Tensor:
    """Map commanded yaw back to the normalized policy interval."""
    return (yaw - YAW_MIN_RAD) / YAW_RANGE_RAD * 2.0 - 1.0


def _wrap_yaw_to_action_range(yaw: torch.Tensor) -> torch.Tensor:
    """Wrap yaw and clamp the excluded wrist sector to the policy interval."""
    yaw = torch.where(yaw > YAW_MAX_RAD, yaw - 2.0 * math.pi, yaw)
    return torch.clamp(yaw, YAW_MIN_RAD, YAW_MAX_RAD)


def _gripper_down_to_yaw_frame_quat(num_envs: int, device: torch.device) -> torch.Tensor:
    """Return the fixed rotation from the gripper-down frame to the yaw frame."""
    return math_utils.quat_from_euler_xyz(
        torch.full((num_envs,), -math.pi, device=device),
        torch.zeros(num_envs, device=device),
        torch.zeros(num_envs, device=device),
    )


def get_nist_gear_insertion_arm_action(
    env: ManagerBasedEnv,
    term_name: str = "arm_action",
) -> NistGearInsertionOscAction:
    """Return the NIST gear insertion OSC action term from an environment."""
    try:
        action_term = env.action_manager.get_term(term_name)
    except KeyError as exc:
        raise KeyError(f"Action term '{term_name}' is required for gear insertion.") from exc
    if not isinstance(action_term, NistGearInsertionOscAction):
        raise TypeError(
            f"Action term '{term_name}' must be {NistGearInsertionOscAction.__name__}; "
            f"got {type(action_term).__name__}."
        )
    return action_term


class NistGearInsertionOscAction(OperationalSpaceControllerAction):
    """Operational-space action term for the 7-D NIST gear insertion policy.

    The policy layout is ``xyz, roll, pitch, yaw, success``. Only ``xyz`` and
    ``yaw`` are sent to the OSC controller; roll and pitch are intentionally
    ignored for the gripper-down insertion strategy. The final channel is an
    auxiliary success prediction consumed by the OSC reward terms.
    """

    cfg: NistGearInsertionOscActionCfg

    def __init__(self, cfg: NistGearInsertionOscActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._smoothed_actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._ema_factor = torch.full((self.num_envs, 1), 0.05, device=self.device)
        self._position_action_bounds = torch.tensor(cfg.pos_action_bounds, device=self.device)

        pos_threshold = torch.tensor(cfg.pos_action_threshold, device=self.device)
        rot_threshold = torch.tensor(cfg.rot_action_threshold, device=self.device)
        self._default_position_step_limits = pos_threshold.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._default_rotation_step_limits = rot_threshold.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._position_step_limits = self._default_position_step_limits.clone()
        self._rotation_step_limits = self._default_rotation_step_limits.clone()

        self._peg_offset = torch.tensor(cfg.peg_offset, device=self.device)
        self._fixed_pos_noise_levels = torch.tensor(cfg.fixed_pos_noise_levels, device=self.device)
        self._fixed_pos_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self._contact_thresholds = torch.full((self.num_envs,), 7.5, device=self.device)
        self._smoothed_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_smoothed_actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)

        self._delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self._success_prediction = torch.full((self.num_envs,), -1.0, device=self.device)
        self._pos_dead_zone = torch.tensor(cfg.pos_dead_zone, device=self.device).unsqueeze(0)
        self._rot_dead_zone = cfg.rot_dead_zone
        self._force_body_idx: int | None = None
        self._force_smoothing_factor = cfg.force_smoothing_factor

    def _get_peg_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the peg target position in each environment frame."""
        board = self._env.scene[self.cfg.fixed_asset_name]
        peg_pos = compute_asset_local_offset_pos(self._env, board, self._peg_offset)
        if env_ids is not None:
            return peg_pos[env_ids]
        return peg_pos

    @property
    def action_dim(self) -> int:
        """Number of policy actions consumed by the OSC term."""
        return ACTION_DIM

    @property
    def smoothed_actions(self) -> torch.Tensor:
        """EMA-filtered policy actions used by observations and penalties."""
        return self._smoothed_actions

    @property
    def previous_smoothed_actions(self) -> torch.Tensor:
        """EMA-filtered policy actions from the previous environment step."""
        return self._prev_smoothed_actions

    @property
    def position_thresholds(self) -> torch.Tensor:
        """Per-environment position command limits after reset randomization."""
        return self._position_step_limits

    @property
    def rotation_thresholds(self) -> torch.Tensor:
        """Per-environment orientation command limits after reset randomization."""
        return self._rotation_step_limits

    @property
    def fixed_pos_noise(self) -> torch.Tensor:
        """Per-environment fixed-asset position noise sampled at reset."""
        return self._fixed_pos_noise

    @property
    def contact_thresholds(self) -> torch.Tensor:
        """Per-environment wrist-force thresholds sampled at reset."""
        return self._contact_thresholds

    @property
    def smoothed_force(self) -> torch.Tensor:
        """EMA-filtered wrist force owned by the action term."""
        return self._smoothed_force

    @property
    def delta_pos(self) -> torch.Tensor:
        """Last unclipped peg-relative position delta requested by the policy."""
        return self._delta_pos

    @property
    def delta_yaw(self) -> torch.Tensor:
        """Last unclipped yaw delta requested by the policy."""
        return self._delta_yaw

    @property
    def success_prediction(self) -> torch.Tensor:
        """Auxiliary success prediction from the final OSC action channel."""
        return self._success_prediction

    def process_actions(self, actions: torch.Tensor):
        """Filter policy actions and write the corresponding OSC command."""
        self._update_smoothed_force()
        self._update_smoothed_actions(actions)
        self._compute_ee_pose()

        # Convert normalized policy output into a bounded OSC pose command.
        ee_pos_b = self._ee_pose_b[:, POS_SLICE]
        ee_quat_b = self._ee_pose_b[:, QUAT_SLICE]
        self._processed_actions[:, POS_SLICE] = self._compute_bounded_target_position_from_policy_action(ee_pos_b)
        self._processed_actions[:, QUAT_SLICE] = self._compute_bounded_target_orientation_from_policy_yaw(ee_quat_b)

        self._compute_task_frame_pose()
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=self._task_frame_pose_b,
        )

    def _update_smoothed_actions(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self._raw_actions[:] = actions
        self._prev_smoothed_actions[:] = self._smoothed_actions
        self._smoothed_actions[:] = self._ema_factor * actions + (1.0 - self._ema_factor) * self._smoothed_actions
        self._success_prediction[:] = self._smoothed_actions[:, SUCCESS_IDX]

    def _ensure_force_body_idx(self) -> None:
        """Resolve the wrist force-sensor body index."""
        if self._force_body_idx is not None:
            return
        body_ids, body_names = self._asset.find_bodies(self.cfg.force_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Body '{self.cfg.force_body_name}' is required by {self.__class__.__name__} on asset "
                f"'{self.cfg.asset_name}'. Use a USD with a wrist force-sensor body or override "
                f"'force_body_name'. Found {len(body_ids)} matches: {body_names}."
            )
        self._force_body_idx = body_ids[0]

    def _update_smoothed_force(self) -> None:
        """Update the wrist-force EMA owned by the action term."""
        self._ensure_force_body_idx()
        raw_force = wp.to_torch(self._asset.root_view.get_link_incoming_joint_force())[:, self._force_body_idx, :3]
        raw_force = torch.nan_to_num(raw_force, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        self._smoothed_force[:] = (
            self._force_smoothing_factor * raw_force + (1.0 - self._force_smoothing_factor) * self._smoothed_force
        )

    def _compute_bounded_target_position_from_policy_action(self, ee_pos_b: torch.Tensor) -> torch.Tensor:
        """Return a bounded peg-relative position target for stable contact search.

        The policy selects a target around the noisy peg estimate, but the OSC
        command is clipped to a small per-step delta so contact corrections do
        not jump through the gear or peg geometry.
        """
        peg_pos_b = self._get_peg_pos() + self._fixed_pos_noise

        pos_actions = self._smoothed_actions[:, POS_SLICE]
        target_pos = peg_pos_b + pos_actions * self._position_action_bounds

        self._delta_pos[:] = target_pos - ee_pos_b
        clipped_delta = self._clip_delta_with_dead_zone(
            self._delta_pos,
            self._position_step_limits,
            self._pos_dead_zone,
        )
        return ee_pos_b + clipped_delta

    def _compute_bounded_target_orientation_from_policy_yaw(self, ee_quat_b: torch.Tensor) -> torch.Tensor:
        """Return a bounded gripper-down yaw target around the gear symmetry axis."""
        target_yaw = _action_to_target_yaw(self._smoothed_actions[:, YAW_IDX])
        target_quat = self._target_quat_from_yaw(target_yaw)
        desired_xyz = self._clip_orientation_delta(ee_quat_b, target_quat)
        return math_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0],
            pitch=desired_xyz[:, 1],
            yaw=desired_xyz[:, 2],
        )

    def _target_quat_from_yaw(self, target_yaw: torch.Tensor) -> torch.Tensor:
        """Return the gripper-down target orientation for the commanded yaw."""
        zero = torch.zeros_like(target_yaw)
        target_yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, target_yaw)
        gripper_down_quat = math_utils.quat_from_euler_xyz(torch.full_like(target_yaw, math.pi), zero, zero)
        return math_utils.quat_mul(gripper_down_quat, target_yaw_quat)

    def _clip_orientation_delta(self, ee_quat_b: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Clip roll, pitch, and yaw deltas before sending the OSC command."""
        curr_roll, curr_pitch, curr_yaw = math_utils.euler_xyz_from_quat(ee_quat_b, wrap_to_2pi=True)
        desired_roll, desired_pitch, desired_yaw = math_utils.euler_xyz_from_quat(target_quat, wrap_to_2pi=True)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        desired_xyz[:, ROT_ROLL_IDX] = self._clip_roll_delta(curr_roll, desired_roll)
        desired_xyz[:, ROT_PITCH_IDX] = self._clip_pitch_delta(curr_pitch, desired_pitch)
        desired_xyz[:, ROT_YAW_IDX] = self._clip_yaw_delta(curr_yaw, desired_yaw)
        return desired_xyz

    def _clip_roll_delta(self, curr_roll: torch.Tensor, desired_roll: torch.Tensor) -> torch.Tensor:
        """Return roll target after applying the per-step rotation limit."""
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * math.pi, desired_roll)
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clamp(
            delta_roll,
            -self._rotation_step_limits[:, ROT_ROLL_IDX],
            self._rotation_step_limits[:, ROT_ROLL_IDX],
        )
        return curr_roll + clipped_roll

    def _clip_pitch_delta(self, curr_pitch: torch.Tensor, desired_pitch: torch.Tensor) -> torch.Tensor:
        """Return pitch target after wrapping into the signed interval and clipping."""
        curr_pitch_w = torch.where(curr_pitch > math.pi, curr_pitch - 2 * math.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * math.pi, desired_pitch)
        desired_pitch_w = torch.where(desired_pitch > math.pi, desired_pitch - 2 * math.pi, desired_pitch)
        delta_pitch = desired_pitch_w - curr_pitch_w
        clipped_pitch = torch.clamp(
            delta_pitch,
            -self._rotation_step_limits[:, ROT_PITCH_IDX],
            self._rotation_step_limits[:, ROT_PITCH_IDX],
        )
        return curr_pitch_w + clipped_pitch

    def _clip_yaw_delta(self, curr_yaw: torch.Tensor, desired_yaw: torch.Tensor) -> torch.Tensor:
        """Return yaw target after wrapping, clipping, and applying the dead zone."""
        curr_yaw = wrap_yaw(curr_yaw)
        desired_yaw = wrap_yaw(desired_yaw)

        self._delta_yaw[:] = desired_yaw - curr_yaw
        clipped_yaw = self._clip_delta_with_dead_zone(
            self._delta_yaw,
            self._rotation_step_limits[:, ROT_YAW_IDX],
            self._rot_dead_zone,
        )
        return curr_yaw + clipped_yaw

    def _clip_delta_with_dead_zone(
        self,
        delta: torch.Tensor,
        limits: torch.Tensor | float,
        dead_zone: torch.Tensor | float,
    ) -> torch.Tensor:
        """Clamp a requested delta and zero small values."""
        clipped = torch.clamp(delta, -limits, limits)
        return torch.where(torch.abs(clipped) > dead_zone, clipped, torch.zeros_like(clipped))

    def _ee_quat_to_yaw_action(self, ee_quat: torch.Tensor) -> torch.Tensor:
        """Convert the current EE orientation to the normalized policy yaw."""
        n = ee_quat.shape[0]
        gripper_down_to_yaw_frame = _gripper_down_to_yaw_frame_quat(n, self.device)
        yaw_frame_quat = math_utils.quat_mul(gripper_down_to_yaw_frame, ee_quat)
        target_yaw = math_utils.euler_xyz_from_quat(yaw_frame_quat, wrap_to_2pi=True)[2]
        target_yaw = _wrap_yaw_to_action_range(target_yaw)
        return _target_yaw_to_action(target_yaw)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_index, num_envs = self._normalize_env_ids(env_ids)
        if num_envs == 0:
            return

        super().reset(env_ids)
        self._reset_action_filter(env_ids_index, num_envs)
        self._reset_command_thresholds(env_ids_index, num_envs)
        self._reset_contact_state(env_ids_index, num_envs)
        self._reset_initial_smoothed_actions(env_ids_index)
        self._reset_debug_state(env_ids_index)

    def _normalize_env_ids(self, env_ids: Sequence[int] | None) -> tuple[Sequence[int] | slice, int]:
        """Return an index object and count for tensor resets."""
        if env_ids is None:
            return slice(None), self.num_envs
        return env_ids, len(env_ids)

    def _reset_action_filter(self, env_ids: Sequence[int] | slice, num_envs: int) -> None:
        """Randomize the EMA factor used to smooth policy actions this episode."""
        lo, hi = self.cfg.ema_factor_range
        self._ema_factor[env_ids] = lo + torch.rand(num_envs, 1, device=self.device) * (hi - lo)

    def _reset_command_thresholds(self, env_ids: Sequence[int] | slice, num_envs: int) -> None:
        """Randomize per-step OSC command limits and peg-position noise."""
        self._position_step_limits[env_ids] = get_random_prop_gains(
            self._default_position_step_limits[env_ids],
            self.cfg.pos_threshold_noise_level,
            num_envs,
            self.device,
        )
        self._rotation_step_limits[env_ids] = get_random_prop_gains(
            self._default_rotation_step_limits[env_ids],
            self.cfg.rot_threshold_noise_level,
            num_envs,
            self.device,
        )
        self._fixed_pos_noise[env_ids] = torch.randn(num_envs, 3, device=self.device) * self._fixed_pos_noise_levels

    def _reset_contact_state(self, env_ids: Sequence[int] | slice, num_envs: int) -> None:
        """Randomize contact threshold and clear the smoothed force state."""
        ct_lo, ct_hi = self.cfg.contact_threshold_range
        self._contact_thresholds[env_ids] = ct_lo + torch.rand(num_envs, device=self.device) * (ct_hi - ct_lo)
        self._smoothed_force[env_ids] = 0.0

    def _reset_initial_smoothed_actions(self, env_ids: Sequence[int] | slice) -> None:
        """Seed smoothed actions from the current EE pose to avoid a reset-time command jump."""
        self._compute_ee_pose()
        ee_pos = self._ee_pose_b[env_ids, POS_SLICE]
        ee_quat = self._ee_pose_b[env_ids, QUAT_SLICE]

        peg_pos = self._get_peg_pos(env_ids) + self._fixed_pos_noise[env_ids]

        pos_actions = (ee_pos - peg_pos) / self._position_action_bounds
        self._smoothed_actions[env_ids, POS_SLICE] = pos_actions

        yaw_action = self._ee_quat_to_yaw_action(ee_quat)

        self._smoothed_actions[env_ids, ROLL_PITCH_SLICE] = 0.0
        self._smoothed_actions[env_ids, YAW_IDX] = yaw_action
        self._smoothed_actions[env_ids, SUCCESS_IDX] = -1.0

        self._prev_smoothed_actions[env_ids] = self._smoothed_actions[env_ids]

    def _reset_debug_state(self, env_ids: Sequence[int] | slice) -> None:
        """Clear diagnostic buffers used by observations and rewards."""
        self._delta_pos[env_ids] = 0.0
        self._delta_yaw[env_ids] = 0.0
        self._success_prediction[env_ids] = -1.0


@configclass
class NistGearInsertionOscActionCfg(OperationalSpaceControllerActionCfg):
    """Config for :class:`NistGearInsertionOscAction`.

    Environments provide the fixed asset name and peg offset because those
    values are scene geometry, not robot configuration.
    """

    class_type: type[ActionTerm] = NistGearInsertionOscAction

    fixed_asset_name: str = MISSING
    """Name of the fixed asset that contains the insertion peg."""

    peg_offset: tuple[float, float, float] = MISSING
    """Local-frame peg offset on :attr:`fixed_asset_name`."""

    fixed_pos_noise_levels: tuple[float, float, float] = (0.001, 0.001, 0.001)
    """Per-axis standard deviation for reset-time target noise."""

    pos_action_bounds: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """Position scale applied to normalized policy actions."""

    pos_action_threshold: tuple[float, float, float] = (0.02, 0.02, 0.02)
    """Maximum per-step position delta sent to the OSC controller."""

    rot_action_threshold: tuple[float, float, float] = (0.097, 0.097, 0.097)
    """Maximum per-step orientation delta sent to the OSC controller."""

    pos_threshold_noise_level: tuple[float, float, float] = (0.25, 0.25, 0.25)
    """Reset-time multiplicative noise for position thresholds."""

    rot_threshold_noise_level: tuple[float, float, float] = (0.29, 0.29, 0.29)
    """Reset-time multiplicative noise for orientation thresholds."""

    ema_factor_range: tuple[float, float] = (0.05, 0.2)
    """Reset-time range for the action EMA factor."""

    contact_threshold_range: tuple[float, float] = (5.0, 10.0)
    """Reset-time wrist-force threshold range."""

    pos_dead_zone: tuple[float, float, float] = (0.0005, 0.0005, 0.0005)
    """Position command dead zone."""

    rot_dead_zone: float = 0.001
    """Orientation command dead zone."""

    force_body_name: str = "force_sensor"
    """Body that exposes the wrist force-sensor joint wrench."""

    force_smoothing_factor: float = 0.25
    """EMA factor used to smooth wrist-force readings."""
