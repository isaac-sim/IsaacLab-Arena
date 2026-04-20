# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""All observation terms for the NIST gear insertion task.

This module contains:

* Task-level observation primitives (peg position, gear-base offsets, deltas).
* Privileged critic helpers (body pose, force/torque at a body link).
* The 24-D policy observation class (``NistGearInsertionPolicyObservations``).
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


# ---------------------------------------------------------------------------
# Task-level observation primitives
# ---------------------------------------------------------------------------


def peg_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Target peg position: fixed asset pose + offset in its local frame."""
    board: RigidObject = env.scene[board_cfg.name]
    pos = wp.to_torch(board.data.root_pos_w) - env.scene.env_origins
    quat = wp.to_torch(board.data.root_quat_w)
    offset = torch.tensor(peg_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    return pos + math_utils.quat_apply(quat, offset)


def held_gear_base_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    held_gear_base_offset: tuple[float, ...] = (2.025e-2, 0.0, 0.0),
) -> torch.Tensor:
    """Position of the held gear's insertion point (root + offset in gear frame) in env frame."""
    gear: RigidObject = env.scene[gear_cfg.name]
    gear_pos = wp.to_torch(gear.data.root_pos_w) - env.scene.env_origins
    gear_quat = wp.to_torch(gear.data.root_quat_w)
    held_off = (
        torch.tensor(held_gear_base_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    )
    return gear_pos + math_utils.quat_apply(gear_quat, held_off)


def check_gear_insertion_geometry(
    held_base_pos: torch.Tensor,
    peg_pos: torch.Tensor,
    gear_peg_height: float,
    z_fraction: float,
    xy_threshold: float,
) -> torch.Tensor:
    """Shared XY-centering + Z-depth insertion check used by rewards, terminations, and observations.

    Args:
        held_base_pos: (N, 3) position of the held gear's insertion base in env frame.
        peg_pos: (N, 3) position of the target peg in env frame.
        gear_peg_height: Physical height of the peg.
        z_fraction: Fraction of peg height that counts as inserted.
        xy_threshold: Maximum XY distance for centering.

    Returns:
        (N,) bool tensor — True where gear is centered and inserted.
    """
    xy_dist = torch.norm(held_base_pos[:, :2] - peg_pos[:, :2], dim=-1)
    z_diff = held_base_pos[:, 2] - peg_pos[:, 2]
    return (xy_dist < xy_threshold) & (z_diff < gear_peg_height * z_fraction)


def peg_delta_from_held_gear_base(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
    held_gear_base_offset: tuple[float, ...] = (2.025e-2, 0.0, 0.0),
) -> torch.Tensor:
    """Vector from held gear insertion point to peg. Positive = peg is ahead in that axis."""
    held_base = held_gear_base_pos_in_env_frame(env, gear_cfg, held_gear_base_offset)
    peg_pos = peg_pos_in_env_frame(env, board_cfg, peg_offset)
    return peg_pos - held_base


# ---------------------------------------------------------------------------
# Privileged critic helpers (body-level pose & wrench)
# ---------------------------------------------------------------------------


def body_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Noiseless body position in env frame (privileged state for critic)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.body_names.index(body_name)
    pos_w = wp.to_torch(robot.data.body_pos_w)[:, idx, :]
    return pos_w - env.scene.env_origins


def body_quat_canonical(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Noiseless body quaternion, canonicalized w >= 0 (privileged state for critic)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.body_names.index(body_name)
    quat = wp.to_torch(robot.data.body_quat_w)[:, idx, :]
    return quat_unique(quat)


def force_torque_at_body(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "force_sensor",
    return_torque: bool = False,
) -> torch.Tensor:
    """Read joint reaction wrench at a specific body link.

    Uses PhysX ``root_view.get_link_incoming_joint_force()`` (same pattern as
    Isaac Lab's ``forge_env.py``). This is the standard way to read F/T sensor
    data in Isaac Sim; no higher-level API is available.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    wrench = wp.to_torch(robot.root_view.get_link_incoming_joint_force())[:, body_idx]
    wrench = torch.nan_to_num(wrench, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    if return_torque:
        return wrench
    return wrench[:, :3]


# ---------------------------------------------------------------------------
# 24-D policy observation (OSC + wrist sensing)
# ---------------------------------------------------------------------------


class NistGearInsertionPolicyObservations(ManagerTermBase):
    """24-D policy observation stack for NIST gear insertion (OSC + wrist sensing).

    Output layout (per env)::

        fingertip_pos_rel_fixed  (3)
        fingertip_quat           (4)
        ee_linvel                (3)
        ee_angvel                (3)
        ft_force                 (3)
        force_threshold          (1)
        prev_actions             (7)
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        p = cfg.params
        self._robot_name: str = p.get("robot_name", "robot")
        self._board_name: str = p.get("board_name", "gears_and_base")
        self._peg_offset = torch.tensor(p.get("peg_offset", [0.0, 0.0, 0.0]), device=env.device)
        self._fingertip_body: str = p.get("fingertip_body_name", "panda_fingertip_centered")
        self._force_body: str = p.get("force_body_name", "force_sensor")

        self._pos_noise: float = p.get("pos_noise_level", 0.00025)
        self._rot_noise_deg: float = p.get("rot_noise_level_deg", 0.1)
        self._force_noise: float = p.get("force_noise_level", 1.0)
        self._ft_alpha: float = p.get("ft_smoothing_factor", 0.25)
        self._contact_thresh_range: tuple[float, float] = tuple(p.get("contact_threshold_range", [5.0, 10.0]))

        n = env.num_envs
        dev = env.device

        self._fingertip_idx: int | None = None
        self._force_idx: int | None = None

        self._flip_quats = torch.ones(n, device=dev)
        self._prev_noisy_pos = torch.zeros(n, 3, device=dev)
        self._prev_noisy_quat = torch.zeros(n, 4, device=dev)
        self._prev_noisy_quat[:, 3] = 1.0

    def _ensure_body_indices(self):
        if self._fingertip_idx is not None:
            return
        robot: Articulation = self._env.scene[self._robot_name]
        self._fingertip_idx = robot.body_names.index(self._fingertip_body)
        self._force_idx = robot.body_names.index(self._force_body)

    def reset(self, env_ids: list[int] | None = None):
        if env_ids is None or len(env_ids) == 0:
            return

        n = len(env_ids)
        dev = self._env.device

        flip = torch.ones(n, device=dev)
        flip[torch.rand(n, device=dev) > 0.5] = -1.0
        self._flip_quats[env_ids] = flip

        self._ensure_body_indices()
        robot: Articulation = self._env.scene[self._robot_name]
        origins = self._env.scene.env_origins
        self._prev_noisy_pos[env_ids] = (
            wp.to_torch(robot.data.body_pos_w)[env_ids, self._fingertip_idx] - origins[env_ids]
        )
        reset_quat = wp.to_torch(robot.data.body_quat_w)[env_ids, self._fingertip_idx].clone()
        reset_quat[:, [2, 3]] = 0.0
        reset_quat = torch.nn.functional.normalize(reset_quat, dim=-1)
        reset_quat = reset_quat * flip.unsqueeze(-1)
        self._prev_noisy_quat[env_ids] = reset_quat

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_name: str = "robot",
        board_name: str = "gears_and_base",
        peg_offset: list[float] | None = None,
        fingertip_body_name: str = "panda_fingertip_centered",
        force_body_name: str = "force_sensor",
        pos_noise_level: float = 0.00025,
        rot_noise_level_deg: float = 0.1,
        force_noise_level: float = 1.0,
    ) -> torch.Tensor:
        self._ensure_body_indices()

        n = env.num_envs
        dev = env.device
        dt = env.step_dt

        robot: Articulation = env.scene[self._robot_name]
        board = env.scene[self._board_name]
        origins = env.scene.env_origins

        ft_pos = wp.to_torch(robot.data.body_pos_w)[:, self._fingertip_idx] - origins
        ft_quat = wp.to_torch(robot.data.body_quat_w)[:, self._fingertip_idx]

        pos_noise = torch.randn(n, 3, device=dev) * self._pos_noise
        noisy_pos = ft_pos + pos_noise

        rot_noise_axis = torch.randn(n, 3, device=dev)
        rot_noise_axis = rot_noise_axis / (torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True) + 1e-8)
        rot_noise_angle = torch.randn(n, device=dev) * math.radians(self._rot_noise_deg)
        noisy_quat = math_utils.quat_mul(
            ft_quat,
            math_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis),
        )
        noisy_quat[:, [2, 3]] = 0.0
        noisy_quat = torch.nn.functional.normalize(noisy_quat, dim=-1)
        noisy_quat = noisy_quat * self._flip_quats.unsqueeze(-1)

        # No public API for action term lookup; _terms is the standard access pattern.
        arm_osc_action = env.action_manager._terms["arm_action"]
        board_pos = wp.to_torch(board.data.root_pos_w) - origins
        board_quat = wp.to_torch(board.data.root_quat_w)
        peg_offset_exp = self._peg_offset.unsqueeze(0).expand(n, 3)
        peg_pos = board_pos + math_utils.quat_apply(board_quat, peg_offset_exp)
        noisy_fixed_pos = peg_pos + arm_osc_action.fixed_pos_noise

        fingertip_pos_rel = noisy_pos - noisy_fixed_pos

        safe_dt = max(dt, 1e-6)
        ee_linvel = (noisy_pos - self._prev_noisy_pos) / safe_dt
        self._prev_noisy_pos[:] = noisy_pos

        rot_diff = math_utils.quat_mul(noisy_quat, math_utils.quat_conjugate(self._prev_noisy_quat))
        rot_diff = rot_diff * torch.sign(rot_diff[:, 3]).unsqueeze(-1)
        ee_angvel = axis_angle_from_quat(rot_diff) / safe_dt
        ee_angvel[:, 0:2] = 0.0
        self._prev_noisy_quat[:] = noisy_quat

        raw_force = wp.to_torch(robot.root_view.get_link_incoming_joint_force())[:, self._force_idx, :3]
        raw_force = torch.nan_to_num(raw_force, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        # Force EMA is updated here (in the observation) rather than in process_actions
        # because the smoothed force must be current *before* the policy reads it.
        # Moving this to process_actions would delay the update by one step.
        arm_osc_action.force_smooth[:] = (
            self._ft_alpha * raw_force + (1.0 - self._ft_alpha) * arm_osc_action.force_smooth
        )
        noisy_force = arm_osc_action.force_smooth + torch.randn(n, 3, device=dev) * self._force_noise

        force_threshold = arm_osc_action.contact_thresholds.unsqueeze(-1)

        prev_actions = arm_osc_action._smoothed_actions.clone()
        prev_actions[:, 3:5] = 0.0

        return torch.cat(
            [
                fingertip_pos_rel,
                noisy_quat,
                ee_linvel,
                ee_angvel,
                noisy_force,
                force_threshold,
                prev_actions,
            ],
            dim=-1,
        )
