# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSC policy observations for Franka NIST gear insertion.

The observation term is paired with :class:`NistGearInsertionOscAction`: it exposes
the peg-relative fingertip state, smoothed wrist force, force threshold, and
filtered previous action values expected by the trained 24-D policy observation.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import axis_angle_from_quat, quat_unique

from isaaclab_arena.tasks.nist_gear_insertion.geometry import compute_asset_local_offset_pos
from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_action import get_nist_gear_insertion_arm_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


FINGERTIP_POS_REL_DIM = 3
FINGERTIP_QUAT_DIM = 4
EE_LINVEL_DIM = 3
EE_ANGVEL_DIM = 3
FT_FORCE_DIM = 3
FORCE_THRESHOLD_DIM = 1
PREV_ACTION_DIM = 7
POLICY_OBS_LAYOUT = (
    ("fingertip_pos_rel_fixed", FINGERTIP_POS_REL_DIM),
    ("fingertip_quat", FINGERTIP_QUAT_DIM),
    ("ee_linvel", EE_LINVEL_DIM),
    ("ee_angvel", EE_ANGVEL_DIM),
    ("ft_force", FT_FORCE_DIM),
    ("force_threshold", FORCE_THRESHOLD_DIM),
    ("prev_actions", PREV_ACTION_DIM),
)
POLICY_OBS_DIM = sum(dim for _, dim in POLICY_OBS_LAYOUT)
PREV_ACTION_ROLL_PITCH_SLICE = slice(3, 5)


@dataclass(frozen=True)
class NistGearInsertionPolicyObsParams:
    """Resolved config values for :class:`NistGearInsertionPolicyObservations`."""

    robot_name: str = "robot"
    board_name: str = "fixed_asset"
    peg_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fingertip_body_name: str = "panda_fingertip_centered"
    force_body_name: str = "force_sensor"
    """Body whose presence is validated because the paired action term reads wrist force."""
    pos_noise_level: float = 0.00025
    rot_noise_level_deg: float = 0.1
    force_noise_level: float = 1.0

    @classmethod
    def from_dict(cls, params: dict) -> NistGearInsertionPolicyObsParams:
        return cls(
            robot_name=params.get("robot_name", cls.robot_name),
            board_name=params.get("board_name", cls.board_name),
            peg_offset=tuple(params.get("peg_offset", cls.peg_offset)),
            fingertip_body_name=params.get("fingertip_body_name", cls.fingertip_body_name),
            force_body_name=params.get("force_body_name", cls.force_body_name),
            pos_noise_level=params.get("pos_noise_level", cls.pos_noise_level),
            rot_noise_level_deg=params.get("rot_noise_level_deg", cls.rot_noise_level_deg),
            force_noise_level=params.get("force_noise_level", cls.force_noise_level),
        )


def _make_reported_quat(quat: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Return the symmetry-reduced quaternion channels exposed to the policy.

    Gear rotation about the peg axis is handled through the yaw action. For
    checkpoint parity, the packed observation keeps the original x/y channels
    and zeros z/w instead of renormalizing the quaternion.
    """
    reported_quat = quat.clone()
    reported_quat[:, [2, 3]] = 0.0
    return reported_quat * flip.unsqueeze(-1)


def _compute_pose_velocities(
    prev_pos: torch.Tensor,
    prev_quat: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate finite-difference linear velocity and yaw angular velocity."""
    safe_dt = max(dt, 1e-6)
    linvel = (pos - prev_pos) / safe_dt

    rot_diff = math_utils.quat_mul(quat, math_utils.quat_conjugate(prev_quat))
    rot_diff = quat_unique(rot_diff)
    angvel = axis_angle_from_quat(rot_diff) / safe_dt
    angvel[:, 0:2] = 0.0
    return linvel, angvel


class NistGearInsertionPolicyObservations(ManagerTermBase):
    """Policy observation term for OSC-based gear insertion.

    The term reads state owned by :class:`NistGearInsertionOscAction`, including
    reset-time peg noise, smoothed force, per-episode force threshold, and
    previous filtered actions. The action term owns the wrist-force EMA and
    updates it at most once per simulation step.

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

        self._params = NistGearInsertionPolicyObsParams.from_dict(cfg.params)
        self._robot_name = self._params.robot_name
        self._board_name = self._params.board_name
        self._peg_offset_values = self._params.peg_offset
        self._peg_offset = torch.tensor(self._peg_offset_values, device=env.device)
        self._fingertip_body = self._params.fingertip_body_name
        self._force_body = self._params.force_body_name

        self._pos_noise = self._params.pos_noise_level
        self._rot_noise_deg = self._params.rot_noise_level_deg
        self._force_noise = self._params.force_noise_level

        n = env.num_envs
        dev = env.device

        self._fingertip_idx: int | None = None
        self._body_key = (self._robot_name, self._fingertip_body, self._force_body)

        self._flip_quats = torch.ones(n, device=dev)
        self._prev_noisy_pos = torch.zeros(n, 3, device=dev)
        self._prev_noisy_quat = torch.zeros(n, 4, device=dev)
        self._prev_noisy_quat[:, 3] = 1.0

    def _resolve_fingertip_idx(
        self,
        robot_name: str,
        fingertip_body_name: str,
        force_body_name: str,
    ) -> int:
        """Resolve the fingertip body index used by the observation term."""
        body_key = (robot_name, fingertip_body_name, force_body_name)
        if self._fingertip_idx is not None and body_key == self._body_key:
            return self._fingertip_idx

        robot: Articulation = self._env.scene[robot_name]
        for body_name in (fingertip_body_name, force_body_name):
            body_ids, body_names = robot.find_bodies([body_name])
            if not body_ids:
                raise ValueError(
                    f"Body '{body_name}' is missing from robot '{robot_name}'. Use a USD that defines this "
                    "body or override the corresponding observation parameter. "
                    f"Matches: {body_names}."
                )

        fingertip_ids, _ = robot.find_bodies([fingertip_body_name])
        fingertip_idx = fingertip_ids[0]
        if body_key == self._body_key:
            self._fingertip_idx = fingertip_idx
        return fingertip_idx

    def _resolve_peg_offset(self, peg_offset: list[float] | None, device: torch.device) -> torch.Tensor:
        """Return the cached peg offset unless the manager supplies different values."""
        if peg_offset is None or tuple(peg_offset) == self._peg_offset_values:
            return self._peg_offset
        return torch.tensor(peg_offset, device=device, dtype=torch.float32)

    def _sample_noisy_pose(
        self,
        env: ManagerBasedRLEnv,
        ft_pos: torch.Tensor,
        ft_quat: torch.Tensor,
        pos_noise_level: float,
        rot_noise_level_deg: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fingertip pose with configured sensor noise."""
        pos_noise = torch.randn(env.num_envs, 3, device=env.device) * pos_noise_level
        noisy_pos = ft_pos + pos_noise

        rot_noise_axis = torch.randn(env.num_envs, 3, device=env.device)
        rot_noise_axis = F.normalize(rot_noise_axis, dim=1, eps=1e-8)
        rot_noise_angle = torch.randn(env.num_envs, device=env.device) * math.radians(rot_noise_level_deg)
        noisy_quat = math_utils.quat_mul(
            ft_quat,
            math_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis),
        )
        return noisy_pos, quat_unique(noisy_quat)

    def _compute_fingertip_target_delta(
        self,
        env: ManagerBasedRLEnv,
        noisy_pos: torch.Tensor,
        board_name: str,
        peg_offset: torch.Tensor,
        arm_osc_action,
    ) -> torch.Tensor:
        """Return fingertip position relative to the noisy peg target."""
        board = env.scene[board_name]
        peg_pos = compute_asset_local_offset_pos(env, board, peg_offset)
        noisy_fixed_pos = peg_pos + arm_osc_action.fixed_pos_noise
        return noisy_pos - noisy_fixed_pos

    def _compute_velocities(
        self,
        noisy_pos: torch.Tensor,
        obs_quat: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate fingertip linear and angular velocity from noisy poses."""
        ee_linvel, ee_angvel = _compute_pose_velocities(
            self._prev_noisy_pos,
            self._prev_noisy_quat,
            noisy_pos,
            obs_quat,
            dt,
        )

        self._prev_noisy_pos[:] = noisy_pos
        self._prev_noisy_quat[:] = obs_quat
        return ee_linvel, ee_angvel

    def _read_smoothed_force(
        self,
        env: ManagerBasedRLEnv,
        arm_osc_action,
        force_noise_level: float,
    ) -> torch.Tensor:
        """Return the smoothed wrist force with configured observation noise."""
        force = arm_osc_action.smoothed_force
        return force + torch.randn(env.num_envs, 3, device=env.device) * force_noise_level

    def _read_fingertip_pose(
        self,
        env: ManagerBasedRLEnv,
        robot_name: str,
        fingertip_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fingertip pose in the environment frame."""
        robot: Articulation = env.scene[robot_name]
        origins = env.scene.env_origins
        fingertip_pos = wp.to_torch(robot.data.body_pos_w)[:, fingertip_idx] - origins
        fingertip_quat = wp.to_torch(robot.data.body_quat_w)[:, fingertip_idx]
        return fingertip_pos, fingertip_quat

    def _compute_fingertip_observations(
        self,
        env: ManagerBasedRLEnv,
        robot_name: str,
        board_name: str,
        peg_offset: torch.Tensor,
        fingertip_idx: int,
        pos_noise_level: float,
        rot_noise_level_deg: float,
        arm_osc_action,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return target-relative pose and velocity observations for the fingertip."""
        ft_pos, ft_quat = self._read_fingertip_pose(env, robot_name, fingertip_idx)
        noisy_pos, noisy_quat_full = self._sample_noisy_pose(env, ft_pos, ft_quat, pos_noise_level, rot_noise_level_deg)
        fingertip_pos_rel = self._compute_fingertip_target_delta(env, noisy_pos, board_name, peg_offset, arm_osc_action)
        obs_quat = _make_reported_quat(noisy_quat_full, self._flip_quats)
        ee_linvel, ee_angvel = self._compute_velocities(noisy_pos, noisy_quat_full, env.step_dt)
        return fingertip_pos_rel, obs_quat, ee_linvel, ee_angvel

    def _compute_force_action_observations(
        self,
        env: ManagerBasedRLEnv,
        arm_osc_action,
        force_noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return wrist-force, force-threshold, and previous-action observations."""
        noisy_force = self._read_smoothed_force(env, arm_osc_action, force_noise_level)
        force_threshold = arm_osc_action.contact_thresholds.unsqueeze(-1)

        prev_actions = arm_osc_action.smoothed_actions.clone()
        prev_actions[:, PREV_ACTION_ROLL_PITCH_SLICE] = 0.0
        return noisy_force, force_threshold, prev_actions

    def _pack_observation(
        self,
        fingertip_pos_rel: torch.Tensor,
        obs_quat: torch.Tensor,
        ee_linvel: torch.Tensor,
        ee_angvel: torch.Tensor,
        noisy_force: torch.Tensor,
        force_threshold: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the policy observation and enforce the documented layout."""
        obs = torch.cat(
            [
                fingertip_pos_rel,
                obs_quat,
                ee_linvel,
                ee_angvel,
                noisy_force,
                force_threshold,
                prev_actions,
            ],
            dim=-1,
        )
        assert obs.shape[-1] == POLICY_OBS_DIM, f"Expected {POLICY_OBS_DIM}D gear insertion policy obs."
        return torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    def _resolve_call_params(
        self,
        robot_name: str | None,
        board_name: str | None,
        fingertip_body_name: str | None,
        force_body_name: str | None,
        pos_noise_level: float | None,
        rot_noise_level_deg: float | None,
        force_noise_level: float | None,
    ) -> tuple[str, str, str, str, float, float, float]:
        """Resolve optional call-time overrides against the configured defaults."""
        return (
            self._robot_name if robot_name is None else robot_name,
            self._board_name if board_name is None else board_name,
            self._fingertip_body if fingertip_body_name is None else fingertip_body_name,
            self._force_body if force_body_name is None else force_body_name,
            self._pos_noise if pos_noise_level is None else pos_noise_level,
            self._rot_noise_deg if rot_noise_level_deg is None else rot_noise_level_deg,
            self._force_noise if force_noise_level is None else force_noise_level,
        )

    def reset(self, env_ids: list[int] | None = None):
        """Reset noisy pose history and quaternion sign for selected environments."""
        env_ids, num_envs = self._normalize_env_ids(env_ids)
        if num_envs == 0:
            return

        dev = self._env.device

        flip = torch.ones(num_envs, device=dev)
        flip[torch.rand(num_envs, device=dev) > 0.5] = -1.0
        self._flip_quats[env_ids] = flip

        fingertip_idx = self._resolve_fingertip_idx(self._robot_name, self._fingertip_body, self._force_body)
        robot: Articulation = self._env.scene[self._robot_name]
        origins = self._env.scene.env_origins
        self._prev_noisy_pos[env_ids] = wp.to_torch(robot.data.body_pos_w)[env_ids, fingertip_idx] - origins[env_ids]
        reset_quat = wp.to_torch(robot.data.body_quat_w)[env_ids, fingertip_idx]
        self._prev_noisy_quat[env_ids] = quat_unique(reset_quat)

    def _normalize_env_ids(
        self,
        env_ids: list[int] | torch.Tensor | None,
    ) -> tuple[torch.Tensor | slice, int]:
        """Return an index object and count for observation reset buffers."""
        if env_ids is None:
            return slice(None), self._env.num_envs
        env_ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        return env_ids, env_ids.numel()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_name: str | None = None,
        board_name: str | None = None,
        peg_offset: list[float] | None = None,
        fingertip_body_name: str | None = None,
        force_body_name: str | None = None,
        pos_noise_level: float | None = None,
        rot_noise_level_deg: float | None = None,
        force_noise_level: float | None = None,
    ) -> torch.Tensor:
        """Return the 24-D policy observation tensor."""
        (
            robot_name,
            board_name,
            fingertip_body_name,
            force_body_name,
            pos_noise_level,
            rot_noise_level_deg,
            force_noise_level,
        ) = self._resolve_call_params(
            robot_name,
            board_name,
            fingertip_body_name,
            force_body_name,
            pos_noise_level,
            rot_noise_level_deg,
            force_noise_level,
        )

        peg_offset_tensor = self._resolve_peg_offset(peg_offset, env.device)
        fingertip_idx = self._resolve_fingertip_idx(robot_name, fingertip_body_name, force_body_name)
        arm_osc_action = get_nist_gear_insertion_arm_action(env)

        fingertip_pos_rel, obs_quat, ee_linvel, ee_angvel = self._compute_fingertip_observations(
            env,
            robot_name,
            board_name,
            peg_offset_tensor,
            fingertip_idx,
            pos_noise_level,
            rot_noise_level_deg,
            arm_osc_action,
        )
        noisy_force, force_threshold, prev_actions = self._compute_force_action_observations(
            env,
            arm_osc_action,
            force_noise_level,
        )

        return self._pack_observation(
            fingertip_pos_rel,
            obs_quat,
            ee_linvel,
            ee_angvel,
            noisy_force,
            force_threshold,
            prev_actions,
        )
