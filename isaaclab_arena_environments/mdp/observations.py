# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom observation terms for assembly environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import axis_angle_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def body_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Noiseless body position in env frame (privileged state for critic)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.body_names.index(body_name)
    pos_w = robot.data.body_pos_w[:, idx, :]
    return pos_w - env.scene.env_origins


def body_quat_canonical(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Noiseless body quaternion, canonicalized w >= 0 (privileged state for critic)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.body_names.index(body_name)
    quat = robot.data.body_quat_w[:, idx, :]
    sign = torch.where(quat[:, 0:1] < 0, -1.0, 1.0)
    return quat * sign


def force_torque_at_body(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "force_sensor",
    return_torque: bool = False,
) -> torch.Tensor:
    """Read joint reaction wrench at a specific body link."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    wrench = robot.root_physx_view.get_link_incoming_joint_force()[:, body_idx]
    if return_torque:
        return wrench
    return wrench[:, :3]


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
        self._contact_thresh_range: tuple[float, float] = tuple(
            p.get("contact_threshold_range", [5.0, 10.0])
        )

        n = env.num_envs
        dev = env.device

        self._fingertip_idx: int | None = None
        self._force_idx: int | None = None

        self._flip_quats = torch.ones(n, device=dev)
        self._prev_noisy_pos = torch.zeros(n, 3, device=dev)
        self._prev_noisy_quat = torch.zeros(n, 4, device=dev)
        self._prev_noisy_quat[:, 0] = 1.0

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
            robot.data.body_pos_w[env_ids, self._fingertip_idx] - origins[env_ids]
        )
        self._prev_noisy_quat[env_ids] = robot.data.body_quat_w[env_ids, self._fingertip_idx]

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

        ft_pos = robot.data.body_pos_w[:, self._fingertip_idx] - origins
        ft_quat = robot.data.body_quat_w[:, self._fingertip_idx]

        pos_noise = torch.randn(n, 3, device=dev) * self._pos_noise
        noisy_pos = ft_pos + pos_noise

        rot_noise_axis = torch.randn(n, 3, device=dev)
        rot_noise_axis = rot_noise_axis / (torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True) + 1e-8)
        rot_noise_angle = torch.randn(n, device=dev) * math.radians(self._rot_noise_deg)
        noisy_quat = torch_utils.quat_mul(
            ft_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis),
        )
        noisy_quat[:, [0, 3]] = 0.0
        noisy_quat = noisy_quat * self._flip_quats.unsqueeze(-1)

        arm_osc_action = env.action_manager._terms["arm_action"]
        board_pos = board.data.root_pos_w - origins
        board_quat = board.data.root_quat_w
        peg_offset_exp = self._peg_offset.unsqueeze(0).expand(n, 3)
        peg_pos = board_pos + math_utils.quat_apply(board_quat, peg_offset_exp)
        noisy_fixed_pos = peg_pos + arm_osc_action.fixed_pos_noise

        fingertip_pos_rel = noisy_pos - noisy_fixed_pos

        ee_linvel = (noisy_pos - self._prev_noisy_pos) / dt
        self._prev_noisy_pos[:] = noisy_pos

        rot_diff = torch_utils.quat_mul(noisy_quat, torch_utils.quat_conjugate(self._prev_noisy_quat))
        rot_diff = rot_diff * torch.sign(rot_diff[:, 0]).unsqueeze(-1)
        ee_angvel = axis_angle_from_quat(rot_diff) / dt
        ee_angvel[:, 0:2] = 0.0
        self._prev_noisy_quat[:] = noisy_quat

        raw_force = robot.root_physx_view.get_link_incoming_joint_force()[:, self._force_idx, :3]
        arm_osc_action.force_smooth[:] = (
            self._ft_alpha * raw_force + (1.0 - self._ft_alpha) * arm_osc_action.force_smooth
        )
        noisy_force = arm_osc_action.force_smooth + torch.randn(n, 3, device=dev) * self._force_noise

        force_threshold = arm_osc_action.contact_thresholds.unsqueeze(-1)

        prev_actions = arm_osc_action._smoothed_actions.clone()
        prev_actions[:, 3:5] = 0.0

        return torch.cat([
            fingertip_pos_rel,
            noisy_quat,
            ee_linvel,
            ee_angvel,
            noisy_force,
            force_threshold,
            prev_actions,
        ], dim=-1)
