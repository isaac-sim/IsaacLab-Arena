# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the NIST gear insertion task.

Includes squashing-function keypoint rewards (baseline, coarse, fine), engagement
and success bonuses, and optional OSC / contact regularizers. Design follows
common assembly peg-insert RL practice; see e.g. Appendix B of
https://arxiv.org/pdf/2408.04587 for related shaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_keypoint_offsets(num_keypoints: int = 4, device: torch.device | None = None) -> torch.Tensor:
    """Uniformly-spaced keypoints along the Z-axis, centered at 0."""
    offsets = torch.zeros((num_keypoints, 3), device=device, dtype=torch.float32)
    offsets[:, 2] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return offsets


class _KeypointDistanceComputer:
    """Pre-cached keypoint distance calculator matching Factory's pattern."""

    def __init__(self, num_envs: int, device: torch.device, num_keypoints: int = 4):
        self.offsets_base = _get_keypoint_offsets(num_keypoints=num_keypoints, device=device)
        self.n_kp = self.offsets_base.shape[0]
        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
            .repeat(num_envs * self.n_kp, 1)
            .contiguous()
        )
        self.offsets_buf = torch.zeros(num_envs, self.n_kp, 3, device=device, dtype=torch.float32)

    def compute(
        self,
        pos_a: torch.Tensor,
        quat_a: torch.Tensor,
        pos_b: torch.Tensor,
        quat_b: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Returns mean keypoint L2 distance, shape (num_envs,)."""
        n = pos_a.shape[0]
        offsets = self.offsets_base * scale
        self.offsets_buf[:n] = offsets.unsqueeze(0)
        off_flat = self.offsets_buf[:n].reshape(-1, 3)
        iq = self.identity_quat[: n * self.n_kp]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(-1, self.n_kp, -1).reshape(-1, t.shape[-1])

        kp_a, _ = combine_frame_transforms(_expand(pos_a), _expand(quat_a), off_flat, iq)
        kp_b, _ = combine_frame_transforms(_expand(pos_b), _expand(quat_b), off_flat, iq)
        per_kp_dist = torch.norm(kp_b.reshape(n, self.n_kp, 3) - kp_a.reshape(n, self.n_kp, 3), p=2, dim=-1)
        return per_kp_dist.mean(-1)


def _squashing_fn(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Squashing function r(x) = 1 / (exp(a*x) + b + exp(-a*x))."""
    return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))


class gear_peg_keypoint_squashing(ManagerTermBase):
    """Squashing-function keypoint reward for gear vs peg alignment.

    Instantiate three times with different [a, b] for baseline/coarse/fine.
    Supports optional per-episode XY noise on the peg offset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.gear_cfg: SceneEntityCfg = cfg.params["gear_cfg"]
        self.board_cfg: SceneEntityCfg = cfg.params["board_cfg"]
        self.peg_offset = torch.tensor(cfg.params.get("peg_offset", [0.0, 0.0, 0.0]), device=env.device, dtype=torch.float32)
        self.held_gear_base_offset = torch.tensor(
            cfg.params.get("held_gear_base_offset", [2.025e-2, 0.0, 0.0]), device=env.device, dtype=torch.float32
        )
        self._xy_noise_range = cfg.params.get("peg_offset_xy_noise", 0.0)
        num_keypoints = cfg.params.get("num_keypoints", 4)
        self.kp = _KeypointDistanceComputer(env.num_envs, env.device, num_keypoints=num_keypoints)
        self._offset_noise = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor) -> None:
        if self._xy_noise_range > 0.0:
            n = len(env_ids)
            self._offset_noise[env_ids, 0] = (torch.rand(n, device=self._offset_noise.device) * 2 - 1) * self._xy_noise_range
            self._offset_noise[env_ids, 1] = (torch.rand(n, device=self._offset_noise.device) * 2 - 1) * self._xy_noise_range

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
        board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
        peg_offset: list[float] = [0.0, 0.0, 0.0],
        held_gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
        keypoint_scale: float = 0.15,
        num_keypoints: int = 4,
        squash_a: float = 50.0,
        squash_b: float = 2.0,
        peg_offset_xy_noise: float = 0.0,
    ) -> torch.Tensor:
        gear: RigidObject = env.scene[self.gear_cfg.name]
        gear_pos = wp.to_torch(gear.data.root_pos_w) - env.scene.env_origins
        gear_quat = wp.to_torch(gear.data.root_quat_w)
        n = gear_pos.shape[0]
        held_offset = self.held_gear_base_offset.unsqueeze(0).expand(n, 3)
        held_base_pos = gear_pos + quat_apply(gear_quat, held_offset)

        board: RigidObject = env.scene[self.board_cfg.name]
        pos = wp.to_torch(board.data.root_pos_w)[:n] - env.scene.env_origins[:n]
        quat = wp.to_torch(board.data.root_quat_w)[:n]
        offset = self.peg_offset.unsqueeze(0).expand(n, 3)
        target_pos = pos + quat_apply(quat, offset) + self._offset_noise[:n]
        target_quat = quat
        kp_dist = self.kp.compute(target_pos, target_quat, held_base_pos, gear_quat, scale=keypoint_scale)
        return _squashing_fn(kp_dist, squash_a, squash_b)


def _check_gear_position(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    board_cfg: SceneEntityCfg,
    peg_offset: list[float],
    held_gear_base_offset: list[float],
    gear_peg_height: float,
    z_fraction: float,
    xy_threshold: float,
) -> torch.Tensor:
    """Return bool tensor indicating whether gear meets XY centering and Z depth criteria.

    Compares the held gear's insertion base (root + offset in gear frame) against
    the peg position (fixed asset + offset in fixed asset frame).
    """
    gear: RigidObject = env.scene[gear_cfg.name]
    gear_pos = wp.to_torch(gear.data.root_pos_w) - env.scene.env_origins
    gear_quat = wp.to_torch(gear.data.root_quat_w)
    held_off = torch.tensor(held_gear_base_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    held_base_pos = gear_pos + quat_apply(gear_quat, held_off)

    board: RigidObject = env.scene[board_cfg.name]
    pos = wp.to_torch(board.data.root_pos_w) - env.scene.env_origins
    quat = wp.to_torch(board.data.root_quat_w)
    offset = torch.tensor(peg_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    peg_pos = pos + quat_apply(quat, offset)

    xy_dist = torch.norm(held_base_pos[:, :2] - peg_pos[:, :2], dim=-1)
    z_diff = held_base_pos[:, 2] - peg_pos[:, 2]
    height_threshold = gear_peg_height * z_fraction

    is_centered = xy_dist < xy_threshold
    is_inserted = z_diff < height_threshold
    return is_centered & is_inserted


def gear_insertion_engagement_bonus(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: list[float] = [0.0, 0.0, 0.0],
    held_gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    gear_peg_height: float = 0.02,
    engage_z_fraction: float = 0.90,
    xy_threshold: float = 0.015,
) -> torch.Tensor:
    """Bonus when the gear is partially engaged on the peg."""
    return _check_gear_position(
        env, gear_cfg, board_cfg, peg_offset, held_gear_base_offset, gear_peg_height, engage_z_fraction, xy_threshold,
    ).float()


def gear_insertion_success_bonus(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: list[float] = [0.0, 0.0, 0.0],
    held_gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    gear_peg_height: float = 0.02,
    success_z_fraction: float = 0.05,
    xy_threshold: float = 0.0025,
) -> torch.Tensor:
    """Bonus when the gear is fully inserted (binary success geometry)."""
    return _check_gear_position(
        env, gear_cfg, board_cfg, peg_offset, held_gear_base_offset, gear_peg_height, success_z_fraction, xy_threshold,
    ).float()


class osc_action_magnitude_penalty(ManagerTermBase):
    """Penalize large asset-relative position/rotation commands."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._pos_thresh: float = cfg.params.get("pos_action_threshold", 0.02)
        self._rot_thresh: float = cfg.params.get("rot_action_threshold", 0.097)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pos_action_threshold: float = 0.02,
        rot_action_threshold: float = 0.097,
    ) -> torch.Tensor:
        action_term = env.action_manager._terms["arm_action"]
        pos_error = torch.norm(action_term.delta_pos, p=2, dim=-1) / self._pos_thresh
        rot_error = torch.abs(action_term.delta_yaw) / self._rot_thresh
        return pos_error + rot_error


class osc_action_delta_penalty(ManagerTermBase):
    """Penalize jerky actions using smoothed action deltas."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = env.action_manager._terms["arm_action"]
        return torch.norm(
            action_term._smoothed_actions - action_term._prev_smoothed_actions,
            p=2,
            dim=-1,
        )


class wrist_contact_force_penalty(ManagerTermBase):
    """Penalize wrist/contact force magnitude above per-episode threshold."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = env.action_manager._terms["arm_action"]
        force_mag = torch.norm(action_term.force_smooth, p=2, dim=-1)
        return torch.nn.functional.relu(force_mag - action_term.contact_thresholds)


class success_prediction_error(ManagerTermBase):
    """Penalize incorrect success predictions from the 7th action dimension.

    ``true_success`` uses held insertion base vs target peg (same geometry as
    ``gear_insertion_success_bonus`` / ``gear_mesh_insertion_success``), not the gear
    rigid-body root, consistent with common assembly peg-insert benchmarks.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._pred_scale = 0.0
        self._delay_until_ratio: float = cfg.params.get("delay_until_ratio", 0.25)
        self._gear_cfg: SceneEntityCfg = cfg.params["gear_cfg"]
        self._board_cfg: SceneEntityCfg = cfg.params["board_cfg"]
        hgo = cfg.params.get("held_gear_base_offset", [2.025e-2, 0.0, 0.0])
        self._held_gear_base_offset = torch.tensor(hgo, device=env.device, dtype=torch.float32)
        self._peg_offset = torch.tensor(cfg.params.get("peg_offset", [0.0, 0.0, 0.0]), device=env.device)
        self._gear_peg_height: float = cfg.params.get("gear_peg_height", 0.02)
        self._success_z_fraction: float = cfg.params.get("success_z_fraction", 0.05)
        self._xy_threshold: float = cfg.params.get("xy_threshold", 0.0025)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
        board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        gear_peg_height: float = 0.02,
        success_z_fraction: float = 0.05,
        xy_threshold: float = 0.0025,
        delay_until_ratio: float = 0.25,
    ) -> torch.Tensor:
        gear: RigidObject = env.scene[self._gear_cfg.name]
        gear_pos = wp.to_torch(gear.data.root_pos_w) - env.scene.env_origins
        gear_quat = wp.to_torch(gear.data.root_quat_w)
        n = gear_pos.shape[0]
        held_off = self._held_gear_base_offset.unsqueeze(0).expand(n, 3)
        held_base_pos = gear_pos + quat_apply(gear_quat, held_off)

        board: RigidObject = env.scene[self._board_cfg.name]
        board_pos = wp.to_torch(board.data.root_pos_w) - env.scene.env_origins
        board_quat = wp.to_torch(board.data.root_quat_w)
        peg_off = self._peg_offset.unsqueeze(0).expand(n, 3)
        peg_pos = board_pos + quat_apply(board_quat, peg_off)

        xy_dist = torch.norm(held_base_pos[:, :2] - peg_pos[:, :2], dim=-1)
        z_diff = held_base_pos[:, 2] - peg_pos[:, 2]
        height_threshold = self._gear_peg_height * self._success_z_fraction
        true_success = (xy_dist < self._xy_threshold) & (z_diff < height_threshold)

        if true_success.float().mean() >= self._delay_until_ratio:
            self._pred_scale = 1.0

        arm_osc_action = env.action_manager._terms["arm_action"]
        pred = (arm_osc_action.success_pred + 1.0) / 2.0
        error = torch.abs(true_success.float() - pred)
        return error * self._pred_scale


