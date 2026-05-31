# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for gear insertion tasks.

The generic rewards in this module only depend on insertion geometry: keypoint
alignment around the peg and sparse bonuses for engagement or completion.
Controller-specific penalties live in the environment MDP package.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.direct.factory.factory_utils import get_keypoint_offsets, squashing_fn

from isaaclab_arena.tasks.nist_gear_insertion import geometry as gear_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _KeypointDistanceComputer:
    """Keypoint distance calculator with reusable buffers.

    Factory/Forge insertion rewards compare multiple keypoints around the
    target and held-object frames instead of comparing only frame origins. The
    keypoint offsets and identity quaternions are allocated once here to avoid
    per-step tensor construction.
    """

    def __init__(self, num_envs: int, device: torch.device, num_keypoints: int = 4):
        self.offsets_base = get_keypoint_offsets(num_keypoints, device)
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
        """Return the mean L2 distance between transformed keypoint sets."""
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


class gear_peg_keypoint_squashing(ManagerTermBase):
    """Factory-style keypoint reward for gear-to-peg alignment.

    The term aligns the held gear insertion frame to the target peg frame using
    a squashed keypoint distance. The squashing parameters are configured by the
    reward group to provide coarse-to-fine shaping with the same geometry.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._peg_offset_values = tuple(cfg.params["peg_offset"])
        self.peg_offset = torch.tensor(self._peg_offset_values, device=env.device, dtype=torch.float32)
        self._held_gear_base_offset_values = tuple(cfg.params["held_gear_base_offset"])
        self.held_gear_base_offset = torch.tensor(
            self._held_gear_base_offset_values, device=env.device, dtype=torch.float32
        )
        self._xy_noise_range = cfg.params.get("peg_offset_xy_noise", 0.0)
        self._num_keypoints: int = cfg.params.get("num_keypoints", 4)
        self.kp = _KeypointDistanceComputer(env.num_envs, env.device, num_keypoints=self._num_keypoints)
        self._offset_noise = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Sample the per-episode XY target perturbation."""
        if self._xy_noise_range <= 0.0:
            return
        if env_ids is None:
            env_ids = torch.arange(self._offset_noise.shape[0], device=self._offset_noise.device)
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        noise_dev = self._offset_noise.device
        self._offset_noise[env_ids, 0] = (torch.rand(n, device=noise_dev) * 2 - 1) * self._xy_noise_range
        self._offset_noise[env_ids, 1] = (torch.rand(n, device=noise_dev) * 2 - 1) * self._xy_noise_range

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg,
        board_cfg: SceneEntityCfg,
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        keypoint_scale: float = 0.15,
        num_keypoints: int = 4,
        peg_offset_xy_noise: float = 0.0,
        squash_a: float = 50.0,
        squash_b: float = 2.0,
    ) -> torch.Tensor:
        """Return the squashed keypoint alignment reward."""
        self._validate_num_keypoints(num_keypoints)
        held_base_pos, gear_quat = self._compute_held_base_pose(env, gear_cfg, held_gear_base_offset)
        target_pos, target_quat = self._compute_target_pose(env, board_cfg, peg_offset)
        target_pos = self._apply_target_noise(target_pos, peg_offset_xy_noise)
        return self._compute_squashed_keypoint_reward(
            target_pos,
            target_quat,
            held_base_pos,
            gear_quat,
            keypoint_scale,
            squash_a,
            squash_b,
        )

    def _validate_num_keypoints(self, num_keypoints: int) -> None:
        """Validate that the reward uses the keypoint layout allocated at construction."""
        if num_keypoints != self._num_keypoints:
            raise ValueError(
                f"num_keypoints is fixed at term initialization. Expected {self._num_keypoints}, got {num_keypoints}."
            )

    def _compute_held_base_pose(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg,
        held_gear_base_offset: list[float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the held gear insertion-point pose in each environment frame."""
        gear: RigidObject = env.scene[gear_cfg.name]
        held_gear_base_offset_tensor = gear_geometry.resolve_offset_tensor(
            held_gear_base_offset,
            self._held_gear_base_offset_values,
            self.held_gear_base_offset,
            env.device,
        )
        return gear_geometry.compute_asset_local_offset_pose(env, gear, held_gear_base_offset_tensor)

    def _compute_target_pose(
        self,
        env: ManagerBasedRLEnv,
        board_cfg: SceneEntityCfg,
        peg_offset: list[float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the target peg pose in each environment frame."""
        board: RigidObject = env.scene[board_cfg.name]
        peg_offset_tensor = gear_geometry.resolve_offset_tensor(
            peg_offset, self._peg_offset_values, self.peg_offset, env.device
        )
        return gear_geometry.compute_asset_local_offset_pose(env, board, peg_offset_tensor)

    def _apply_target_noise(self, target_pos: torch.Tensor, peg_offset_xy_noise: float) -> torch.Tensor:
        """Apply the per-reset XY target offset used for insertion robustness."""
        if peg_offset_xy_noise <= 0.0:
            return target_pos
        return target_pos + self._offset_noise[: target_pos.shape[0]]

    def _compute_squashed_keypoint_reward(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        held_base_pos: torch.Tensor,
        gear_quat: torch.Tensor,
        keypoint_scale: float,
        squash_a: float,
        squash_b: float,
    ) -> torch.Tensor:
        """Return the squashed mean distance between target and held-gear keypoints."""
        kp_dist = self.kp.compute(target_pos, target_quat, held_base_pos, gear_quat, scale=keypoint_scale)
        return squashing_fn(kp_dist, squash_a, squash_b)


class gear_insertion_geometry_bonus(ManagerTermBase):
    """Bonus when the gear satisfies the configured insertion geometry.

    Different reward terms reuse this class with different ``z_fraction`` values:
    one for early peg engagement and one for the final success depth.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._peg_offset_values = tuple(cfg.params["peg_offset"])
        self._peg_offset = torch.tensor(self._peg_offset_values, device=env.device, dtype=torch.float32)
        self._held_gear_base_offset_values = tuple(cfg.params["held_gear_base_offset"])
        self._held_gear_base_offset = torch.tensor(
            self._held_gear_base_offset_values, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg,
        board_cfg: SceneEntityCfg,
        gear_peg_height: float,
        z_fraction: float,
        xy_threshold: float,
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
    ) -> torch.Tensor:
        """Return a binary insertion-geometry bonus as a float tensor."""
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
            z_fraction,
            xy_threshold,
        ).float()
