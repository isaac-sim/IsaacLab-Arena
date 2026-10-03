# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from functools import partial
from typing import Literal

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers import (
    EventTermCfg,
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModel, NoiseModelCfg, UniformNoiseCfg

from isaaclab_arena.assets import displayport_insertion_geometry as geometry
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
from isaaclab_arena.tasks.task_base import TaskBase

DisplayPortProfile = Literal["insertion", "passive_drop_test"]
CurriculumMode = Literal[
    "disabled",
    "fixed80",
    "anneal_80_0_500",
    "anneal_80_0_1000",
    "anneal_80_20_500",
    "anneal_80_20_1000",
]

_DEFAULT_KP_EXP_COEFFS = [(50, 0.0001), (300, 0.0001), (600, 0.0001), (2000, 0.0001)]


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    return wp.to_torch(value)


def _asset_root_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    data = asset.data
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        return _to_torch(data.root_pos_w), _to_torch(data.root_quat_w)
    root_pose_w = _to_torch(data.root_pose_w)
    return root_pose_w[:, :3], root_pose_w[:, 3:7]


def _identity_quat(env: ManagerBasedEnv, count: int | None = None) -> torch.Tensor:
    count = env.num_envs if count is None else count
    quat = torch.zeros((count, 4), device=env.device, dtype=torch.float32)
    quat[:, 3] = 1.0
    return quat


def _offset_tensor(env: ManagerBasedEnv, offset: tuple[float, float, float] | list[float], count: int | None = None):
    count = env.num_envs if count is None else count
    return torch.tensor(offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(count, -1)


def displayport_object_pos_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    offset: tuple[float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Return root or offset object position relative to each environment origin."""
    pos_w, quat_w = _asset_root_pose(env, asset_cfg)
    if offset is not None:
        pos_w, _ = math_utils.combine_frame_transforms(pos_w, quat_w, _offset_tensor(env, offset), _identity_quat(env))
    return pos_w - env.scene.env_origins


def displayport_object_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return object root quaternion in world frame."""
    _, quat_w = _asset_root_pose(env, asset_cfg)
    positive_quat = quat_w.clone()
    positive_quat[quat_w[:, 3] < 0] = -quat_w[quat_w[:, 3] < 0]
    return positive_quat


def _mate_frames(
    env: ManagerBasedEnv,
    socket_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
    socket_offset: tuple[float, float, float] | list[float] = geometry.SOCKET_INSERTION_OFFSET,
    plug_offset: tuple[float, float, float] | list[float] = geometry.PLUG_INSERTION_OFFSET,
    plug_goal_rot_inv: tuple[float, float, float, float] | list[float] = geometry.PLUG_GOAL_ROT_INV,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    socket_pos, socket_quat = _asset_root_pose(env, socket_cfg)
    plug_pos, plug_quat = _asset_root_pose(env, plug_cfg)
    socket_mate_pos, socket_mate_quat = math_utils.combine_frame_transforms(
        socket_pos,
        socket_quat,
        _offset_tensor(env, socket_offset),
        _identity_quat(env),
    )
    plug_goal_rot_inv_batch = torch.tensor(plug_goal_rot_inv, device=env.device, dtype=torch.float32).repeat(
        env.num_envs, 1
    )
    plug_mate_pos, plug_mate_quat = math_utils.combine_frame_transforms(
        plug_pos,
        plug_quat,
        _offset_tensor(env, plug_offset),
        plug_goal_rot_inv_batch,
    )
    return socket_mate_pos, socket_mate_quat, plug_mate_pos, plug_mate_quat


def displayport_mate_pos_error(
    env: ManagerBasedEnv,
    socket_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
    socket_offset: tuple[float, float, float] | list[float] = geometry.SOCKET_INSERTION_OFFSET,
    plug_offset: tuple[float, float, float] | list[float] = geometry.PLUG_INSERTION_OFFSET,
    plug_goal_rot_inv: tuple[float, float, float, float] | list[float] = geometry.PLUG_GOAL_ROT_INV,
) -> torch.Tensor:
    """Return mate-point position error between socket and plug."""
    socket_pos, _, plug_pos, _ = _mate_frames(env, socket_cfg, plug_cfg, socket_offset, plug_offset, plug_goal_rot_inv)
    return torch.linalg.norm(plug_pos - socket_pos, dim=-1)


def _keypoint_offsets(device: torch.device, keypoint_scale: float) -> torch.Tensor:
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0) * keypoint_scale


def displayport_mate_pos_error_obs(
    env: ManagerBasedEnv,
    socket_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return mate-point position error as a one-column observation term."""
    return displayport_mate_pos_error(env, socket_cfg, plug_cfg).unsqueeze(-1)


def displayport_keypoint_distances(
    env: ManagerBasedEnv,
    socket_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
    keypoint_scale: float = 0.15,
    socket_offset: tuple[float, float, float] | list[float] = geometry.SOCKET_INSERTION_OFFSET,
    plug_offset: tuple[float, float, float] | list[float] = geometry.PLUG_INSERTION_OFFSET,
    plug_goal_rot_inv: tuple[float, float, float, float] | list[float] = geometry.PLUG_GOAL_ROT_INV,
) -> torch.Tensor:
    """Return per-keypoint distances between aligned mate-frame keypoints."""
    socket_pos, socket_quat, plug_pos, plug_quat = _mate_frames(
        env, socket_cfg, plug_cfg, socket_offset, plug_offset, plug_goal_rot_inv
    )
    offsets = _keypoint_offsets(env.device, keypoint_scale)
    num_keypoints = offsets.shape[0]
    offsets_flat = offsets.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3)
    ident_flat = _identity_quat(env, env.num_envs).unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4)
    socket_points = math_utils.combine_frame_transforms(
        socket_pos.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 3),
        socket_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4),
        offsets_flat,
        ident_flat,
    )[0].reshape(env.num_envs, num_keypoints, 3)
    plug_points = math_utils.combine_frame_transforms(
        plug_pos.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 3),
        plug_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4),
        offsets_flat,
        ident_flat,
    )[0].reshape(env.num_envs, num_keypoints, 3)
    return torch.linalg.norm(plug_points - socket_points, dim=-1)


def displayport_keypoint_dist(
    env: ManagerBasedEnv,
    socket_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
    keypoint_scale: float = 0.15,
    socket_offset: tuple[float, float, float] | list[float] = geometry.SOCKET_INSERTION_OFFSET,
    plug_offset: tuple[float, float, float] | list[float] = geometry.PLUG_INSERTION_OFFSET,
    plug_goal_rot_inv: tuple[float, float, float, float] | list[float] = geometry.PLUG_GOAL_ROT_INV,
) -> torch.Tensor:
    """Return mean distance between 7 aligned mate-frame keypoints."""
    return displayport_keypoint_distances(
        env,
        socket_cfg,
        plug_cfg,
        keypoint_scale=keypoint_scale,
        socket_offset=socket_offset,
        plug_offset=plug_offset,
        plug_goal_rot_inv=plug_goal_rot_inv,
    ).mean(dim=-1)


def displayport_mate_success(
    env: ManagerBasedEnv,
    socket_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
    pos_threshold: float = 0.003,
    keypoint_threshold: float | None = None,
    keypoint_scale: float = 0.15,
) -> torch.Tensor:
    """Return non-terminating DisplayPort insertion success flags."""
    success = displayport_mate_pos_error(env, socket_cfg, plug_cfg) < pos_threshold
    if keypoint_threshold is not None:
        success &= (
            displayport_keypoint_dist(env, socket_cfg, plug_cfg, keypoint_scale=keypoint_scale) < keypoint_threshold
        )
    return success


def _compute_episode_displayport_fields(
    env: ManagerBasedEnv,
    env_id: int,
    socket_name: str,
    plug_name: str,
    success_pos_threshold: float,
    keypoint_scale: float,
) -> dict[str, float | bool]:
    socket_cfg = SceneEntityCfg(socket_name)
    plug_cfg = SceneEntityCfg(plug_name)
    pos_error = displayport_mate_pos_error(env, socket_cfg, plug_cfg)
    keypoint_dist = displayport_keypoint_dist(env, socket_cfg, plug_cfg, keypoint_scale=keypoint_scale)
    success = pos_error < success_pos_threshold
    return {
        "success": bool(success[env_id].item()),
        "mate_pos_error_m": float(pos_error[env_id].item()),
        "mate_keypoint_dist_m": float(keypoint_dist[env_id].item()),
    }


def record_displayport_insertion_episode(
    env: ManagerBasedEnv,
    env_id: int,
    socket_name: str,
    plug_name: str,
    success_pos_threshold: float,
    keypoint_scale: float,
) -> dict[str, dict[str, float | bool]]:
    """Record DisplayPort final mate state for one finishing episode."""
    return {
        "displayport_insertion": _compute_episode_displayport_fields(
            env, env_id, socket_name, plug_name, success_pos_threshold, keypoint_scale
        )
    }


class DisplayPortKeypointError(ManagerTermBase):
    """Reward term returning mate-frame keypoint distance."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.socket_cfg: SceneEntityCfg = cfg.params["socket_cfg"]
        self.plug_cfg: SceneEntityCfg = cfg.params["plug_cfg"]
        self.keypoint_scale: float = cfg.params.get("keypoint_scale", 0.15)

    def __call__(
        self,
        env: ManagerBasedEnv,
        socket_cfg: SceneEntityCfg,
        plug_cfg: SceneEntityCfg,
        keypoint_scale: float = 0.15,
    ) -> torch.Tensor:
        return displayport_keypoint_dist(env, self.socket_cfg, self.plug_cfg, keypoint_scale=self.keypoint_scale)


class DisplayPortKeypointErrorExp(DisplayPortKeypointError):
    """Exponential mate-frame keypoint reward."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.kp_exp_coeffs: list[tuple[float, float]] = cfg.params.get("kp_exp_coeffs", _DEFAULT_KP_EXP_COEFFS)
        self.kp_use_sum_of_exps: bool = cfg.params.get("kp_use_sum_of_exps", False)

    def __call__(
        self,
        env: ManagerBasedEnv,
        socket_cfg: SceneEntityCfg,
        plug_cfg: SceneEntityCfg,
        keypoint_scale: float = 0.15,
        kp_exp_coeffs: list[tuple[float, float]] | None = None,
        kp_use_sum_of_exps: bool | None = None,
    ) -> torch.Tensor:
        dist_sep = displayport_keypoint_distances(
            env, self.socket_cfg, self.plug_cfg, keypoint_scale=self.keypoint_scale
        )
        reward = torch.zeros_like(dist_sep[:, 0])
        coeffs = self.kp_exp_coeffs if kp_exp_coeffs is None else kp_exp_coeffs
        use_sum_of_exps = self.kp_use_sum_of_exps if kp_use_sum_of_exps is None else kp_use_sum_of_exps
        if use_sum_of_exps:
            for a, b in coeffs:
                reward += (1.0 / (torch.exp(a * dist_sep) + b + torch.exp(-a * dist_sep))).mean(-1)
        else:
            dist = dist_sep.mean(-1)
            for a, b in coeffs:
                reward += 1.0 / (torch.exp(a * dist) + b + torch.exp(-a * dist))
        return reward


class ResetDisplayPortPlugCurriculum(ManagerTermBase):
    """Reset the DisplayPort plug along the socket insertion axis."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.plug_cfg: SceneEntityCfg = cfg.params["plug_cfg"]
        self.socket_cfg: SceneEntityCfg = cfg.params["socket_cfg"]
        self.plug = env.scene[self.plug_cfg.name]
        self.socket = env.scene[self.socket_cfg.name]
        self.at_goal_prob: float = cfg.params.get("at_goal_prob", 0.0)
        self.at_goal_prob_final: float | None = cfg.params.get("at_goal_prob_final")
        self.anneal_start_iter: float = cfg.params.get("anneal_start_iter", 0.0)
        self.anneal_end_iter: float | None = cfg.params.get("anneal_end_iter")
        self.num_steps_per_env: int | None = cfg.params.get("num_steps_per_env")
        self.insertion_axis = torch.tensor(cfg.params["insertion_axis"], device=env.device, dtype=torch.float32)
        self.insertion_axis = self.insertion_axis / self.insertion_axis.norm()
        self.socket_offset = torch.tensor(cfg.params["socket_insertion_offset"], device=env.device, dtype=torch.float32)
        self.plug_offset = torch.tensor(cfg.params["plug_insertion_offset"], device=env.device, dtype=torch.float32)
        self.goal_rot = torch.tensor(cfg.params["goal_rot"], device=env.device, dtype=torch.float32)
        self.normal_pose_range: dict[str, tuple[float, float]] = cfg.params.get("normal_pose_range", {})
        self.at_goal_depth_range = cfg.params.get("at_goal_depth_range", (0.0, 0.015))
        self.approach_depth_range = cfg.params.get("approach_depth_range", (0.02, 0.06))

    def _current_at_goal_prob(self, env: ManagerBasedEnv) -> float:
        if self.at_goal_prob_final is None or self.anneal_end_iter is None or not self.num_steps_per_env:
            return self.at_goal_prob
        current_iter = env.common_step_counter / float(self.num_steps_per_env)
        span = max(float(self.anneal_end_iter) - float(self.anneal_start_iter), 1e-9)
        frac = min(max((current_iter - float(self.anneal_start_iter)) / span, 0.0), 1.0)
        return self.at_goal_prob + frac * (float(self.at_goal_prob_final) - self.at_goal_prob)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        plug_cfg: SceneEntityCfg,
        socket_cfg: SceneEntityCfg,
        at_goal_prob: float = 0.0,
        at_goal_prob_final: float | None = None,
        anneal_start_iter: float = 0.0,
        anneal_end_iter: float | None = None,
        num_steps_per_env: int | None = None,
        insertion_axis: list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0),
        socket_insertion_offset: list[float] | tuple[float, float, float] = geometry.SOCKET_INSERTION_OFFSET,
        plug_insertion_offset: list[float] | tuple[float, float, float] = geometry.PLUG_INSERTION_OFFSET,
        goal_rot: list[float] | tuple[float, float, float, float] = geometry.PLUG_GOAL_ROT,
        at_goal_depth_range: list[float] | tuple[float, float] = (0.0, 0.015),
        approach_depth_range: list[float] | tuple[float, float] = (0.02, 0.06),
        normal_pose_range: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        num_envs = len(env_ids)
        socket_pos = _to_torch(self.socket.data.root_pos_w)[env_ids]
        socket_quat = _to_torch(self.socket.data.root_quat_w)[env_ids]

        id_quat = _identity_quat(env, num_envs)
        socket_offset = self.socket_offset.unsqueeze(0).expand(num_envs, -1)
        mate_origin_w, _ = math_utils.combine_frame_transforms(socket_pos, socket_quat, socket_offset, id_quat)
        insertion_axis_w = math_utils.quat_apply(socket_quat, self.insertion_axis.unsqueeze(0).expand(num_envs, -1))
        goal_quat_w = math_utils.quat_mul(socket_quat, self.goal_rot.unsqueeze(0).expand(num_envs, -1))
        plug_offset_w = math_utils.quat_apply(goal_quat_w, self.plug_offset.unsqueeze(0).expand(num_envs, -1))

        at_goal_mask = torch.rand(num_envs, device=env.device) < self._current_at_goal_prob(env)
        depth_at_goal = torch.empty(num_envs, device=env.device).uniform_(
            float(self.at_goal_depth_range[0]), float(self.at_goal_depth_range[1])
        )
        depth_approach = torch.empty(num_envs, device=env.device).uniform_(
            float(self.approach_depth_range[0]), float(self.approach_depth_range[1])
        )
        depth = torch.where(at_goal_mask, depth_at_goal, depth_approach)

        rand_pos = torch.zeros(num_envs, 3, device=env.device)
        pose_range = self.normal_pose_range
        for axis, key in enumerate(("x", "y", "z")):
            low, high = pose_range.get(key, (0.0, 0.0))
            rand_pos[:, axis] = torch.empty(num_envs, device=env.device).uniform_(low, high)
        rand_pos[at_goal_mask] = 0.0

        plug_pos = mate_origin_w + depth.unsqueeze(-1) * insertion_axis_w - plug_offset_w + rand_pos
        root_pose = torch.cat([plug_pos, goal_quat_w], dim=-1)
        self.plug.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.plug.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


class DisplayPortInsertionSuccessRecorder(RecorderTerm):
    """Record non-terminating DisplayPort success at episode end."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.socket_name = cfg.socket_name
        self.plug_name = cfg.plug_name
        self.success_pos_threshold = cfg.success_pos_threshold
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        if self.first_reset:
            self.first_reset = False
            return None, None
        success = displayport_mate_success(
            self._env,
            SceneEntityCfg(self.socket_name),
            SceneEntityCfg(self.plug_name),
            pos_threshold=self.success_pos_threshold,
        )
        return self.name, success[env_ids]


@configclass
class DisplayPortInsertionSuccessRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = DisplayPortInsertionSuccessRecorder
    name: str = "displayport_insertion_success"
    socket_name: str = MISSING
    plug_name: str = MISSING
    success_pos_threshold: float = 0.003


def compute_displayport_insertion_success_rate(recorded_metric_data: list[np.ndarray]) -> float:
    """Compute episode-end DisplayPort insertion success rate."""
    if not recorded_metric_data:
        return 0.0
    return float(np.concatenate(recorded_metric_data).mean())


class DisplayPortInsertionSuccessRateMetric(MetricBase):
    """Episode-end success rate for DisplayPort insertion without terminating on success."""

    name = "displayport_insertion_success_rate"
    recorder_term_name = "displayport_insertion_success"

    def __init__(self, socket_name: str, plug_name: str, success_pos_threshold: float = 0.003):
        self.socket_name = socket_name
        self.plug_name = plug_name
        self.success_pos_threshold = success_pos_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return DisplayPortInsertionSuccessRecorderCfg(
            name=self.recorder_term_name,
            socket_name=self.socket_name,
            plug_name=self.plug_name,
            success_pos_threshold=self.success_pos_threshold,
        )

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_displayport_insertion_success_rate,
            params={},
            recorder_term_name=self.recorder_term_name,
        )


class ResetSampledConstantNoiseModel(NoiseModel):
    """Noise model that samples one additive noise value per env reset."""

    def __init__(self, noise_model_cfg: NoiseModelCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self._sampled_noise = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        num_resets = env_ids.stop - env_ids.start if isinstance(env_ids, slice) else len(env_ids)
        dummy_data = torch.zeros((num_resets, 1), device=self._device)
        sampled_noise = self._noise_model_cfg.noise_cfg.func(dummy_data, self._noise_model_cfg.noise_cfg)
        self._sampled_noise[env_ids] = sampled_noise

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self._num_components is None:
            self._num_components = data.shape[-1]
            self._sampled_noise = self._sampled_noise.repeat(1, self._num_components)

        noise_cfg = self._noise_model_cfg.noise_cfg
        if noise_cfg.operation == "add":
            return data + self._sampled_noise
        if noise_cfg.operation == "scale":
            return data * self._sampled_noise
        if noise_cfg.operation == "abs":
            return self._sampled_noise
        raise ValueError(f"Unknown operation in noise: {noise_cfg.operation}")


@configclass
class ResetSampledConstantNoiseModelCfg(NoiseModelCfg):
    """Configure reset-sampled constant observation noise."""

    class_type: type[NoiseModel] = ResetSampledConstantNoiseModel
    noise_cfg: NoiseCfg = MISSING


@configclass
class DisplayPortActiveObservationsCfg:
    """DisplayPort active insertion observations."""

    policy: ObservationGroupCfg = MISSING
    critic: ObservationGroupCfg = MISSING

    def __init__(self, socket_name: str, plug_name: str):
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            socket_pos = ObservationTermCfg(
                func=displayport_object_pos_w,
                params={"asset_cfg": SceneEntityCfg(socket_name), "offset": geometry.SOCKET_INSERTION_OFFSET},
                noise=ResetSampledConstantNoiseModelCfg(
                    noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
                ),
            )
            socket_quat = ObservationTermCfg(
                func=displayport_object_quat_w,
                params={"asset_cfg": SceneEntityCfg(socket_name)},
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class CriticCfg(ObservationGroupCfg):
            socket_pos = ObservationTermCfg(
                func=displayport_object_pos_w,
                params={"asset_cfg": SceneEntityCfg(socket_name), "offset": geometry.SOCKET_INSERTION_OFFSET},
            )
            socket_quat = ObservationTermCfg(
                func=displayport_object_quat_w,
                params={"asset_cfg": SceneEntityCfg(socket_name)},
            )
            plug_pos = ObservationTermCfg(
                func=displayport_object_pos_w,
                params={"asset_cfg": SceneEntityCfg(plug_name), "offset": geometry.PLUG_INSERTION_OFFSET},
            )
            plug_quat = ObservationTermCfg(
                func=displayport_object_quat_w,
                params={"asset_cfg": SceneEntityCfg(plug_name)},
            )

        self.policy = PolicyCfg()
        self.critic = CriticCfg()


@configclass
class DisplayPortPassiveObservationsCfg:
    """DisplayPort passive drop-test observations."""

    policy: ObservationGroupCfg = MISSING

    def __init__(self, socket_name: str, plug_name: str):
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            plug_pos = ObservationTermCfg(
                func=displayport_object_pos_w,
                params={"asset_cfg": SceneEntityCfg(plug_name)},
            )
            plug_quat = ObservationTermCfg(
                func=displayport_object_quat_w,
                params={"asset_cfg": SceneEntityCfg(plug_name)},
            )
            socket_pos = ObservationTermCfg(
                func=displayport_object_pos_w,
                params={"asset_cfg": SceneEntityCfg(socket_name)},
            )
            socket_quat = ObservationTermCfg(
                func=displayport_object_quat_w,
                params={"asset_cfg": SceneEntityCfg(socket_name)},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.policy = PolicyCfg()


@configclass
class DisplayPortRewardsCfg:
    """DisplayPort connector rewards."""

    plug_socket_keypoint_tracking: RewardTermCfg = MISSING
    plug_socket_keypoint_tracking_exp: RewardTermCfg = MISSING

    def __init__(
        self,
        socket_name: str,
        plug_name: str,
        keypoint_scale: float,
        exp_reward_weight: float = 1.5,
    ):
        self.plug_socket_keypoint_tracking = RewardTermCfg(
            func=DisplayPortKeypointError,
            weight=-1.5,
            params={
                "socket_cfg": SceneEntityCfg(socket_name),
                "plug_cfg": SceneEntityCfg(plug_name),
                "keypoint_scale": keypoint_scale,
            },
        )
        self.plug_socket_keypoint_tracking_exp = RewardTermCfg(
            func=DisplayPortKeypointErrorExp,
            weight=exp_reward_weight,
            params={
                "socket_cfg": SceneEntityCfg(socket_name),
                "plug_cfg": SceneEntityCfg(plug_name),
                "kp_exp_coeffs": _DEFAULT_KP_EXP_COEFFS,
                "kp_use_sum_of_exps": False,
                "keypoint_scale": keypoint_scale,
            },
        )


@configclass
class DisplayPortTerminationsCfg:
    """DisplayPort connector terminations."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class DisplayPortInsertionEventsCfg:
    """DisplayPort connector insertion events."""

    reset_all: EventTermCfg = MISSING
    plug_physics_material: EventTermCfg = MISSING
    socket_physics_material: EventTermCfg = MISSING
    randomize_socket_pose: EventTermCfg = MISSING
    reset_plug_curriculum: EventTermCfg = MISSING


@configclass
class DisplayPortPassiveEventsCfg:
    """Passive drop-test reset events."""

    reset_all: EventTermCfg = MISSING


@configclass
class DisplayPortInsertionEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Episode recorder for DisplayPort insertion diagnostics."""

    func: Callable[..., dict[str, dict[str, float | bool]]] = record_displayport_insertion_episode


def _curriculum_params(mode: CurriculumMode) -> dict[str, float | None]:
    table = {
        "disabled": {"at_goal_prob": 0.0, "at_goal_prob_final": None, "anneal_end_iter": None},
        "fixed80": {"at_goal_prob": 0.8, "at_goal_prob_final": None, "anneal_end_iter": None},
        "anneal_80_0_500": {"at_goal_prob": 0.8, "at_goal_prob_final": 0.0, "anneal_end_iter": 500.0},
        "anneal_80_0_1000": {"at_goal_prob": 0.8, "at_goal_prob_final": 0.0, "anneal_end_iter": 1000.0},
        "anneal_80_20_500": {"at_goal_prob": 0.8, "at_goal_prob_final": 0.2, "anneal_end_iter": 500.0},
        "anneal_80_20_1000": {"at_goal_prob": 0.8, "at_goal_prob_final": 0.2, "anneal_end_iter": 1000.0},
    }
    assert mode in table, f"Unknown DisplayPort curriculum mode {mode!r}. Options: {sorted(table)}"
    return table[mode]


@register_task
class DisplayPortInsertionTask(TaskBase):
    """Connector-only DisplayPort insertion simulation task."""

    def __init__(
        self,
        plug: Asset,
        socket: Asset,
        profile: DisplayPortProfile = "insertion",
        socket_pos_range: tuple[float, float, float] = (0.01, 0.01, 0.02),
        socket_orn_deg: float = 2.0,
        curriculum_mode: CurriculumMode = "anneal_80_0_500",
        at_goal_depth_range: tuple[float, float] = (0.0, 0.015),
        approach_depth_range: tuple[float, float] = (0.02, 0.06),
        success_pos_threshold: float = 0.003,
        keypoint_scale: float = 0.15,
        exp_reward_weight: float = 1.5,
        episode_length_s: float | None = None,
    ):
        if episode_length_s is None:
            episode_length_s = 10.0 if profile == "passive_drop_test" else 6.66
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Insert the DisplayPort plug into the DisplayPort socket.",
        )
        self.plug = plug
        self.socket = socket
        self.profile = profile
        self.socket_pos_range = socket_pos_range
        self.socket_orn_deg = socket_orn_deg
        self.curriculum_mode = curriculum_mode
        self.at_goal_depth_range = at_goal_depth_range
        self.approach_depth_range = approach_depth_range
        self.success_pos_threshold = success_pos_threshold
        self.keypoint_scale = keypoint_scale
        self.exp_reward_weight = exp_reward_weight
        self.scene_config = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def get_scene_cfg(self):
        return self.scene_config

    def get_observation_cfg(self):
        if self.profile == "passive_drop_test":
            return DisplayPortPassiveObservationsCfg(socket_name=self.socket.name, plug_name=self.plug.name)
        return DisplayPortActiveObservationsCfg(socket_name=self.socket.name, plug_name=self.plug.name)

    def get_rewards_cfg(self):
        if self.profile != "insertion":
            return None
        return DisplayPortRewardsCfg(
            socket_name=self.socket.name,
            plug_name=self.plug.name,
            keypoint_scale=self.keypoint_scale,
            exp_reward_weight=self.exp_reward_weight,
        )

    def get_termination_cfg(self):
        return DisplayPortTerminationsCfg()

    def get_events_cfg(self):
        if self.profile == "passive_drop_test":
            return DisplayPortPassiveEventsCfg(
                reset_all=EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
            )
        curriculum = _curriculum_params(self.curriculum_mode)
        return DisplayPortInsertionEventsCfg(
            reset_all=EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
            plug_physics_material=EventTermCfg(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(self.plug.name, body_names=".*"),
                    "static_friction_range": (0.001, 0.001),
                    "dynamic_friction_range": (0.001, 0.001),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            ),
            socket_physics_material=EventTermCfg(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(self.socket.name, body_names=".*"),
                    "static_friction_range": (0.001, 0.001),
                    "dynamic_friction_range": (0.001, 0.001),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            ),
            randomize_socket_pose=EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-self.socket_pos_range[0], self.socket_pos_range[0]),
                        "y": (-self.socket_pos_range[1], self.socket_pos_range[1]),
                        "z": (-self.socket_pos_range[2], self.socket_pos_range[2]),
                        "roll": (-math.radians(self.socket_orn_deg), math.radians(self.socket_orn_deg)),
                        "pitch": (-math.radians(self.socket_orn_deg), math.radians(self.socket_orn_deg)),
                        "yaw": (-math.radians(self.socket_orn_deg), math.radians(self.socket_orn_deg)),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(self.socket.name),
                },
            ),
            reset_plug_curriculum=EventTermCfg(
                func=ResetDisplayPortPlugCurriculum,
                mode="reset",
                params={
                    "plug_cfg": SceneEntityCfg(self.plug.name),
                    "socket_cfg": SceneEntityCfg(self.socket.name),
                    "at_goal_prob": curriculum["at_goal_prob"],
                    "at_goal_prob_final": curriculum["at_goal_prob_final"],
                    "anneal_start_iter": 0.0,
                    "anneal_end_iter": curriculum["anneal_end_iter"],
                    "num_steps_per_env": 512,
                    "insertion_axis": [1.0, 0.0, 0.0],
                    "socket_insertion_offset": geometry.SOCKET_INSERTION_OFFSET,
                    "plug_insertion_offset": geometry.PLUG_INSERTION_OFFSET,
                    "goal_rot": geometry.PLUG_GOAL_ROT,
                    "at_goal_depth_range": self.at_goal_depth_range,
                    "approach_depth_range": self.approach_depth_range,
                    "normal_pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
                },
            ),
        )

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("DisplayPortInsertionTask does not define a mimic environment.")

    def get_metrics(self) -> list[MetricBase]:
        return [
            DisplayPortInsertionSuccessRateMetric(
                socket_name=self.socket.name,
                plug_name=self.plug.name,
                success_pos_threshold=self.success_pos_threshold,
            )
        ]

    def get_progress_objectives(self) -> list[ProgressObjective]:
        socket_cfg = SceneEntityCfg(self.socket.name)
        plug_cfg = SceneEntityCfg(self.plug.name)
        return [
            ProgressObjective(
                name="displayport_insertion",
                predicate_groups=[
                    partial(displayport_mate_success, socket_cfg=socket_cfg, plug_cfg=plug_cfg, pos_threshold=0.06),
                    partial(displayport_mate_success, socket_cfg=socket_cfg, plug_cfg=plug_cfg, pos_threshold=0.015),
                    partial(
                        displayport_mate_success,
                        socket_cfg=socket_cfg,
                        plug_cfg=plug_cfg,
                        pos_threshold=self.success_pos_threshold,
                        keypoint_threshold=self.success_pos_threshold,
                        keypoint_scale=self.keypoint_scale,
                    ),
                ],
            )
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.profile == "passive_drop_test":
            return ViewerCfg(eye=(0.3, 0.3, 0.3), lookat=(0.0, 0.0, geometry.PASSIVE_DROP_SOCKET_POS[2]))
        return ViewerCfg(eye=(0.5, -1.8, 1.2), lookat=(0.5, 0.0, 0.5))

    def get_episode_recorder_term_cfg(self) -> EpisodeRecorderTermCfg:
        return DisplayPortInsertionEpisodeRecorderTermCfg(
            params={
                "socket_name": self.socket.name,
                "plug_name": self.plug.name,
                "success_pos_threshold": self.success_pos_threshold,
                "keypoint_scale": self.keypoint_scale,
            }
        )


def make_displayport_episode_recorder_term(task: DisplayPortInsertionTask) -> EpisodeRecorderTermCfg:
    """Return the DisplayPort episode recorder term for environment factories."""
    return task.get_episode_recorder_term_cfg()
