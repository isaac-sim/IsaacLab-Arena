# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import random
import torch

import isaaclab.envs.mdp as mdp_isaac_lab
import warp as wp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.utils.pose import Pose

from .pose_utils import pose_range_from_quat


def reset_all_distractors_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distractor_names: list[str],
    pose_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Reset each listed distractor asset with random pose offset."""
    if pose_range is None:
        pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-np.pi / 6, np.pi / 6),
        }
    for name in distractor_names:
        if name not in env.scene.keys():
            continue
        mdp_isaac_lab.reset_root_state_uniform(
            env,
            env_ids,
            pose_range=pose_range,
            velocity_range={},
            asset_cfg=SceneEntityCfg(name),
        )


def set_object_pose_random_choice(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_choices: list[Pose],
    reference_cfg: SceneEntityCfg | None = None,
) -> None:
    """Reset an object to one of several discrete poses chosen per environment."""
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]

    ref_offset = None
    if reference_cfg is not None:
        reference = env.scene[reference_cfg.name]
        if hasattr(reference, "data") and hasattr(reference.data, "root_pos_w"):
            root_pos_w = reference.data.root_pos_w
            if not isinstance(root_pos_w, torch.Tensor):
                root_pos_w = wp.to_torch(root_pos_w)
            ref_offset = root_pos_w - env.scene.env_origins
        else:
            raw = reference.get_world_poses()[0]
            ref_pos_w = (
                (raw.detach().clone() if isinstance(raw, torch.Tensor) else torch.tensor(raw, device=env.device))
                .reshape(-1, 3)
                .to(env.device)
            )
            ref_offset = ref_pos_w - env.scene.env_origins

    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        pose = random.choice(pose_choices)
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device)
        if ref_offset is not None:
            pose_t_xyz_q_xyzw[:3] += ref_offset[cur_env]
        pose_t_xyz_q_xyzw[:3] += env.scene.env_origins[cur_env, :].squeeze()
        cur_ids = torch.tensor([cur_env], device=env.device)
        if hasattr(asset, "write_root_pose_to_sim"):
            asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw.unsqueeze(0), env_ids=cur_ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=cur_ids)
        else:
            asset.set_world_poses(
                positions=pose_t_xyz_q_xyzw[:3].unsqueeze(0),
                orientations=pose_t_xyz_q_xyzw[3:].unsqueeze(0),
                indices=cur_ids,
            )


def set_relative_initial_pose(
    obj,
    reference,
    offset_range: dict[str, tuple[float, float]],
    rotation_xyzw: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    yaw_jitter: float | tuple[float, float] = 0.0,
    roll_jitter: float | tuple[float, float] = 0.0,
    pitch_jitter: float | tuple[float, float] = 0.0,
) -> None:
    """Configure an object to reset relative to a reference object's current pose."""
    from isaaclab.utils.math import euler_xyz_from_quat

    ref_pose = reference._get_initial_pose_as_pose()
    ref_pos = (0.0, 0.0, 0.0) if ref_pose is None else ref_pose.position_xyz

    ox = offset_range.get("x", (0.0, 0.0))
    oy = offset_range.get("y", (0.0, 0.0))
    oz = offset_range.get("z", (0.0, 0.0))

    obj.set_initial_pose(
        pose_range_from_quat(
            position_xyz_min=(ref_pos[0] + ox[0], ref_pos[1] + oy[0], ref_pos[2] + oz[0]),
            position_xyz_max=(ref_pos[0] + ox[1], ref_pos[1] + oy[1], ref_pos[2] + oz[1]),
            rotation_xyzw=rotation_xyzw,
            yaw_jitter=yaw_jitter,
            roll_jitter=roll_jitter,
            pitch_jitter=pitch_jitter,
        )
    )

    q = torch.tensor(rotation_xyzw).unsqueeze(0)
    roll, pitch, yaw = (v.item() for v in euler_xyz_from_quat(q))

    def _jitter_range(center: float, jitter: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(jitter, (list, tuple)):
            return (center + jitter[0], center + jitter[1])
        return (center - jitter, center + jitter)

    obj.event_cfg = EventTermCfg(
        func=reset_object_relative_to,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(obj.name),
            "reference_cfg": SceneEntityCfg(reference.name),
            "offset_range": {
                "x": ox,
                "y": oy,
                "z": oz,
                "roll": _jitter_range(roll, roll_jitter),
                "pitch": _jitter_range(pitch, pitch_jitter),
                "yaw": _jitter_range(yaw, yaw_jitter),
            },
        },
    )


def reset_object_relative_to(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg,
    offset_range: dict[str, tuple[float, float]],
) -> None:
    """Reset an object's position relative to another object's current position."""
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    reference = env.scene[reference_cfg.name]

    if hasattr(reference, "data") and hasattr(reference.data, "root_pos_w"):
        root_pos_w = reference.data.root_pos_w
        if not isinstance(root_pos_w, torch.Tensor):
            root_pos_w = torch.as_tensor(root_pos_w, device=env.device)
        ref_pos = root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    else:
        raw = reference.get_world_poses()[0]
        ref_pos_w = (
            (raw.detach().clone() if isinstance(raw, torch.Tensor) else torch.tensor(raw, device=env.device))
            .reshape(-1, 3)
            .to(env.device)
        )
        ref_pos = (ref_pos_w - env.scene.env_origins)[env_ids]

    num_envs = len(env_ids)
    x_off = torch.empty(num_envs, device=env.device).uniform_(*offset_range.get("x", (0.0, 0.0)))
    y_off = torch.empty(num_envs, device=env.device).uniform_(*offset_range.get("y", (0.0, 0.0)))
    z_off = torch.empty(num_envs, device=env.device).uniform_(*offset_range.get("z", (0.0, 0.0)))
    roll = torch.empty(num_envs, device=env.device).uniform_(*offset_range.get("roll", (0.0, 0.0)))
    pitch = torch.empty(num_envs, device=env.device).uniform_(*offset_range.get("pitch", (0.0, 0.0)))
    yaw = torch.empty(num_envs, device=env.device).uniform_(*offset_range.get("yaw", (0.0, 0.0)))

    from isaaclab.utils.math import quat_from_euler_xyz

    pos = ref_pos + torch.stack([x_off, y_off, z_off], dim=-1)
    pos += env.scene.env_origins[env_ids]
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    if hasattr(asset, "write_root_pose_to_sim"):
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)
    else:
        asset.set_world_poses(positions=pos, orientations=quat, indices=env_ids)
