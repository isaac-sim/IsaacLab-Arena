# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply precomputed layouts (e.g. from relation solver) to env at reset."""

import math
import random
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.terms.events import set_object_pose_per_env
from isaaclab_arena.utils.pose import Pose


def _yaw_z_to_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for rotation by yaw_rad around Z."""
    half = yaw_rad * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def apply_layout_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    layouts: list[dict[str, tuple[float, float, float]]],
    object_names: list[str],
) -> None:
    """Set each env's object poses from layouts[env_id][obj] with random yaw per object."""
    if env_ids is None or len(env_ids) == 0 or not layouts or not object_names:
        return
    num_envs = env.scene.num_envs
    identity_wxyz = (1.0, 0.0, 0.0, 0.0)
    dummy = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=identity_wxyz)
    for obj_name in object_names:
        pose_list: list[Pose] = [dummy] * num_envs
        for cur_env in env_ids.tolist():
            layout_idx = min(int(cur_env), len(layouts) - 1)
            xyz = layouts[layout_idx][obj_name]
            yaw_rad = random.uniform(0.0, 2.0 * math.pi)
            rot_wxyz = _yaw_z_to_quat_wxyz(yaw_rad)
            pose_list[cur_env] = Pose(position_xyz=xyz, rotation_wxyz=rot_wxyz)
        set_object_pose_per_env(env, env_ids, SceneEntityCfg(obj_name), pose_list)
    env.scene.write_data_to_sim()
