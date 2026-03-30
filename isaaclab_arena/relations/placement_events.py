# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Placement event: apply placement layouts per env on reset (num_envs>1)."""

from __future__ import annotations

import dataclasses
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.terms.events import set_object_pose_per_env
from isaaclab_arena.utils.pose import Pose


def _env_ids_to_list(env: ManagerBasedEnv, env_ids: torch.Tensor | slice) -> list[int]:
    """Convert env_ids (``torch.Tensor`` or ``slice(None)``) to a plain list of int indices.

    Args:
        env: The environment instance (used to determine num_envs for slice).
        env_ids: Environment indices from the Isaac Lab event manager.
    """
    if isinstance(env_ids, slice):
        return list(range(env.num_envs))
    return env_ids.tolist()


@configclass
class PlacementEventsCfg:
    """Event config for applying placement layouts per env on reset."""

    set_object_pose_per_env_from_layouts: EventTermCfg = dataclasses.MISSING  # type: ignore[assignment]


def make_placement_event_cfg(
    positions_all_envs_by_name: list[dict[str, tuple[float, float, float]]],
    object_names: list[str],
    anchor_names: list[str] | None = None,
) -> PlacementEventsCfg:
    """Build event config for applying placement layouts per env on reset."""
    params: dict = {
        "positions_all_envs_by_name": positions_all_envs_by_name,
        "object_names": object_names,
        "anchor_names": anchor_names or [],
    }
    return PlacementEventsCfg(
        set_object_pose_per_env_from_layouts=EventTermCfg(
            func=set_object_pose_per_env_from_layouts,
            mode="reset",
            params=params,
        )
    )


def set_object_pose_per_env_from_layouts(
    env: ManagerBasedEnv,
    env_ids,
    positions_all_envs_by_name: list[dict[str, tuple[float, float, float]]],
    object_names: list[str],
    anchor_names: list[str] | None = None,
) -> None:
    """Set each object's root pose per env from layout dicts; anchors first."""
    assert (
        len(positions_all_envs_by_name) == env.num_envs
    ), f"Expected {env.num_envs} layout dicts, got {len(positions_all_envs_by_name)}"
    env_id_list = _env_ids_to_list(env, env_ids)
    if not env_id_list:
        return
    env_ids_t = torch.tensor(env_id_list, device=env.device)
    anchor_set = set(anchor_names or [])
    ordered_names = [n for n in object_names if n in anchor_set]
    ordered_names += [n for n in object_names if n not in anchor_set]
    identity_quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    for name in ordered_names:
        if name not in env.scene.keys():
            continue
        asset = env.scene[name]
        if not hasattr(asset, "write_root_pose_to_sim"):
            continue
        pose_list = []
        for e in range(len(positions_all_envs_by_name)):
            xyz = positions_all_envs_by_name[e].get(name)
            if xyz is not None:
                x, y, z = xyz
                pose_list.append(Pose(position_xyz=(x, y, z), rotation_wxyz=identity_quat_wxyz))
            else:
                pose_list.append(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=identity_quat_wxyz))
        set_object_pose_per_env(env, env_ids_t, SceneEntityCfg(name), pose_list)
