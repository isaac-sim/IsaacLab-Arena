# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assign complete upstream gear layouts to independent Arena slots."""

from __future__ import annotations

import copy
import math

from .variation import _SURFACE_ORIGIN_ARENA, _VARIANTS


def coupled_layout_clones(combinations, num_clones, device):
    """Keep board and gear variant indices coupled in the clone planner."""
    varying = combinations.amax(dim=0) > 0
    columns = combinations[:, varying]
    if not columns.numel():
        return combinations[:1].expand(num_clones, -1).to(device)
    rows = combinations[(columns == columns[:, :1]).all(dim=1)]
    if len(rows) != num_clones:
        raise ValueError("gear clone variants must match the environment count")
    return rows.to(device)


def apply_layout_poses(layout, row, gear_names):
    """Replace source XY/yaw while preserving Arena's solved support heights."""
    by_name = {asset.name: asset for asset in layout.positions}
    for index, name in enumerate(("board", *gear_names)):
        asset = by_name[name]
        source = {"xy": row["board_xy"], "yaw": row["board_yaw"]} if index == 0 else row["gears"][index - 1]
        layout.positions[asset] = (
            _SURFACE_ORIGIN_ARENA[0] + source["xy"][0],
            _SURFACE_ORIGIN_ARENA[1] + source["xy"][1],
            layout.positions[asset][2],
        )
        layout.orientations[asset] = source["yaw"]
        layout.rotations[asset] = (
            0.0,
            0.0,
            math.sin(source["yaw"] / 2),
            math.cos(source["yaw"] / 2),
        )


def record_gear_layout(env, env_id, layout_names):
    """Stamp the upstream source row into Arena's authoritative episode record."""
    return {"gear_layout": layout_names[env_id]}


def configure_parallel_layouts(env_cfg, *, family, gear_names, layout_names=()):
    """Bind geometry, placement and scoring to the same ordered source rows."""
    from isaaclab.sim import MultiAssetSpawnerCfg

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME

    from ..asset_factories import GEAR_MESH_ASSET_PATHS

    count = env_cfg.scene.num_envs
    if count == 1 and not layout_names:
        return env_cfg
    if not layout_names:
        raise ValueError("parallel gear layouts require explicit layout_names, one per environment")
    if len(layout_names) != count or len(set(layout_names)) != count:
        raise ValueError("gear layout_names must contain one distinct name per environment")
    available = {row["name"]: row for row in _VARIANTS if row["family"] == family}
    if any(name not in available for name in layout_names):
        raise ValueError("gear layout_names must belong to the selected upstream family")
    rows = [available[name] for name in layout_names]
    for index, name in enumerate(("board", *gear_names)):
        asset = getattr(env_cfg.scene, name)
        variants = []
        for row in rows:
            spawn = copy.deepcopy(asset.spawn)
            if index == 0:
                spawn.usd_path = str(GEAR_MESH_ASSET_PATHS["board"])
                spawn.variants = {"layout": row["board_variant"]}
            else:
                spawn.usd_path = str(GEAR_MESH_ASSET_PATHS[f"{row['gears'][index - 1]['teeth']}t"])
            variants.append(spawn)
        asset.spawn = MultiAssetSpawnerCfg(assets_cfg=variants, random_choice=False)
    env_cfg.scene.clone_cfg.clone_strategy = coupled_layout_clones
    event = getattr(env_cfg.events, PLACEMENT_RESET_EVENT_NAME)
    pool = event.params["placement_pool"].pool
    for layouts, row in zip(pool.layouts_per_env(), rows, strict=True):
        for layout in layouts:
            apply_layout_poses(layout, row, gear_names)
    # Source rows are fixed starts. Never refill with the scalar template's poses.
    pool.recycle_layouts = True
    env_cfg.episode_recorders.gear_layout = EpisodeRecorderTermCfg(
        func=record_gear_layout, params={"layout_names": list(layout_names)}
    )
    env_cfg.terminations.success.params.update(
        target_offsets_xyz=[row["target_offsets_xyz"] for row in rows],
        gear_teeth=[row["stations"] for row in rows],
    )
    return env_cfg
