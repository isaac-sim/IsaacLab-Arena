# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation placement in MDP2: a reset event that draws relation-solved layouts from a pool on the GPU."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2 import State, Term, TermContext
from isaaclab_experimental.mdp2.rng import uniform

from isaaclab_arena.relations.runtime.placement_poses import get_scene_root_poses_from_layout

if TYPE_CHECKING:
    from isaaclab_arena.relations.runtime.placement_sampler import PlacementSampler

LAYOUT_EVENT_NAME = "relation_placement"
"""Name of the reset event that draws relation-solved layouts."""


def _draw_layout(mask: torch.Tensor, rng: torch.Tensor, pool: torch.Tensor, env_origins: torch.Tensor, **poses) -> None:
    """Write one pooled layout per reset world: each ``pose_<k>`` output gets columns ``7k:7k+7`` of the layout."""
    index = uniform(rng, 0.0, float(pool.shape[0])).long().clamp(max=pool.shape[0] - 1)
    layout = pool[index]
    reset = mask[:, None]
    for name, pose in poses.items():
        if not name.startswith("pose_"):
            continue
        k = int(name.removeprefix("pose_"))
        T_W_O = layout[:, 7 * k : 7 * k + 7].clone()
        T_W_O[:, :3] += env_origins
        pose.copy_(torch.where(reset, T_W_O, pose))
        velocity = poses[f"vel_{k}"]
        velocity.copy_(torch.where(reset, torch.zeros_like(velocity), velocity))


def make_layout_pool_term(layouts: torch.Tensor) -> Term:
    """Return a reset event term that draws relation-solved layouts from a pool, at random for each reset world.

    Args:
        layouts: Solved layouts, one row per layout and ``(x, y, z, qx, qy, qz, qw)`` environment-local root poses of
            the placed assets side by side. They fill the term's pool once, at startup.

    Returns:
        An event term with outputs ``pose_<k>`` and ``vel_<k>``, the root pose and velocity of placed asset ``k``, and
        the input ``env_origins``.
    """
    assert layouts.ndim == 2 and layouts.shape[1] % 7 == 0, "Layouts must hold 7 pose values per placed asset"
    host_layouts = layouts.detach().to("cpu", torch.float32)

    def fill_pool(ctx: TermContext, **_) -> None:
        ctx.state["pool"].copy_(host_layouts.to(ctx.state["pool"].device))

    return Term(
        torch=_draw_layout,
        state={"pool": State(shape=(host_layouts.shape[0], host_layouts.shape[1]), persistent=True)},
        random=True,
        init=fill_pool,
    )


def make_layout_task_cfg(placement_sampler: PlacementSampler, pool_size: int) -> mdp2.TaskCfg:
    """Return a partial task with a reset event that writes relation-solved layouts of the movable assets.

    Args:
        placement_sampler: The live placement sampler, whose pool holds solved layouts.
        pool_size: Number of layouts drawn from the sampler's pool to fill the event's pool.

    Returns:
        A partial task config with the event :data:`LAYOUT_EVENT_NAME`.
    """
    scene_keys = [key for asset in placement_sampler.write_assets for key in asset.get_scene_root_keys()]
    rows = []
    for result in placement_sampler.placement_pool.sample_with_replacement(pool_size):
        poses_by_asset = get_scene_root_poses_from_layout(placement_sampler.assets, result)
        poses = {key: pose for root_poses in poses_by_asset.values() for key, pose in root_poses.items()}
        rows.append([value for key in scene_keys for value in (*poses[key].position_xyz, *poses[key].rotation_xyzw)])
    outputs = {}
    for k, key in enumerate(scene_keys):
        entity = mdp2.Entity(key)
        outputs[f"pose_{k}"] = entity.root_link_pose_w
        outputs[f"vel_{k}"] = entity.root_com_vel_w
    event = mdp2.Event(
        make_layout_pool_term(torch.tensor(rows, dtype=torch.float32)),
        inputs={"env_origins": mdp2.Entity("scene").env_origins},
        outputs=outputs,
    )
    return mdp2.TaskCfg(events={LAYOUT_EVENT_NAME: event})
