# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_poses import IDENTITY_ROTATION_XYZW, get_scene_root_poses_from_layout
from isaaclab_arena.relations.placement_sampler import PlacementSample, PlacementSampler
from isaaclab_arena.relations.relations import RotateAroundSolution

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer

# Name of the reset event term that owns the pooled object placer.
PLACEMENT_RESET_EVENT_NAME = "placement_reset"


class PlacementPoolHandle:
    """Opaque holder for a placement sampler to bypass event-param traversal.

    PlacementSampler may own a PooledObjectPlacer used to set the initial spawn pose. Isaac Lab deep-copies and
    validates EventTermCfg params, leading to two crashes: deepcopy fails for the Warp GPU cache wp.Mesh BVHs
    ("ctypes objects containing pointers cannot be pickled"); validation hits RecursionError when recursively
    walking all dicts and reaches placement assets (including embodiments with cyclic scene configs).

    The sampler may own a live pool or route draws through recorded replay.
    """

    __slots__ = ("sampler",)
    """Keep runtime state out of an instance dictionary so config validation does not traverse it."""

    def __init__(self, sampler: PlacementSampler) -> None:
        self.sampler = sampler

    @property
    def pool(self) -> PooledObjectPlacer | None:
        """Return the live placement pool, if this is not recorded replay."""
        return self.sampler.placement_pool

    @property
    def last_results(self) -> dict[int, PlacementResult]:
        """Return layouts applied by the most recent live reset."""
        return self.sampler.last_results

    def __deepcopy__(self, memo: dict[int, object]) -> PlacementPoolHandle:
        """Share the live pool across ``copy.deepcopy`` to avoid deep-copying the Warp cache BVHs."""
        memo[id(self)] = self
        return self


def get_placement_pool(env) -> PooledObjectPlacer | None:
    """Return the pooled placer stored on the env reset event, or ``None`` when absent.

    Lets a runtime caller reach the pool (e.g. to run the post-reset settle check) from the env alone,
    without holding the builder. The pool is reached through the env's event manager.

    Args:
        env: The gym-wrapped Isaac Lab env; the base env is reached via ``env.unwrapped``.
    """
    try:
        term_cfg = env.unwrapped.event_manager.get_term_cfg(PLACEMENT_RESET_EVENT_NAME)
    except ValueError:
        return None
    handle = term_cfg.params.get("placement_pool")
    assert handle is not None, f"'{PLACEMENT_RESET_EVENT_NAME}' event is missing its placement_pool parameter."
    assert handle.pool is not None, "Relation placement is using recorded replay, not a live placement pool."
    return handle.pool


def get_reset_placement_results(env: ManagerBasedEnv) -> dict[int, PlacementResult]:
    """Return the layouts applied by the most recent pooled placement reset."""
    term = env.unwrapped.event_manager.get_term_cfg(PLACEMENT_RESET_EVENT_NAME)
    return dict(term.params["placement_pool"].last_results)


def get_movable_asset_names(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
) -> list[str]:
    """Return scene names for non-anchor placement assets."""
    return [asset.get_scene_key() for asset in assets if asset not in anchor_assets]


def get_base_rotation_per_asset(
    assets: list[PlaceableAsset],
) -> dict[PlaceableAsset, tuple[float, float, float, float]]:
    """Return each asset's RotateAroundSolution rotation, or identity."""
    rotations = {}
    for asset in assets:
        rotate_marker = next((r for r in asset.get_relations() if isinstance(r, RotateAroundSolution)), None)
        rotations[asset] = rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW
    return rotations


def write_layout_to_sim(
    env: ManagerBasedEnv,
    env_id: int,
    result: PlacementResult,
    anchor_assets: set[PlaceableAsset],
    base_rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
) -> None:
    """Write one solved layout for offline pool validation."""
    missing_assets = [
        asset.name for asset in base_rotations if asset not in anchor_assets and asset not in result.positions
    ]
    assert not missing_assets, f"Placement layout is missing non-anchor assets: {missing_assets}"
    poses_by_asset = get_scene_root_poses_from_layout(list(base_rotations), result, anchor_assets)
    sample = PlacementSample(
        layout_id="offline_validation",
        poses={scene_key: pose for root_poses in poses_by_asset.values() for scene_key, pose in root_poses.items()},
    )
    env_ids = torch.tensor([env_id], device=env.device)
    write_placement_samples_to_sim(env, env_ids, [sample], list(poses_by_asset))


def solve_and_place_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    placement_pool: PlacementPoolHandle,
) -> None:
    """Coordinated reset event that draws layouts from the pool and writes poses.

    Registered as a single EventTermCfg(mode="reset"). Layouts are env-indexed:
    one layout is consumed for each requested absolute env id, so partial resets
    only advance the pools of the resetting envs.

    Args:
        env: The Isaac Lab environment.
        env_ids: 1-D tensor of environment indices being reset.
        placement_pool: Opaque handle to the runtime pool of solved placement layouts.
            Layout assets come from ``placement_pool.pool.objects``.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    sampler = placement_pool.sampler
    pool = sampler.placement_pool
    if pool is not None:
        num_scene_envs = env.scene.env_origins.shape[0]
        assert (
            pool.num_envs == num_scene_envs
        ), f"Placement pool has {pool.num_envs} envs, but scene has {num_scene_envs} env origins."
    else:
        assert sampler.replays_recorded_samples, "Relation placement has neither recorded samples nor a live pool"
    samples = sampler.sample(len(env_ids), env_ids)
    write_placement_samples_to_sim(env, env_ids, samples, sampler.write_assets)


def write_placement_samples_to_sim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    samples: list[PlacementSample],
    assets: list[PlaceableAsset],
) -> None:
    """Write prevalidated placement samples for the selected environments."""
    assert len(samples) == len(env_ids), "Placement sample count must match env_ids"
    for asset in assets:
        root_keys = asset.get_scene_root_keys()
        owned_poses = {
            scene_key: torch.stack([sample.poses[scene_key].to_tensor(device=env.device) for sample in samples])
            for scene_key in root_keys
        }
        asset.write_scene_root_poses_to_sim(env, env_ids, owned_poses)
