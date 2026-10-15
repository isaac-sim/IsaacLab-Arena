# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spawn entry indices of multi-asset objects and the clone strategy that makes Isaac Lab spawn them."""

from __future__ import annotations

import torch
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg, get_asset_indices_by_env

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def assign_asset_indices(objects: Iterable[PlaceableAsset], num_envs: int, seed: int | None) -> None:
    """Write one spawn entry index per environment into every multi-asset object's spawn configuration that has none.

    Indices already in a spawn configuration are kept. A RigidObjectSet with random_choice draws its indices;
    every other object repeats its entry order across environments.

    Args:
        objects: Scene objects. Objects with one spawn entry are skipped.
        num_envs: Number of environments.
        seed: Seed for random draws, or None to draw from torch's global generator. The k-th object whose
            indices are written here, counting from 0, draws with seed + k.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet

    assert num_envs > 0, "Spawn entry assignment requires at least one environment."
    seed_offset = 0
    for obj in objects:
        if not obj.has_multiple_assets:
            continue
        assert obj.object_type == ObjectType.RIGID, (
            f"Object '{obj.name}' has several spawn entries, but only rigid objects spawn a different asset per"
            f" environment; got object type '{obj.object_type.value}'."
        )
        # Asserts that all spawn entries share one rigid-body path, which Isaac Lab reads once for all environments.
        obj.get_contact_sensor_prim_path()
        spawn_cfg = obj.object_cfg.spawn
        if isinstance(spawn_cfg, PinnedMultiAssetSpawnerCfg) and spawn_cfg.asset_indices_by_env is None:
            entry_count = len(spawn_cfg.assets_cfg)
            if isinstance(obj, RigidObjectSet) and obj.random_choice:
                if seed is None:
                    drawn_indices = torch.randint(0, entry_count, (num_envs,))
                else:
                    generator = torch.Generator().manual_seed(seed + seed_offset)
                    drawn_indices = torch.randint(0, entry_count, (num_envs,), generator=generator)
                spawn_cfg.asset_indices_by_env = tuple(drawn_indices.tolist())
            else:
                spawn_cfg.asset_indices_by_env = tuple(env_id % entry_count for env_id in range(num_envs))
            seed_offset += 1
        # Rejects Isaac Lab's own MultiAssetSpawnerCfg, and existing indices for another number of environments
        # or out of range.
        get_asset_indices_by_env(spawn_cfg, num_envs, obj.name)


def validate_scene_asset_indices(scene_cfg: InteractiveSceneCfg) -> list[tuple[str, PinnedMultiAssetSpawnerCfg]]:
    """Check that every multi-entry spawner of a scene configuration is pinned and holds valid spawn entry indices.

    Args:
        scene_cfg: Scene configuration Isaac Lab builds, including later edits and overrides.

    Returns:
        Scene key and spawn configuration of every multi-entry spawner, in scene declaration order, which is the
        order of Isaac Lab's clone-plan columns. Empty when the scene has no multi-entry spawner.
    """
    # The same configurations InteractiveScene._collect_asset_cfgs() clones, with collection members in place.
    multi_entry_cfgs = []
    for scene_key, asset_cfg in vars(scene_cfg).items():
        if scene_key in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
            continue
        is_collection = isinstance(asset_cfg, RigidObjectCollectionCfg)
        cloned_cfgs = list(asset_cfg.rigid_objects.values()) if is_collection else [asset_cfg]
        for cloned_cfg in cloned_cfgs:
            if not isinstance(cloned_cfg, (AssetBaseCfg, CameraCfg, RayCasterCfg)) or cloned_cfg.spawn is None:
                continue
            if cloner.num_spawn_variants(cloned_cfg.spawn) > 1:
                multi_entry_cfgs.append((scene_key, cloned_cfg, is_collection))
    if not multi_entry_cfgs:
        return []

    clone_cfg = scene_cfg.clone_cfg
    assert not clone_cfg.clone_combinations, (
        "Pinned spawn entries cannot be combined with clone_combinations: the clone strategy finds each spawner's"
        " column in the full product of spawn entries."
    )
    pinned_spawners = []
    for scene_key, cloned_cfg, is_collection_member in multi_entry_cfgs:
        assert not is_collection_member, (
            f"A member of rigid object collection '{scene_key}' has several spawn entries, which pinned spawn"
            " entries do not support."
        )
        assert isinstance(
            cloned_cfg, AssetBaseCfg
        ), f"'{scene_key}' has several spawn entries but is a {type(cloned_cfg).__name__}, not an asset."
        prim_path = cloner.expand_env_regex_ns(cloned_cfg.prim_path, clone_cfg.clone_template)
        assert cloner.path.match(prim_path, clone_cfg.clone_template) is not None, (
            f"'{scene_key}' spawns one entry per environment, so its prim path must be inside each environment;"
            f" got '{cloned_cfg.prim_path}'."
        )
        get_asset_indices_by_env(cloned_cfg.spawn, scene_cfg.num_envs, scene_key)
        pinned_spawners.append((scene_key, cloned_cfg.spawn))
    return pinned_spawners


@contextmanager
def asset_assignment_clone_strategy(scene_cfg: InteractiveSceneCfg) -> Iterator[None]:
    """Make Isaac Lab spawn the entry each pinned spawner names for each environment while a scene is built.

    The scene's clone strategy is replaced inside the context and restored on exit. A scene without a multi-entry
    spawner is not changed.

    Args:
        scene_cfg: Scene configuration Isaac Lab builds.
    """
    pinned_spawners = validate_scene_asset_indices(scene_cfg)
    if not pinned_spawners:
        yield
        return
    # Read once here, so the strategy pins the values validation checked.
    pinned_columns = []
    for scene_key, spawn_cfg in pinned_spawners:
        pinned_columns.append((scene_key, tuple(spawn_cfg.asset_indices_by_env), len(spawn_cfg.assets_cfg)))

    # A plain function, not a callable object or functools.partial: to_dict() names a callable by its
    # __name__, so those would make the scene configuration fail to serialize while it is installed.
    def select_assigned_spawn_entries(combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor:
        """Return each pinned spawner's entry per environment, and entry 0 in every other column."""
        # combinations is the full product of entry indices (validation rejects clone_combinations), with one
        # column per spawned scene asset in declaration order and sensors last. Every multi-entry spawner is a
        # pinned asset outside any collection, so the multi-entry columns are the pinned spawners in order.
        entry_counts = combinations.max(dim=0).values + 1
        multi_entry_columns = (entry_counts > 1).nonzero().flatten().tolist()
        assert len(multi_entry_columns) == len(pinned_columns), (
            f"Isaac Lab plans {len(multi_entry_columns)} multi-entry spawners, but the scene configuration pins"
            f" {len(pinned_columns)}."
        )
        chosen_entries = torch.zeros((num_clones, combinations.shape[1]), dtype=torch.long, device=device)
        for column, (scene_key, asset_indices_by_env, entry_count) in zip(multi_entry_columns, pinned_columns):
            assert int(entry_counts[column]) == entry_count, (
                f"Isaac Lab plans {int(entry_counts[column])} spawn entries for '{scene_key}', but its spawn"
                f" configuration has {entry_count}."
            )
            assert len(asset_indices_by_env) == num_clones, (
                f"'{scene_key}' has spawn entry indices for {len(asset_indices_by_env)} environments, but Isaac Lab"
                f" clones {num_clones}."
            )
            chosen_entries[:, column] = torch.tensor(asset_indices_by_env, dtype=torch.long, device=device)
        return chosen_entries

    # Isaac Lab reads the strategy once, when the scene plans its clones, so it is installed only for the
    # build. Left on the configuration, the closure would serialize to a "module:name" string that does not
    # resolve, and a later build with another num_envs would fail its row-count check.
    original_clone_strategy = scene_cfg.clone_cfg.clone_strategy
    scene_cfg.clone_cfg.clone_strategy = select_assigned_spawn_entries
    try:
        yield
    finally:
        # Restore the exact object found: after a Hydra round trip it is a string that resolves to the strategy.
        scene_cfg.clone_cfg.clone_strategy = original_clone_strategy
