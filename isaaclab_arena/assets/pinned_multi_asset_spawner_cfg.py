# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-asset spawn configuration that also stores which spawn entry each environment spawns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg


@configclass
class PinnedMultiAssetSpawnerCfg(MultiAssetSpawnerCfg):
    """MultiAssetSpawnerCfg that also says which spawn entry each environment spawns."""

    asset_indices_by_env: tuple[int, ...] | None = None
    """Index into assets_cfg for each environment, in environment order, or None until assigned."""


def get_asset_indices_by_env(spawn_cfg: SpawnerCfg, num_envs: int, name: str) -> tuple[int, ...]:
    """Return the spawn entry index of each environment stored on a spawn configuration with several entries.

    Args:
        spawn_cfg: Spawn configuration with several spawn entries.
        num_envs: Number of environments the indices must cover.
        name: Object name or scene key used in assertion messages.

    Returns:
        One index into spawn_cfg.assets_cfg per environment, in environment order.
    """
    assert isinstance(spawn_cfg, PinnedMultiAssetSpawnerCfg), (
        f"'{name}' spawns several entries from a {type(spawn_cfg).__name__}; "
        "use PinnedMultiAssetSpawnerCfg so each environment's entry can be stored."
    )
    assert spawn_cfg.asset_indices_by_env is not None, (
        f"'{name}' has no spawn entry indices; set asset_indices_by_env to one spawn entry index per environment"
        " before the scene is built (ArenaEnvBuilder.compose_manager_cfg() does this)."
    )
    asset_indices_by_env = tuple(spawn_cfg.asset_indices_by_env)
    assert (
        len(asset_indices_by_env) == num_envs
    ), f"'{name}' has spawn entry indices for {len(asset_indices_by_env)} environments, not {num_envs}."
    entry_count = len(spawn_cfg.assets_cfg)
    indices_in_range = all(type(index) is int and 0 <= index < entry_count for index in asset_indices_by_env)
    assert (
        indices_in_range
    ), f"'{name}' has invalid spawn entry indices {asset_indices_by_env}; expected integers in [0, {entry_count})."
    return asset_indices_by_env
