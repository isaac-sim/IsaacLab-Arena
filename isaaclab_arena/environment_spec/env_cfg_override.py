# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply environment overrides using the shared typed-config merger."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from isaaclab_arena.utils.config_override import apply_config_override

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def apply_env_cfg_override(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    override: dict[str, Any],
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Validate overrides on a copy, then publish the completed environment configuration.

    Args:
        env_cfg: Arena manager-based RL environment configuration to update.
        override: Nested mapping from graph env_cfg_override.

    Returns:
        The updated env_cfg instance.
    """
    assert isinstance(override, dict), "env_cfg_override must be a mapping"
    values = deepcopy(override)
    _prepare_spawn_overrides(env_cfg, values)
    working_cfg = apply_config_override(env_cfg, values, path="env_cfg_override")
    # Keep the caller's root config identity, but publish nothing until every override succeeds.
    for field in fields(working_cfg):
        setattr(env_cfg, field.name, getattr(working_cfg, field.name))
    return env_cfg


def _prepare_spawn_overrides(env_cfg: Any, values: dict[str, Any]) -> None:
    """Prepare per-prim spawn addons through the same path as objects and embodiments."""
    from isaaclab_arena.assets.physics_spawner import with_spawn_cfg_addon

    scene_values = values.get("scene", {})
    if not isinstance(scene_values, dict):
        return
    scene_cfg = getattr(env_cfg, "scene", None)
    for name, asset_values in scene_values.items():
        if not isinstance(asset_values, dict):
            continue
        spawn_values = asset_values.get("spawn")
        if not isinstance(spawn_values, dict) or "prim_physics" not in spawn_values:
            continue
        asset_cfg = getattr(scene_cfg, name, None)
        assert asset_cfg is not None, f"Unknown scene asset {name!r} in env_cfg_override"
        assert getattr(asset_cfg, "spawn", None) is not None, f"Scene asset {name!r} has no spawn config"
        asset_values["spawn"] = with_spawn_cfg_addon(asset_cfg.spawn, spawn_values)
