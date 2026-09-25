# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply asset-relative physics configuration during USD spawning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.string import string_to_callable
from pxr import Usd

from isaaclab_arena.assets.physics_config import UsdFileCfgPrimPhysicsWrapper, UsdPrimSpawnPhysicsCfg
from isaaclab_arena.utils.usd.prim_paths import get_prim_relative_to_root


def make_usd_spawn_cfg_with_addons(cfg: UsdFileCfg, addons: dict[str, Any]) -> UsdFileCfg:
    """Return a spawn config with ordinary and per-prim addons applied together.

    Args:
        cfg: Existing USD spawn configuration.
        addons: USD spawn fields and an optional string-to-UsdPrimSpawnPhysicsCfg prim_physics mapping.

    Returns:
        An independent USD config retaining unspecified spawn options. With prim_physics
        addons, its spawner also applies per-prim physics before cloning.
    """
    assert isinstance(cfg, UsdFileCfg), "Spawn addons require a USD spawn config."
    assert isinstance(addons, dict), "Spawn addons must be a dictionary of USD spawn fields."
    # Ordinary addons follow the UsdFileCfg constructor's field-replacement semantics.
    options = dict(addons)
    if "prim_physics" not in options:
        return cfg.replace(**options)

    overrides = options.pop("prim_physics")
    assert isinstance(
        overrides, dict
    ), "prim_physics must be a dictionary of paths to UsdPrimSpawnPhysicsCfg instances."
    cfg = cfg.replace(**options)
    # Retain other prim entries, but replace each explicitly supplied entry as a typed config.
    overrides = {**getattr(cfg, "prim_physics", {}), **overrides}
    return make_usd_spawn_cfg_with_prim_physics(cfg, overrides)


def _validate_prim_physics_types(overrides: dict[str, UsdPrimSpawnPhysicsCfg]) -> None:
    """Reject malformed override mappings before configuration or USD authoring."""
    assert isinstance(
        overrides, dict
    ), "prim_physics must be a dictionary of paths to UsdPrimSpawnPhysicsCfg instances."
    for path, cfg in overrides.items():
        assert isinstance(path, str) and path, "Physics target must be a nonempty relative prim path."
        assert isinstance(cfg, UsdPrimSpawnPhysicsCfg), f"Physics override for {path} must be UsdPrimSpawnPhysicsCfg."


def make_usd_spawn_cfg_with_prim_physics(
    cfg: UsdFileCfg, overrides: dict[str, UsdPrimSpawnPhysicsCfg]
) -> UsdFileCfgPrimPhysicsWrapper:
    """Return an independent USD spawn config that applies the given per-prim overrides.

    Args:
        cfg: USD config with a standard or custom @clone-decorated USD spawner.
            Its body must load one prim and return it without cloning.
        overrides: Exact asset-relative paths and physics settings; replaces any previous mapping.

    Returns:
        A UsdFileCfgPrimPhysicsWrapper using spawn_usd_with_physics; assign it to the asset's spawn field.
    """
    # 1. Validate the supported USD config type and per-prim override mapping.
    assert type(cfg) in (
        UsdFileCfg,
        UsdFileCfgPrimPhysicsWrapper,
    ), "Per-prim physics requires a standard USD spawn config."
    _validate_prim_physics_types(overrides)

    # 2. Keep the original USD spawner, including backend-specific setup.
    spawn_func = string_to_callable(str(cfg.func)) if isinstance(cfg.func, str) else cfg.func
    if spawn_func is spawn_usd_with_physics:
        spawn_func = cfg.usd_spawn_func
    spawn_func = _resolve_usd_spawn_func(spawn_func)

    # 3. Collect existing constructor fields, preserving nested typed configs.
    values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}

    # 4. Add the per-prim overrides and select the physics-aware spawn function.
    values.update(prim_physics=overrides, func=spawn_usd_with_physics, usd_spawn_func=spawn_func)

    # 5. Return an independent wrapper; construction deep-copies mutable values.
    # Declared fields ensure the overrides survive later configclass.copy()/replace() calls.
    return UsdFileCfgPrimPhysicsWrapper(**values)


def _resolve_usd_spawn_func(spawn_func: Callable | str) -> Callable:
    """Resolve a saved USD spawner and require its @clone-wrapped single-prim body."""
    if isinstance(spawn_func, str):
        spawn_func = string_to_callable(str(spawn_func))
    assert callable(getattr(spawn_func, "__wrapped__", None)), "USD spawn functions must use @clone."
    return spawn_func


def _resolve_and_validate_overrides(
    root: Usd.Prim, overrides: dict[str, UsdPrimSpawnPhysicsCfg]
) -> list[tuple[Usd.Prim, UsdPrimSpawnPhysicsCfg]]:
    """Resolve and validate all targets without applying physics edits.

    Args:
        root: Spawned asset root used to resolve relative target paths.
        overrides: Asset-relative prim paths mapped to physics configurations.

    Returns:
        Validated (prim, config) pairs in mapping order, ready for application.
    """
    # 1. Recheck types: this public API also accepts configs edited after construction.
    _validate_prim_physics_types(overrides)
    targets = []
    for path, cfg in overrides.items():
        # 2. Resolve the relative path, rejecting escaping paths and missing targets.
        prim = get_prim_relative_to_root(root, path)
        # 3. Physics edits require an editable prim, not an instance proxy.
        assert (
            not prim.IsInstanceProxy()
        ), f"Physics target {prim.GetPath()} is an instance proxy; set make_uninstanceable=True to edit it."
        targets.append((prim, cfg))

    # 4. Validate every config's target and settings without mutating the stage.
    for prim, cfg in targets:
        cfg.validate_target(prim, root)
    return targets


def apply_prim_physics(root: Usd.Prim, overrides: dict[str, UsdPrimSpawnPhysicsCfg]) -> None:
    """Apply physics settings to selected prims on the spawned asset.

    Args:
        root: Spawned asset root on the stage to edit.
        overrides: Exact asset-relative paths and their physics configuration.
    """
    # Check every target before applying any edits. If validation fails, no physics settings change.
    for prim, cfg in _resolve_and_validate_overrides(root, overrides):
        cfg.apply(prim, root)


# NOTE: Keep @clone on this outer wrapper to preserve this order:
# 1. Load the USD in the first matching environment.
# 2. Apply the per-prim physics edits to that asset.
# 3. Copy the configured asset into the remaining matching environments.
# Every clone inherits the same physics edits.
@clone
def spawn_usd_with_physics(
    prim_path: str,
    cfg: UsdFileCfgPrimPhysicsWrapper,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn USD, apply selected physics properties, then clone the configured asset.

    Args:
        prim_path: Asset path, optionally with an environment expression in its parent path.
        cfg: USD config returned by make_usd_spawn_cfg_with_prim_physics.
        translation: Root translation, following the ordinary USD spawner convention.
        orientation: Root quaternion in XYZW order.
        **kwargs: Additional arguments forwarded to the ordinary USD spawner.

    Returns:
        The first spawned asset root.
    """
    # 1. Run the original spawner at one path, without its @clone post-processing.
    # Our outer decorator applies visibility, labels, and contact sensors once, then clones.
    spawn_func = _resolve_usd_spawn_func(cfg.usd_spawn_func)
    prim = spawn_func.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    # 2. Apply physics edits to this asset before any copies are made.
    apply_prim_physics(prim, cfg.prim_physics)
    # 3. Return the configured asset so @clone can copy it to the remaining environments.
    return prim
