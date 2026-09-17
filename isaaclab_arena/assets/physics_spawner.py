# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply asset-relative physics configuration during USD spawning."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils.string import string_to_callable
from pxr import Sdf, Usd

if TYPE_CHECKING:
    from .physics_config import UsdFileCfgPrimPhysicsWrapper, UsdPrimSpawnPhysicsCfg


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
    _validate_prim_physics_types(overrides)
    assert "func" not in options, "Custom spawn functions must call apply_prim_physics before cloning."
    cfg = cfg.replace(**options)
    # Retain other prim entries, but replace each explicitly supplied entry as a typed config.
    overrides = {**getattr(cfg, "prim_physics", {}), **overrides}
    return make_usd_spawn_cfg_with_prim_physics(cfg, overrides)


def _validate_prim_physics_types(overrides: dict[str, UsdPrimSpawnPhysicsCfg]) -> None:
    """Reject malformed override mappings before configuration or USD authoring."""
    from .physics_config import UsdPrimSpawnPhysicsCfg

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
        cfg: Standard USD config whose ordinary spawn options are retained.
        overrides: Exact asset-relative paths and physics settings; replaces any previous mapping.

    Returns:
        A UsdFileCfgPrimPhysicsWrapper using spawn_usd_with_physics; assign it to the asset's spawn field.
    """
    from .physics_config import UsdFileCfgPrimPhysicsWrapper

    # 1. Validate the supported USD config type and per-prim override mapping.
    assert type(cfg) in (
        UsdFileCfg,
        UsdFileCfgPrimPhysicsWrapper,
    ), "Per-prim physics requires a standard USD spawn config."
    _validate_prim_physics_types(overrides)

    # 2. Resolve the spawn callable and reject custom spawners that would be replaced.
    spawn_func = string_to_callable(str(cfg.func)) if isinstance(cfg.func, str) else cfg.func
    assert spawn_func in (
        spawn_from_usd,
        spawn_usd_with_physics,
    ), "Custom spawn functions must call apply_prim_physics before cloning."

    # 3. Collect existing constructor fields, preserving nested typed configs.
    values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}

    # 4. Add the per-prim overrides and select the physics-aware spawn function.
    values.update(prim_physics=overrides, func=spawn_usd_with_physics)

    # 5. Return an independent wrapper; construction deep-copies mutable values.
    # Declared fields ensure the overrides survive later configclass.copy()/replace() calls.
    return UsdFileCfgPrimPhysicsWrapper(**values)


def _parse_relative_prim_path(relative_path: str) -> Sdf.Path:
    """Parse an exact prim path that cannot escape its asset root."""
    assert isinstance(relative_path, str) and relative_path, "Physics target must be a nonempty relative prim path."
    path = Sdf.Path(relative_path)
    assert (
        not path.IsAbsolutePath()
        and (path.IsPrimPath() or path == Sdf.Path.reflexiveRelativePath)
        and ".." not in relative_path.split("/")
        and "{" not in relative_path
    ), f"Physics target must be an asset-relative prim path: {relative_path!r}"
    return path


def _get_prim_relative_to_root(root: Usd.Prim, relative_path: str) -> Usd.Prim:
    """Return an editable prim selected by an exact path relative to the asset root."""
    # Validate the path before resolving it on the spawned asset's stage.
    path = _parse_relative_prim_path(relative_path)
    prim = root.GetStage().GetPrimAtPath(path.MakeAbsolutePath(root.GetPath()))
    assert prim.IsValid(), f"Physics target does not exist: {root.GetPath()}/{relative_path}"
    assert (
        not prim.IsInstanceProxy()
    ), f"Physics target {prim.GetPath()} is an instance proxy; set make_uninstanceable=True to edit it."
    return prim


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
    # 1. Check the override mapping's key and value types.
    _validate_prim_physics_types(overrides)
    targets = []
    for path, cfg in overrides.items():
        # 2. Resolve the relative prim path under the spawned asset root.
        # 3. _get_prim_relative_to_root also rejects missing targets, escaping paths, and instance proxies.
        targets.append((_get_prim_relative_to_root(root, path), cfg))

    # 4. Validate every config's target and settings without mutating the stage.
    for prim, cfg in targets:
        cfg.validate_target(prim, root)
    return targets


def apply_prim_physics(root: Usd.Prim, overrides: dict[str, UsdPrimSpawnPhysicsCfg]) -> None:
    """Author physics on selected prims without changing the source asset or backend builder.

    Args:
        root: Spawned asset root, whose stage receives the authored opinions.
        overrides: Exact asset-relative paths and their physics configuration.
    """
    # Validate the full mapping first so a bad later target does not leave earlier overrides applied.
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
    cfg: UsdFileCfg,
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
    # The outer clone decorator resolves the pattern to one concrete prototype path first.
    # Loading that one path cannot clone siblings; they are copied only after our overrides.
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    apply_prim_physics(prim, cfg.prim_physics)
    return prim
