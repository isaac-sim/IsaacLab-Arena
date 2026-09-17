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

    assert type(cfg) in (
        UsdFileCfg,
        UsdFileCfgPrimPhysicsWrapper,
    ), "Per-prim physics requires a standard USD spawn config."
    _validate_prim_physics_types(overrides)
    spawn_func = string_to_callable(str(cfg.func)) if isinstance(cfg.func, str) else cfg.func
    assert spawn_func in (
        spawn_from_usd,
        spawn_usd_with_physics,
    ), "Custom spawn functions must call apply_prim_physics before cloning."
    # Preserve typed fields rather than converting nested configs to dictionaries.
    values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}
    values.update(prim_physics=overrides, func=spawn_usd_with_physics)
    # Declared fields survive configclass.copy()/replace(); construction deep-copies mutable values.
    return UsdFileCfgPrimPhysicsWrapper(**values)


def _relative_target(root: Usd.Prim, relative_path: str) -> Usd.Prim:
    """Resolve an exact path inside one asset and require an editable prim."""
    assert isinstance(relative_path, str) and relative_path, "Physics target must be a nonempty relative prim path."
    path = Sdf.Path(relative_path)
    assert (
        not path.IsAbsolutePath()
        and (path.IsPrimPath() or path == Sdf.Path.reflexiveRelativePath)
        and ".." not in relative_path.split("/")
        and "{" not in relative_path
    ), f"Physics target must be an asset-relative prim path: {relative_path!r}"
    prim = root.GetStage().GetPrimAtPath(path.MakeAbsolutePath(root.GetPath()))
    assert prim.IsValid(), f"Physics target does not exist: {root.GetPath()}/{relative_path}"
    assert (
        not prim.IsInstanceProxy()
    ), f"Physics target {prim.GetPath()} is an instance proxy; set make_uninstanceable=True to edit it."
    return prim


def _resolve_overrides(
    root: Usd.Prim, overrides: dict[str, UsdPrimSpawnPhysicsCfg]
) -> list[tuple[Usd.Prim, UsdPrimSpawnPhysicsCfg]]:
    """Resolve and validate all targets before writing any per-prim properties."""
    _validate_prim_physics_types(overrides)
    targets = []
    for path, cfg in overrides.items():
        targets.append((_relative_target(root, path), cfg))

    # Subclasses own schema and value checks; validation must not mutate the stage.
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
    for prim, cfg in _resolve_overrides(root, overrides):
        cfg.apply(prim, root)


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
