# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bake Franka on-stand USD assets for an absolute ``stand_height_m``."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path
from pxr import Gf, Usd, UsdGeom

# Prim names under the stock ``franka_panda_hand_on_stand.usd`` default prim.
FRANKA_ON_STAND_ROOT_PRIM_NAME = "panda"
FRANKA_ON_STAND_STAND_PRIM_NAME = "stand_instanceable"
FRANKA_ON_STAND_ROBOT_BASE_PRIM_NAME = "panda_link0"

_DEFAULT_HEIGHT_EPS = 1e-4


def _default_cache_dir() -> Path:
    env = os.environ.get("ISAACLAB_ARENA_USD_CACHE")
    root = Path(env) if env else Path.home() / ".cache" / "isaaclab_arena" / "usd"
    return root / "franka_stand_height"


def _root_prim_path() -> str:
    return f"/{FRANKA_ON_STAND_ROOT_PRIM_NAME}"


def _stand_prim_path() -> str:
    return f"/{FRANKA_ON_STAND_ROOT_PRIM_NAME}/{FRANKA_ON_STAND_STAND_PRIM_NAME}"


def _robot_base_prim_path() -> str:
    return f"/{FRANKA_ON_STAND_ROOT_PRIM_NAME}/{FRANKA_ON_STAND_ROBOT_BASE_PRIM_NAME}"


def assert_franka_on_stand_prims(stage: Usd.Stage) -> tuple[Usd.Prim, Usd.Prim, Usd.Prim]:
    """Return ``(root, stand, robot_base)`` prims; assert if any expected name is missing.

    Args:
        stage: Opened Franka on-stand USD stage.

    Returns:
        The root, stand, and robot-base prims.
    """
    root = stage.GetPrimAtPath(_root_prim_path())
    stand = stage.GetPrimAtPath(_stand_prim_path())
    robot_base = stage.GetPrimAtPath(_robot_base_prim_path())
    assert (
        root.IsValid()
    ), f"Franka on-stand USD missing root prim '{FRANKA_ON_STAND_ROOT_PRIM_NAME}' (expected path {_root_prim_path()!r})"
    assert stand.IsValid(), (
        f"Franka on-stand USD missing stand prim '{FRANKA_ON_STAND_STAND_PRIM_NAME}' "
        f"(expected path {_stand_prim_path()!r})"
    )
    assert robot_base.IsValid(), (
        f"Franka on-stand USD missing robot base prim '{FRANKA_ON_STAND_ROBOT_BASE_PRIM_NAME}' "
        f"(expected path {_robot_base_prim_path()!r})"
    )
    return root, stand, robot_base


def _bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])


def _world_aligned_range(prim: Usd.Prim, cache: UsdGeom.BBoxCache | None = None):
    cache = cache or _bbox_cache()
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _stand_scale(stand_prim: Usd.Prim) -> Gf.Vec3d:
    for op in UsdGeom.Xformable(stand_prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            value = op.Get()
            assert value is not None, f"stand scale op has no value on {stand_prim.GetPath()}"
            return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))
    return Gf.Vec3d(1.0, 1.0, 1.0)


def _set_stand_scale_z(stand_prim: Usd.Prim, scale_z: float) -> None:
    xform = UsdGeom.Xformable(stand_prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            value = op.Get()
            assert value is not None, f"stand scale op has no value on {stand_prim.GetPath()}"
            op.Set(type(value)(float(value[0]), float(value[1]), float(scale_z)))
            return
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, float(scale_z)))


def _add_translate_z(prim: Usd.Prim, delta_z: float) -> None:
    """Add ``delta_z`` to the prim's translate op (create one if missing)."""
    if abs(delta_z) <= 1e-12:
        return
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            assert value is not None, f"translate op has no value on {prim.GetPath()}"
            op.Set(type(value)(float(value[0]), float(value[1]), float(value[2]) + float(delta_z)))
            return
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(delta_z)))


def measure_native_stand_height_m(usd_path: str) -> float:
    """Return the stand's current world z-height in meters from the given USD.

    Args:
        usd_path: Path or Nucleus URI to a Franka on-stand USD.

    Returns:
        Stand world AABB height in meters.
    """
    resolved = retrieve_file_path(usd_path)
    stage = Usd.Stage.Open(resolved)
    assert stage is not None, f"could not open Franka on-stand USD: {usd_path}"
    _, stand, _ = assert_franka_on_stand_prims(stage)
    height = float(_world_aligned_range(stand).GetSize()[2])
    assert height > 0.0, f"non-positive Franka stand height {height} from {usd_path}"
    return height


def ensure_franka_stand_height_usd(
    source_usd_path: str,
    stand_height_m: float,
    *,
    cache_dir: Path | None = None,
    height_eps: float = _DEFAULT_HEIGHT_EPS,
) -> str:
    """Return a USD path whose stand height is ``stand_height_m`` (absolute meters).

    Scales the stand child, keeps stand-top / robot-base contact, then shifts both the stand
    and robot-base by ``(stand_height_m - H0)`` so the stand bottom stays at the stock
    ``-H0`` (``H0`` = native stand height). When ``stand_height_m`` matches ``H0``, returns
    ``source_usd_path`` unchanged.

    Args:
        source_usd_path: Stock Franka on-stand USD path (Nucleus URI or local).
        stand_height_m: Desired absolute stand height in meters.
        cache_dir: Optional bake cache directory.
        height_eps: Tolerance for treating the request as the stock height.

    Returns:
        Spawnable USD path (stock URI or a local baked cache file).
    """
    assert stand_height_m > 0.0, f"stand_height_m must be positive, got {stand_height_m}"

    resolved = retrieve_file_path(source_usd_path)
    native_height = measure_native_stand_height_m(resolved)
    if abs(stand_height_m - native_height) <= height_eps:
        return source_usd_path

    cache_root = cache_dir or _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    # v4: stand bottom fixed at -H0; stand+robot shift by (stand_height_m - H0).
    cache_key = hashlib.sha1(f"{resolved}:{stand_height_m:.6f}:bottom_at_neg_h0_v4".encode()).hexdigest()[:16]
    out_path = cache_root / f"franka_on_stand_{cache_key}.usd"
    if out_path.exists():
        return str(out_path)

    stage = Usd.Stage.Open(resolved)
    assert stage is not None, f"could not open Franka on-stand USD: {source_usd_path}"
    _, stand, robot_base = assert_franka_on_stand_prims(stage)

    scale = _stand_scale(stand)
    assert scale[2] > 0.0, f"non-positive stand scale_z {scale[2]} on {stand.GetPath()}"
    unit_height = native_height / float(scale[2])
    _set_stand_scale_z(stand, stand_height_m / unit_height)

    # Keep stand top flush with the robot base, then shift both so the bottom stays at -H0.
    cache = _bbox_cache()
    stand_max_z = float(_world_aligned_range(stand, cache).GetMax()[2])
    robot_min_z = float(_world_aligned_range(robot_base, cache).GetMin()[2])
    _add_translate_z(stand, robot_min_z - stand_max_z)

    delta_z = stand_height_m - native_height
    _add_translate_z(stand, delta_z)
    _add_translate_z(robot_base, delta_z)

    stage.Export(str(out_path))
    return str(out_path)
