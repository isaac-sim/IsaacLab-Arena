# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared robot-on-stand USD compose helpers for Franka-based embodiments."""

from __future__ import annotations

import functools
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path
from pxr import Gf, Usd, UsdGeom

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir

_ROBOT_ON_STAND_USD_CACHE_DIR = "robot_on_stand"

_HEIGHT_ATOL = 1e-3
_ALIGN_ATOL = 5e-2


@dataclass(frozen=True)
class RobotPrimSpec:
    """Robot USD and prim layout for on-stand compose."""

    robot_usd_path: str
    root_prim_path: str
    robot_base_prim_name: str
    stand_prim_name: str

    @property
    def robot_base_prim_path(self) -> str:
        """PhysX articulation root link path under the composed default prim."""
        return f"{self.root_prim_path}/{self.robot_base_prim_name}"

    @property
    def stand_prim_path(self) -> str:
        """Stand mount prim path parented under the robot base link."""
        return f"{self.robot_base_prim_path}/{self.stand_prim_name}"


@dataclass(frozen=True)
class StandPrimSpec:
    """Stand USD reference and footprint for normalized on-stand compose."""

    stand_usd_path: str
    ref_prim_path: str
    payload_child_name: str
    footprint_translate_xyz: tuple[float, float, float]
    footprint_scale_xy: tuple[float, float]
    stand_default_height: float


@functools.cache
def compose_on_stand_usd(
    robot: RobotPrimSpec,
    stand: StandPrimSpec,
    *,
    stand_height_m: float,
    output_basename: str,
) -> str:
    """Build a robot+stand USD with stand under the robot base link.

    Composes once per unique arguments, writes the result to ``~/.cache/isaaclab_arena/usd/robot_on_stand/``,
    and returns that path.

    Args:
        robot: Robot USD path and prim layout under the composed default prim.
        stand: Stand reference and footprint parameters.
        stand_height_m: Target absolute stand height after align.
        output_basename: Stable basename for the composed USD (embodiment-specific).

    Returns:
        Local path to the composed on-stand USD.
    """
    assert stand_height_m > 0.0, f"stand_height_m must be positive, got {stand_height_m}"

    cache_root = get_arena_asset_cache_dir().parent / "usd" / _ROBOT_ON_STAND_USD_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{output_basename}_{stand_height_m:.3f}.usd"

    with tempfile.NamedTemporaryFile(suffix=".usd", dir=cache_root, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        stage = Usd.Stage.CreateNew(str(tmp_path))
        root = stage.DefinePrim(robot.root_prim_path, "Xform")
        robot_resolved = retrieve_file_path(robot.robot_usd_path)
        root.GetReferences().AddReference(robot_resolved, robot.root_prim_path)
        stage.SetDefaultPrim(root)

        _mount_stand_normalized(stage, robot, stand, stand_height_m)
        assert stage.GetRootLayer().Save(), f"failed to save composed on-stand USD to {tmp_path}"
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return str(out_path)


def _mount_stand_normalized(
    stage: Usd.Stage,
    robot: RobotPrimSpec,
    stand: StandPrimSpec,
    stand_height_m: float,
) -> None:
    """Parent a stand payload under the robot base link, scale, and align to the robot base.

    Composed USD hierarchy::

        /panda/panda_link0/stand_instanceable     # outer mount
          translate: (footprint_xy, align_z)
          scale:    (footprint_xy, stand_height_m / native_height)
          /<payload_child_name>/                   # inner payload
            reference: stand USD @ ref_prim_path

    Args:
        stage: Composed robot-on-stand stage (robot reference already mounted).
        robot: Robot prim layout under the composed default prim.
        stand: Stand reference and footprint parameters.
        stand_height_m: Target stand height after align.
    """
    # Robot-only stage: bottom of the base link is the align target (before stand exists).
    robot_base = stage.GetPrimAtPath(robot.robot_base_prim_path)
    assert robot_base.IsValid(), f"On-stand USD missing robot base prim at {robot.robot_base_prim_path!r}"
    pre_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    robot_range = pre_cache.ComputeWorldBound(robot_base).ComputeAlignedRange()
    assert not robot_range.IsEmpty(), f"empty robot base bounds at {robot_base.GetPath()}"
    robot_min_z = float(robot_range.GetMin()[2])

    stand_resolved = retrieve_file_path(stand.stand_usd_path)
    tx, ty, tz = stand.footprint_translate_xyz
    sx, sy = stand.footprint_scale_xy
    stand_prim_path = robot.stand_prim_path

    # Outer mount under link0; Z scale stays at 1 until native height is measured.
    stand_xf = UsdGeom.Xform.Define(stage, stand_prim_path)
    translate_op = stand_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(tx, ty, tz))
    scale_op = stand_xf.AddScaleOp()
    scale_op.Set(Gf.Vec3d(sx, sy, 1.0))

    payload_prim = stage.DefinePrim(f"{stand_prim_path}/{stand.payload_child_name}")
    payload_prim.GetReferences().AddReference(stand_resolved, stand.ref_prim_path)

    stand_prim = stand_xf.GetPrim()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # Native height at footprint XY scale; then scale Z to the requested stand height.
    stand_range = bbox_cache.ComputeWorldBound(stand_prim).ComputeAlignedRange()
    assert not stand_range.IsEmpty(), f"empty stand bounds at {stand_prim.GetPath()}"
    native_height = float(stand_range.GetSize()[2])
    assert native_height > 0.0, f"non-positive stand height at {stand_prim.GetPath()}"
    scale_op.Set(Gf.Vec3d(sx, sy, stand_height_m / native_height))

    # Raise/lower outer translate so stand top meets robot_min_z.
    translate_op.Set(Gf.Vec3d(tx, ty, robot_min_z))

    # Verify stand/robot alignment and stand height on a fresh BBoxCache.
    verify_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    stand_range = verify_cache.ComputeWorldBound(stand_prim).ComputeAlignedRange()
    stand_height = float(stand_range.GetSize()[2])
    stand_max_z = float(stand_range.GetMax()[2])
    assert abs(stand_height - stand_height_m) < _HEIGHT_ATOL, stand_height
    assert abs(stand_max_z - robot_min_z) < _ALIGN_ATOL, (stand_max_z, robot_min_z)
