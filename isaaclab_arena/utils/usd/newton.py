# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD compatibility helpers for Newton/MuJoCo-Warp."""

from __future__ import annotations

import shutil
from pathlib import Path


def ensure_newton_valid_rigid_body_inertias_usd(
    usd_path: str,
    min_mass: float = 0.02,
    min_diagonal_inertia: float = 1.0e-5,
) -> str:
    """Return a cached USD copy with positive mass and inertia on rigid bodies."""
    from isaaclab.utils.assets import retrieve_file_path

    source = Path(retrieve_file_path(usd_path, force_download=False))
    assert source.is_file(), f"USD path must resolve to a local file: {usd_path}"

    target = source.with_name(f"{source.stem}_newton_inertia{source.suffix}")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime and _has_valid_rigid_body_inertias(target):
        return str(target)

    shutil.copy2(source, target)
    _author_minimum_rigid_body_inertias(target, min_mass=min_mass, min_diagonal_inertia=min_diagonal_inertia)
    return str(target)


def _has_valid_rigid_body_inertias(usd_path: Path) -> bool:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"
    return all(not (prim.HasAPI(UsdPhysics.RigidBodyAPI) and _needs_minimum_inertia(prim)) for prim in stage.Traverse())


def _author_minimum_rigid_body_inertias(
    usd_path: Path,
    min_mass: float,
    min_diagonal_inertia: float,
) -> None:
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    diagonal_inertia = Gf.Vec3f(min_diagonal_inertia, min_diagonal_inertia, min_diagonal_inertia)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not _needs_minimum_inertia(prim):
            continue

        mass_api = UsdPhysics.MassAPI(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)

        if _invalid_positive_value(mass_api.GetMassAttr().Get()):
            mass_api.CreateMassAttr().Set(min_mass)
        if _invalid_diagonal_inertia(mass_api.GetDiagonalInertiaAttr().Get()):
            mass_api.CreateDiagonalInertiaAttr().Set(diagonal_inertia)

    stage.GetRootLayer().Save()


def _needs_minimum_inertia(prim) -> bool:
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.MassAPI):
        return True
    mass_api = UsdPhysics.MassAPI(prim)
    return _invalid_positive_value(mass_api.GetMassAttr().Get()) or _invalid_diagonal_inertia(
        mass_api.GetDiagonalInertiaAttr().Get()
    )


def _invalid_positive_value(value) -> bool:
    return value is None or float(value) <= 0.0


def _invalid_diagonal_inertia(value) -> bool:
    return value is None or any(float(component) <= 0.0 for component in value)
