# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone


@clone
def spawn_from_usd_and_add_articulation_root(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD file and apply ArticulationRootAPI to the root prim.

    Use this for USD assets that have joints connecting rigid body links but
    were authored without an ArticulationRootAPI. The API is applied to the
    root prim after loading, so Isaac Lab can initialise the asset as an
    articulation.

    The function delegates to the standard USD spawning pipeline
    (``_spawn_from_usd_file``) and then calls
    ``schemas.define_articulation_root_properties`` which creates the
    ArticulationRootAPI if it is not already present.

    Args:
        prim_path: The prim path or pattern to spawn the asset at.
        cfg: The UsdFileCfg configuration instance.
        translation: Translation w.r.t. parent prim. Defaults to None.
        orientation: Orientation (x, y, z, w) w.r.t. parent prim. Defaults to None.

    Returns:
        The prim of the spawned asset.
    """
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    if cfg.articulation_props is not None:
        schemas.define_articulation_root_properties(prim_path, cfg.articulation_props)
    return prim
