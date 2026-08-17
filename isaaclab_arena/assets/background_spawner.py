# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone

from isaaclab_arena.utils.usd.rigid_bodies import freeze_loose_rigid_bodies

if TYPE_CHECKING:
    from pxr import Usd


@clone
def spawn_from_usd_with_frozen_loose_rigid_bodies(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD background and make its rigid bodies without joint connections kinematic."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # spawn_from_usd is also clone-decorated. This outer clone is load-bearing: it copies the
    # kinematic attributes authored below from the modified source prim to every environment.
    freeze_loose_rigid_bodies(prim)
    return prim
