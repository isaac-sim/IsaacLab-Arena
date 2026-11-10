# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometry computed from one native object spawn configuration before spawning."""

from __future__ import annotations

from copy import deepcopy

from isaaclab.sim import (
    CapsuleCfg,
    ConeCfg,
    CuboidCfg,
    CylinderCfg,
    MultiAssetSpawnerCfg,
    MultiUsdFileCfg,
    SphereCfg,
    UsdFileCfg,
)
from isaaclab.sim.spawners.shapes.shapes_cfg import ShapeCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.assets import retrieve_file_path
from pxr import Sdf, Usd

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_stage
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage
from isaaclab_arena.utils.usd.variants import apply_usd_variant_selections, usd_variant_selections_to_dict


def get_geometry_key(spawn_cfg: SpawnerCfg) -> tuple:
    """Return the spawn settings that an object's bounds and contact body depend on, as a hashable key.

    Physics, rendering and semantic settings are not part of the key, so editing them keeps cached geometry.

    Args:
        spawn_cfg: One concrete native spawn configuration.

    Returns:
        A tuple that changes whenever the bounds or the contact body could change.
    """
    _assert_single_spawn_cfg(spawn_cfg)
    if isinstance(spawn_cfg, UsdFileCfg):
        scale = tuple(spawn_cfg.scale) if spawn_cfg.scale is not None else None
        variant_selections = tuple(sorted(usd_variant_selections_to_dict(spawn_cfg.variants).items()))
        return ("UsdFileCfg", str(spawn_cfg.usd_path), scale, variant_selections)
    if isinstance(spawn_cfg, ShapeCfg):
        has_rigid_body = spawn_cfg.rigid_props is not None
        return (type(spawn_cfg).__name__, _get_shape_half_extents(spawn_cfg), has_rigid_body)
    # Other spawners have no geometry to derive. SpawnCfgGeometry reports that when it is queried.
    return (type(spawn_cfg).__name__,)


class SpawnCfgGeometry:
    """Bounds and contact body computed from one native spawn configuration, before anything is spawned."""

    def __init__(self, spawn_cfg: SpawnerCfg):
        _assert_single_spawn_cfg(spawn_cfg)
        self.spawn_cfg = deepcopy(spawn_cfg)
        """Independent copy of the spawn settings, so later edits cannot change cached results."""

        self._bounding_box: AxisAlignedBoundingBox | None = None
        self._contact_body_path: str | None = None

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return local bounds relative to the spawned asset root.

        USD bounds are measured from the default prim's origin, with its authored rotation kept.
        A tuple spawn scale replaces the authored root scale, and None keeps it.
        """
        if self._bounding_box is None:
            if isinstance(self.spawn_cfg, UsdFileCfg):
                self._bounding_box = compute_local_bounding_box_from_stage(
                    self.open_usd_stage(), scale=self.spawn_cfg.scale
                )
            else:
                half_extents = _get_shape_half_extents(self.spawn_cfg)
                self._bounding_box = AxisAlignedBoundingBox(tuple(-extent for extent in half_extents), half_extents)
        return self._bounding_box

    def get_contact_body_path(self) -> str:
        """Return the shallowest rigid body's path relative to the spawned asset root, or "" for the root."""
        if self._contact_body_path is None:
            if isinstance(self.spawn_cfg, UsdFileCfg):
                body_path = find_shallowest_rigid_body_from_stage(
                    self.open_usd_stage(), within_default_prim=True, relative_to_default_prim=True
                )
                assert body_path is not None, f"No rigid body found in {self.spawn_cfg.usd_path}."
            else:
                assert isinstance(self.spawn_cfg, ShapeCfg), (
                    "Contact-body discovery supports USD and primitive shape spawners; "
                    f"got {type(self.spawn_cfg).__name__}."
                )
                assert (
                    self.spawn_cfg.rigid_props is not None
                ), "Primitive rigid objects need rigid_props so Isaac Lab creates their rigid body."
                body_path = ""
            self._contact_body_path = body_path
        return self._contact_body_path

    def open_usd_stage(self) -> Usd.Stage:
        """Reference the USD asset at /Asset in a new in-memory stage with its USD variant selections applied.

        The source file is never edited.
        """
        assert isinstance(self.spawn_cfg, UsdFileCfg), "A USD spawn configuration is required."
        usd_path = retrieve_file_path(self.spawn_cfg.usd_path)
        layer = Sdf.Layer.FindOrOpen(usd_path)
        assert layer is not None and layer.defaultPrim, f"USD asset {self.spawn_cfg.usd_path} needs a default prim."
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Asset")
        root.GetReferences().AddReference(usd_path)
        stage.SetDefaultPrim(root)
        apply_usd_variant_selections(stage, self.spawn_cfg.variants)
        return stage


def _get_shape_half_extents(spawn_cfg: SpawnerCfg) -> tuple[float, float, float]:
    """Return the half extents of a standard Isaac Lab primitive shape along x, y and z."""
    if isinstance(spawn_cfg, CuboidCfg):
        half_extents = [dimension / 2 for dimension in spawn_cfg.size]
    elif isinstance(spawn_cfg, SphereCfg):
        half_extents = [spawn_cfg.radius] * 3
    elif isinstance(spawn_cfg, (CylinderCfg, ConeCfg, CapsuleCfg)):
        half_extents = [spawn_cfg.radius] * 3
        axis = "XYZ".index(spawn_cfg.axis)
        half_extents[axis] = spawn_cfg.height / 2
        if isinstance(spawn_cfg, CapsuleCfg):
            half_extents[axis] += spawn_cfg.radius
    else:
        assert False, f"Set Object.bounding_box for objects spawned with {type(spawn_cfg).__name__}."
    return tuple(half_extents)


def _assert_single_spawn_cfg(spawn_cfg: SpawnerCfg) -> None:
    """Reject multi-asset spawners, which have no single geometry."""
    assert not isinstance(
        spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
    ), "Geometry requires one concrete native spawn configuration."
