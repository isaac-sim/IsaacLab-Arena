# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from copy import deepcopy
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_addons
from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry, get_geometry_key
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import has_light, open_stage
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths
from isaaclab_arena.variations.asset_selection_variation import assert_asset_definition


class Object(RootedObjectBase):
    """Scene object spawned from a USD file or a spawn configuration, as a rigid, articulated or static (``BASE``) asset.

    A rigid object can also be declared with ``asset``, a (name, spawn configuration) pair such as
    AssetRegistry.get_asset_definition() returns, or without an asset, in which case object_cfg.spawn is None
    until a spawn configuration is set.
    """

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType | None = None,
        usd_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_pose: Pose | None = None,
        relations: list[RelationBase] = [],
        spawner_cfg: SpawnerCfg | None = None,
        asset: tuple[str, SpawnerCfg] | None = None,
        **kwargs,
    ):
        # Pull out addons (and remove them from kwargs before passing to super)
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        assert usd_path is None or spawner_cfg is None, "usd_path and spawner_cfg cannot both be provided."
        if asset is not None or (usd_path is None and spawner_cfg is None):
            # The asset comes from asset= or, at build time, from an AssetSelectionVariation, so spawn options
            # belong in that asset's spawn configuration.
            assert (
                usd_path is None and spawner_cfg is None
            ), f"Object '{name}': asset= cannot be combined with usd_path or spawner_cfg."
            assert object_type in (
                None,
                ObjectType.RIGID,
            ), f"Object '{name}' is declared with asset= or without an asset, so it must be rigid; got {object_type}."
            assert (
                tuple(scale) == (1.0, 1.0, 1.0) and not spawn_cfg_addon
            ), f"Object '{name}': set scale and spawn_cfg_addon in the asset's spawn configuration, not on the Object."
            object_type = ObjectType.RIGID
            if asset is not None:
                assert_asset_definition(asset, owner=f"Object '{name}' asset=")
                spawner_cfg = deepcopy(asset[1])
        if spawner_cfg is not None:
            assert object_type is not None, "object_type must be provided if spawner_cfg is provided"
        # Detect object type if not provided
        if object_type is None:
            assert usd_path is not None, (
                "object_type is None (indicating auto-detect) but usd_path is also None. usd_path is required to detect"
                " object type"
            )
            object_type = detect_object_type(usd_path=usd_path, variants=spawn_cfg_addon.get("variants"))
        super().__init__(name=name, prim_path=prim_path, object_type=object_type, **kwargs)
        self.usd_path = usd_path
        self.spawner_cfg = spawner_cfg
        self.scale = scale
        self.initial_pose = initial_pose
        self.relations = list(relations)
        self.reset_pose = True
        # Keep nested addon settings independent when multiple objects reuse the same input mapping.
        self.spawn_cfg_addon = deepcopy(spawn_cfg_addon)
        self.asset_cfg_addon = asset_cfg_addon
        self.bounding_box: AxisAlignedBoundingBox | None = None
        """Explicit local bounds. None derives bounds from object_cfg.spawn."""
        # TODO(cvolk, 2026.10.13): Replace this per-object cache with one cache in spawn_cfg_geometry.py
        # shared across objects, keyed by get_geometry_key(), so objects and asset-selection candidates that
        # use the same asset compute its geometry once.
        self._spawn_cfg_geometry_by_key: dict[tuple, SpawnCfgGeometry] = {}
        """Geometry per spawn configuration, keyed by get_geometry_key(). Editing a geometric setting of
        object_cfg.spawn, or replacing it, gives a new key and new geometry. Physics or appearance edits
        keep the cached entry."""
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin).

        Returns bounding_box when it is set, otherwise bounds derived from object_cfg.spawn.
        """
        if self.bounding_box is not None:
            return self.bounding_box
        return self._get_spawn_cfg_geometry(self.object_cfg.spawn).get_bounding_box()

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        return self.get_bounding_box().get_corners_at(pos)

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self._pose_event_cfg = self._build_reset_event()

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self._pose_event_cfg = self._build_reset_event()

    def get_contact_sensor_prim_path(self) -> str:
        """Return the scene prim path where this object's contact sensor is attached."""
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        return self.prim_path + self._get_spawn_cfg_geometry(self.object_cfg.spawn).get_contact_body_path()

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        # We override this function from the parent class because some assets do not have their rigid body
        # at the root of the USD file. To be robust to this, we find the shallowest rigid body and add the
        # contact sensor to it.
        if contact_against_object is not None:
            assert isinstance(
                contact_against_object, RootedObjectBase
            ), "Contact sensors against deformable objects and other non-rooted objects are not supported"
        contact_sensor_prim_path = self.get_contact_sensor_prim_path()
        if isinstance(contact_against_object, Object) and contact_against_object.object_type == ObjectType.BASE:
            # PhysX needs a separate filter expression for each nested target body.
            body_paths = read_asset_rigid_body_paths(
                contact_against_object.usd_path,
                variants=(contact_against_object.spawn_cfg_addon or {}).get("variants"),
            )
            assert body_paths, "Contact targets must contain rigid bodies."
            filter_prim_paths = []
            for path in body_paths:
                relative_path = path.removeprefix("/Asset")
                filter_prim_paths.append(contact_against_object.get_prim_path() + relative_path)
        elif isinstance(contact_against_object, Object):
            # This branch supports rigid targets only. For ObjectInTask in kitchen scenes,
            # supply receptacles such as the microwave as Background objects so the branch
            # above filters contacts against their nested rigid bodies.
            filter_prim_paths = [contact_against_object.get_contact_sensor_prim_path()]
        elif isinstance(contact_against_object, ObjectBase):
            # Handles ObjectReference.
            # NOTE(alexmillane, 2026.04.10): ObjectReference is assumed to have its rigid body at its scene prim path.
            filter_prim_paths = [contact_against_object.get_prim_path()]
        elif contact_against_object is None:
            filter_prim_paths = []
        return ContactSensorCfg(
            prim_path=contact_sensor_prim_path,
            filter_prim_paths_expr=filter_prim_paths,
        )

    def get_object_cfg(self) -> tuple[str, AssetBaseCfg]:
        """Return the scene key and concrete asset config. The object must have an asset."""
        self._assert_has_asset(self.object_cfg.spawn)
        return super().get_object_cfg()

    def _assert_has_asset(self, spawn_cfg: SpawnerCfg | None) -> None:
        """Assert that ``spawn_cfg`` is set, naming this object and the ways to give it an asset."""
        assert spawn_cfg is not None, (
            f"Object '{self.name}' has no asset. Declare it with asset=, or add an AssetSelectionVariation to it and"
            f" enable it (for example '{self.name}.asset_selection.enabled=true')."
        )

    def _get_spawn_cfg_geometry(self, spawn_cfg: SpawnerCfg | None) -> SpawnCfgGeometry:
        """Return cached geometry for a spawn configuration, recomputed when its geometric settings change."""
        self._assert_has_asset(spawn_cfg)
        key = get_geometry_key(spawn_cfg)
        if key not in self._spawn_cfg_geometry_by_key:
            self._spawn_cfg_geometry_by_key[key] = SpawnCfgGeometry(spawn_cfg)
        return self._spawn_cfg_geometry_by_key[key]

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False) -> SpawnerCfg | None:
        """Return the custom spawn config, a USD spawn config with addons, or None for an object without an asset."""
        if self.spawner_cfg is not None:
            assert not self.spawn_cfg_addon, (
                "spawn_cfg_addon cannot be combined with spawner_cfg. "
                "Configure spawn options and physics on the custom spawner instead."
            )
            return self.spawner_cfg
        if self.usd_path is None:
            return None
        spawn_cfg = UsdFileCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
        )
        return make_usd_spawn_cfg_with_addons(spawn_cfg, self.spawn_cfg_addon)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=True),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(object_cfg)

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=True),
            **self.asset_cfg_addon,
            actuators={},
        )
        return self._add_initial_pose_to_cfg(object_cfg)

    def _generate_base_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.BASE
        if self.spawner_cfg is None:
            with open_stage(self.usd_path) as stage:
                if has_light(stage):
                    print(
                        "WARNING: Base object has lights, this may cause issues when using with multiple environments."
                    )
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(object_cfg)

    def _add_initial_pose_to_cfg(
        self, object_cfg: RigidObjectCfg | ArticulationCfg | AssetBaseCfg
    ) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg:
        # Optionally specify initial pose
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is not None:
            object_cfg.init_state.pos = initial_pose.position_xyz
            object_cfg.init_state.rot = initial_pose.rotation_xyzw
        return object_cfg

    def _requires_reset_pose_event(self) -> bool:
        return super()._requires_reset_pose_event() and self.reset_pose
