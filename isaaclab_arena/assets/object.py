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
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_addons
from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import get_asset_indices_by_env
from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry, get_geometry_key
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import has_light, open_stage
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths


class Object(RootedObjectBase):
    """Scene object spawned from a USD file or a spawn configuration, as a rigid, articulated or static (``BASE``) asset."""

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
        **kwargs,
    ):
        # Pull out addons (and remove them from kwargs before passing to super)
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        assert usd_path is not None or spawner_cfg is not None, "Either usd_path or spawner_cfg must be provided"
        assert usd_path is None or spawner_cfg is None, "Either usd_path or spawner_cfg must be provided (not both)"
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
        With several spawn entries, returns the bounds of the tallest entry.
        """
        if self.bounding_box is not None:
            return self.bounding_box
        spawn_entries = self._get_spawn_entries()
        if len(spawn_entries) == 1:
            return self._get_spawn_cfg_geometry(spawn_entries[0]).get_bounding_box()
        entry_bounds = [self._get_spawn_cfg_geometry(entry).get_bounding_box() for entry in spawn_entries]
        return max(entry_bounds, key=lambda bounds: bounds.size[0, 2].item())

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return local bounds with one row per environment, from the spawn entry each environment spawns.

        bounding_box, when set, applies to every environment. With several spawn entries, each environment's
        entry comes from object_cfg.spawn.asset_indices_by_env.

        Args:
            num_envs: Number of environments.

        Returns:
            Bounds whose min_point and max_point have shape (num_envs, 3), in environment order.
        """
        if self.bounding_box is not None or not self.has_multiple_assets:
            return super().get_bounding_box_per_env(num_envs)
        asset_indices_by_env = get_asset_indices_by_env(self.object_cfg.spawn, num_envs, self.name)
        entry_bounds = [self._get_spawn_cfg_geometry(entry).get_bounding_box() for entry in self._get_spawn_entries()]
        return AxisAlignedBoundingBox(
            min_point=torch.stack([entry_bounds[index].min_point[0] for index in asset_indices_by_env]),
            max_point=torch.stack([entry_bounds[index].max_point[0] for index in asset_indices_by_env]),
        )

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
        """Return the scene prim path where this object's contact sensor is attached.

        All spawn entries of object_cfg.spawn must have their rigid body at the same path.
        """
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        # Isaac Lab finds the rigid body in one environment and uses its path in every environment.
        spawn_entries = self._get_spawn_entries()
        body_paths = {self._get_spawn_cfg_geometry(entry).get_contact_body_path() for entry in spawn_entries}
        assert len(body_paths) == 1, (
            f"Object '{self.name}' spawn entries must have the same contact-sensor prim path; "
            f"got rigid-body paths {sorted(body_paths)}."
        )
        return self.prim_path + body_paths.pop()

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

    @property
    def has_multiple_assets(self) -> bool:
        """Whether object_cfg.spawn is a MultiAssetSpawnerCfg with more than one spawn entry."""
        spawn_cfg = self.object_cfg.spawn
        return isinstance(spawn_cfg, MultiAssetSpawnerCfg) and len(spawn_cfg.assets_cfg) > 1

    def _get_spawn_entries(self) -> list[SpawnerCfg]:
        """Return the assets_cfg entries of a MultiAssetSpawnerCfg object_cfg.spawn, or object_cfg.spawn alone."""
        spawn_cfg = self.object_cfg.spawn
        if isinstance(spawn_cfg, MultiAssetSpawnerCfg):
            assert spawn_cfg.assets_cfg, f"Object '{self.name}' needs at least one spawn entry."
            return spawn_cfg.assets_cfg
        return [spawn_cfg]

    def _get_spawn_cfg_geometry(self, spawn_cfg: SpawnerCfg) -> SpawnCfgGeometry:
        """Return cached geometry for a spawn configuration, recomputed when its geometric settings change."""
        key = get_geometry_key(spawn_cfg)
        if key not in self._spawn_cfg_geometry_by_key:
            self._spawn_cfg_geometry_by_key[key] = SpawnCfgGeometry(spawn_cfg)
        return self._spawn_cfg_geometry_by_key[key]

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False) -> SpawnerCfg:
        """Return the custom spawn config if set, otherwise a USD spawn config with addons."""
        if self.spawner_cfg is not None:
            assert not self.spawn_cfg_addon, (
                "spawn_cfg_addon cannot be combined with spawner_cfg. "
                "Configure spawn options and physics on the custom spawner instead."
            )
            return self.spawner_cfg
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
