# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import trimesh
from collections.abc import Collection
from copy import deepcopy
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_geometry import ObjectGeometry
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_addons
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import has_light, open_stage
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths


class Object(RootedObjectBase):
    """A scene object whose native spawn configuration owns its asset settings."""

    def __getattribute__(self, name: str) -> Any:
        # Library subclasses declare class defaults under these names, which would
        # shadow properties on Object. Instance access must use the current config.
        if name in ("scale", "usd_path"):
            return getattr(super().__getattribute__("spawn_cfg"), name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("scale", "usd_path"):
            raise AttributeError(f"Configure {name} through spawn_cfg.{name}")
        super().__setattr__(name, value)

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType | None = None,
        usd_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        spawn_cfg: SpawnerCfg | None = None,
        **kwargs,
    ):
        """Create one concrete asset shared across environments through native spawning."""
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        assert (usd_path is None) != (spawn_cfg is None), "Provide exactly one of usd_path or spawn_cfg"
        if spawn_cfg is not None:
            assert object_type is not None, "object_type must be provided if spawn_cfg is provided"
            assert not spawn_cfg_addon, "Configure spawn options directly on spawn_cfg"
            assert not isinstance(
                spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), "Use PerEnvironmentObject to select different objects across environments"
            spawn_cfg = deepcopy(spawn_cfg)
        else:
            if object_type is None:
                object_type = detect_object_type(usd_path=usd_path, variants=spawn_cfg_addon.get("variants"))
            spawn_cfg = make_usd_spawn_cfg_with_addons(
                UsdFileCfg(
                    usd_path=usd_path,
                    scale=scale,
                    activate_contact_sensors=object_type in (ObjectType.RIGID, ObjectType.ARTICULATION),
                ),
                spawn_cfg_addon,
            )
        self._initialize_object(name, prim_path, object_type, spawn_cfg, initial_pose, relations, **kwargs)

    def _initialize_object(
        self,
        name: str,
        prim_path: str | None,
        object_type: ObjectType,
        spawn_cfg: SpawnerCfg,
        initial_pose: Pose | None,
        relations: list[RelationBase] | None,
        **kwargs,
    ) -> None:
        """Initialize native scene configuration and pose state for Object and PerEnvironmentObject."""
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        super().__init__(name=name, prim_path=prim_path, object_type=object_type, **kwargs)
        self.initial_pose = initial_pose
        self.relations = list(relations or [])
        self.reset_pose = True
        self.bounding_box: AxisAlignedBoundingBox | None = None
        self._geometry: ObjectGeometry | None = None
        cfg_options = deepcopy(asset_cfg_addon)
        if object_type == ObjectType.ARTICULATION:
            cfg_options.setdefault("actuators", {})
            cfg_type = ArticulationCfg
        elif object_type == ObjectType.RIGID:
            cfg_type = RigidObjectCfg
        else:
            cfg_type = AssetBaseCfg
            if isinstance(spawn_cfg, UsdFileCfg):
                with open_stage(spawn_cfg.usd_path) as stage:
                    if has_light(stage):
                        print("WARNING: Base object has lights, which may cause issues with multiple environments.")
        self.object_cfg = cfg_type(prim_path=self.prim_path, spawn=spawn_cfg, **cfg_options)
        if initial_pose is not None:
            self._set_initial_pose(initial_pose)
        self._pose_event_cfg = self._build_reset_event()

    @property
    def spawn_cfg(self) -> SpawnerCfg:
        """The native configuration used for both spawning and geometry queries."""
        return self.object_cfg.spawn

    @spawn_cfg.setter
    def spawn_cfg(self, value: SpawnerCfg) -> None:
        assert not isinstance(
            value, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
        ), "Construct PerEnvironmentObject to configure native alternatives"
        self.object_cfg.spawn = value
        self._geometry = None

    def _get_geometry(self) -> ObjectGeometry:
        """Refresh derived geometry after this object's native configuration changes."""
        assert not isinstance(
            self.spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
        ), "Use PerEnvironmentObject for alternatives"
        if self._geometry is None or not self._geometry.matches(self.spawn_cfg):
            self._geometry = ObjectGeometry(self.spawn_cfg, self.object_type)
        return self._geometry

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return local bounds in the frame used to write this object's pose."""
        return self.bounding_box if self.bounding_box is not None else self._get_geometry().get_bounding_box()

    def get_bounding_box_for_env(self, env_id: int) -> AxisAlignedBoundingBox:
        """Return the same local bounds for every environment."""
        assert env_id >= 0, "Environment index must be non-negative"
        return self.get_bounding_box()

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return collision geometry in the same frame as placement bounds."""
        return self._get_geometry().get_collision_mesh(excluded_prim_paths)

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
        """Return this object's rigid-body path."""
        return self.prim_path + self._get_geometry().get_contact_body_path()

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        """Configure contacts against the target's current rigid-body paths."""
        if contact_against_object is not None:
            assert isinstance(
                contact_against_object, RootedObjectBase
            ), "Contact sensors against deformable objects and other non-rooted objects are not supported"
        contact_sensor_prim_path = self.get_contact_sensor_prim_path()
        if isinstance(contact_against_object, Object) and contact_against_object.object_type == ObjectType.BASE:
            target_spawn_cfg = contact_against_object.spawn_cfg
            assert isinstance(target_spawn_cfg, UsdFileCfg), "BASE contact targets require a USD spawn configuration"
            target_variants = target_spawn_cfg.variants
            if target_variants is not None and not isinstance(target_variants, dict):
                target_variants = target_variants.to_dict()
            body_paths = read_asset_rigid_body_paths(target_spawn_cfg.usd_path, variants=target_variants)
            assert body_paths, "Contact targets must contain rigid bodies."
            filter_prim_paths = []
            for path in body_paths:
                filter_prim_paths.append(contact_against_object.get_prim_path() + path.removeprefix("/Asset"))
        elif isinstance(contact_against_object, Object):
            filter_prim_paths = [contact_against_object.get_contact_sensor_prim_path()]
        elif isinstance(contact_against_object, ObjectBase):
            # References already name the body in their configured scene path.
            filter_prim_paths = [contact_against_object.get_prim_path()]
        else:
            filter_prim_paths = []
        return ContactSensorCfg(prim_path=contact_sensor_prim_path, filter_prim_paths_expr=filter_prim_paths)

    def _requires_reset_pose_event(self) -> bool:
        return super()._requires_reset_pose_event() and self.reset_pose
