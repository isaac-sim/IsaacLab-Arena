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

    # TODO(cvolk, 2026.10.09): [object-config-migration] Rename library class defaults
    # before replacing scale/usd_path forwarding with normal properties.
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
        asset: tuple[str, SpawnerCfg] | None = None,
        **kwargs,
    ):
        """Declare a scene object with native asset settings or a build-time asset choice.

        Args:
            asset: A library name and native rigid spawn configuration, as returned by
                AssetRegistry.get_asset_definition(). Cannot be combined with usd_path or spawn_cfg.

        Without asset settings, the object declares a rigid body whose spawn configuration
        must be supplied before placement and scene construction.
        """
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        assert (
            sum(value is not None for value in (asset, usd_path, spawn_cfg)) <= 1
        ), "Provide at most one of asset, usd_path or spawn_cfg"
        if asset is not None:
            assert (
                isinstance(asset, tuple) and len(asset) == 2
            ), "asset must contain a library name and spawn configuration"
            asset_name, spawn_cfg = asset
            assert isinstance(asset_name, str) and asset_name.strip(), "Asset names must be nonempty."
            assert isinstance(spawn_cfg, SpawnerCfg), "asset must contain a native spawn configuration"
            assert object_type in (None, ObjectType.RIGID), "Library asset definitions support rigid objects only"
            assert scale == (1.0, 1.0, 1.0), "Configure asset scale in its native spawn configuration"
            object_type = ObjectType.RIGID
        if spawn_cfg is not None:
            assert object_type is not None, "object_type must be provided if spawn_cfg is provided"
            assert not spawn_cfg_addon, "Configure spawn options directly on spawn_cfg"
            assert not isinstance(
                spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), "Use RigidObjectSet to select different objects across environments"
            spawn_cfg = deepcopy(spawn_cfg)
        elif usd_path is not None:
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
        else:
            assert object_type in (None, ObjectType.RIGID), "Objects without asset settings must be rigid"
            assert not spawn_cfg_addon and scale == (1.0, 1.0, 1.0), "Supply spawn settings through an asset definition"
            object_type = ObjectType.RIGID
        self._initialize_object(name, prim_path, object_type, spawn_cfg, initial_pose, relations, **kwargs)

    def _initialize_object(
        self,
        name: str,
        prim_path: str | None,
        object_type: ObjectType,
        spawn_cfg: SpawnerCfg | None,
        initial_pose: Pose | None,
        relations: list[RelationBase] | None,
        **kwargs,
    ) -> None:
        """Initialize native scene configuration and pose state for Object and RigidObjectSet."""
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        super().__init__(name=name, prim_path=prim_path, object_type=object_type, **kwargs)
        self.initial_pose = initial_pose
        self.relations = list(relations or [])
        self.reset_pose = True
        self.bounding_box: AxisAlignedBoundingBox | None = None
        self._asset_indices_by_env: tuple[int, ...] | None = None
        self._asset_geometry: dict[int, ObjectGeometry] = {}
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
        self._assert_assets_resolved()
        return self.object_cfg.spawn

    @spawn_cfg.setter
    def spawn_cfg(self, value: SpawnerCfg) -> None:
        assert not isinstance(
            value, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
        ), "Construct RigidObjectSet to configure native alternatives"
        self.object_cfg.spawn = value
        self._asset_geometry.clear()

    def _assert_assets_resolved(self) -> None:
        """Require native asset settings before geometry or scene configuration is requested."""
        assert self.object_cfg.spawn is not None, (
            f"Object '{self.name}' has no asset; supply asset settings or enable an asset-selection variation "
            "before building the environment."
        )

    def _resolve_assets(self, spawn_configs: list[SpawnerCfg], indices: tuple[int, ...]) -> None:
        """Install prepared native configurations with a fixed assignment for each environment."""
        assert (
            self._asset_indices_by_env is None
        ), f"Object '{self.name}' is already resolved; create a fresh environment."
        assert spawn_configs, "Asset resolution requires at least one candidate."
        self.object_cfg.spawn = (
            MultiAssetSpawnerCfg(assets_cfg=spawn_configs, random_choice=False, activate_contact_sensors=None)
            if len(spawn_configs) > 1
            else spawn_configs[0]
        )
        self._asset_geometry.clear()
        self.bind_asset_assignment(indices)

    def get_object_cfg(self) -> tuple[str, AssetBaseCfg]:
        self._assert_assets_resolved()
        return super().get_object_cfg()

    @property
    def has_multiple_assets(self) -> bool:
        """Whether multiple asset alternatives are configured, regardless of assignment."""
        spawn_cfg = self.object_cfg.spawn
        return isinstance(spawn_cfg, MultiAssetSpawnerCfg) and len(spawn_cfg.assets_cfg) > 1

    @property
    def asset_indices_by_env(self) -> tuple[int, ...] | None:
        """Configured asset indices in environment order, fixed before placement."""
        return self._asset_indices_by_env

    def bind_asset_assignment(self, indices: tuple[int, ...]) -> None:
        """Bind one valid asset index per environment without allowing reassignment."""
        indices = tuple(indices)
        asset_count = len(self._get_asset_spawn_configs())
        assert indices and all(
            type(index) is int and 0 <= index < asset_count for index in indices
        ), f"Object '{self.name}' has invalid variant indices."
        assert self._asset_indices_by_env in (
            None,
            indices,
        ), f"Object '{self.name}' already has a different variant assignment; construct a new object for a new scene."
        self._asset_indices_by_env = indices

    def _get_asset_spawn_configs(self) -> list[SpawnerCfg]:
        """Read alternatives from the authoritative native spawn configuration."""
        if isinstance(self.spawn_cfg, MultiAssetSpawnerCfg):
            assert self.spawn_cfg.assets_cfg, f"Object '{self.name}' requires at least one native variant."
            return self.spawn_cfg.assets_cfg
        return [self.spawn_cfg]

    def _get_geometry(self, asset_index: int = 0) -> ObjectGeometry:
        """Refresh one asset's derived geometry after native configuration changes."""
        spawn_cfg = self._get_asset_spawn_configs()[asset_index]
        geometry = self._asset_geometry.get(asset_index)
        if geometry is None or not geometry.matches(spawn_cfg):
            geometry = ObjectGeometry(spawn_cfg, self.object_type)
            self._asset_geometry[asset_index] = geometry
        return geometry

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return local bounds in the frame used to write this object's pose."""
        self._assert_assets_resolved()
        assert not self.has_multiple_assets, f"Object '{self.name}' requires per-environment bounding boxes."
        return self.bounding_box if self.bounding_box is not None else self._get_geometry().get_bounding_box()

    def get_bounding_box_for_env(self, env_id: int) -> AxisAlignedBoundingBox:
        """Return the assigned asset's local bounds for one environment."""
        self._assert_assets_resolved()
        assert env_id >= 0, "Environment index must be non-negative"
        if not self.has_multiple_assets:
            return self.get_bounding_box()
        indices = self.asset_indices_by_env
        assert indices is not None, f"Object '{self.name}' needs a variant assignment before geometry queries."
        assert env_id < len(indices), f"Object '{self.name}' has no assignment for environment {env_id}."
        return self._get_geometry(indices[env_id]).get_bounding_box()

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return assigned local bounds with one row per environment."""
        self._assert_assets_resolved()
        assert num_envs > 0, "Per-environment bounds require at least one environment."
        if not self.has_multiple_assets:
            bounds = self.get_bounding_box()
            return AxisAlignedBoundingBox(bounds.min_point.expand(num_envs, 3), bounds.max_point.expand(num_envs, 3))
        indices = self.asset_indices_by_env
        assert (
            indices is not None and len(indices) == num_envs
        ), f"Object '{self.name}' needs a variant assignment for {num_envs} environments before geometry queries."
        bounds = [self._get_geometry(index).get_bounding_box() for index in range(len(self._get_asset_spawn_configs()))]
        return AxisAlignedBoundingBox(
            min_point=torch.stack([bounds[index].min_point[0] for index in indices]),
            max_point=torch.stack([bounds[index].max_point[0] for index in indices]),
        )

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return collision geometry in the same frame as placement bounds."""
        self._assert_assets_resolved()
        if self.has_multiple_assets:
            assert not excluded_prim_paths, "Object exclusions require a concrete asset."
            return None
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
        """Return the rigid-body path shared by this object's native assets."""
        self._assert_assets_resolved()
        if not self.has_multiple_assets:
            return self.prim_path + self._get_geometry().get_contact_body_path()
        body_paths = {
            self._get_geometry(index).get_contact_body_path() for index in range(len(self._get_asset_spawn_configs()))
        }
        assert len(body_paths) == 1, f"Object '{self.name}' has incompatible rigid-body paths."
        return self.prim_path + body_paths.pop()

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
