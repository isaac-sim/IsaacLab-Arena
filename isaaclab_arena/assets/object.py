# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import trimesh
from collections.abc import Collection, Sequence
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
from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths


class Object(RootedObjectBase):
    """A scene object whose native spawn configuration owns its asset settings."""

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType | None = None,
        usd_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        spawner_cfg: SpawnerCfg | None = None,
        variants: Sequence[SpawnerCfg] | None = None,
        random_choice: bool = False,
        **kwargs,
    ):
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        source_count = sum(source is not None for source in (usd_path, spawner_cfg, variants))
        assert source_count == 1, "Provide exactly one of usd_path, spawner_cfg, or variants"
        if variants is not None:
            assert variants, "Object variants require at least one native spawn configuration"
            assert object_type in (None, ObjectType.RIGID), "Object variants support rigid objects only"
            assert not spawn_cfg_addon and scale == (1.0, 1.0, 1.0), "Configure spawn options on each variant"
            for variant_cfg in variants:
                assert isinstance(variant_cfg, SpawnerCfg), "Object variants must be native spawn configurations"
                assert not isinstance(
                    variant_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
                ), "Object variants must each spawn one rigid body; nested multi-spawners are not supported"
            object_type = ObjectType.RIGID
            spawn_configs = prepare_rigid_object_variants(variants)
            spawn_cfg = spawn_configs[0]
            if len(spawn_configs) > 1:
                spawn_cfg = MultiAssetSpawnerCfg(
                    assets_cfg=spawn_configs,
                    random_choice=False,
                    # None keeps each member's native setting; the wrapper's False default replaces it.
                    activate_contact_sensors=None,
                )
        elif spawner_cfg is not None:
            assert object_type is not None, "object_type must be provided if spawner_cfg is provided"
            assert not spawn_cfg_addon, "Configure spawn options directly on spawner_cfg"
            assert not isinstance(
                spawner_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), "Use variants with concrete native spawn configurations instead of passing a multi-spawner"
            spawn_cfg = deepcopy(spawner_cfg)
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
        super().__init__(name=name, prim_path=prim_path, object_type=object_type, **kwargs)
        self.initial_pose = initial_pose
        self.relations = list(relations or [])
        self.reset_pose = True
        self.bounding_box: AxisAlignedBoundingBox | None = None
        self.random_choice = random_choice
        self._variant_indices_by_env: tuple[int, ...] | None = None
        self._geometry: dict[int, ObjectGeometry] = {}
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
        ), "Construct Object(variants=...) to configure native alternatives"
        self.object_cfg.spawn = value
        self._geometry.clear()

    @property
    def has_variants(self) -> bool:
        """Whether this scene object can have different geometry across environments."""
        return len(self._get_variant_spawn_configs()) > 1

    @property
    def variant_indices_by_env(self) -> tuple[int, ...] | None:
        """The fixed environment-to-variant assignment established before placement."""
        return self._variant_indices_by_env

    def bind_variant_assignment(self, indices: tuple[int, ...]) -> None:
        """Bind one variant per environment without allowing reassignment.

        The builder snapshots native configurations alongside this assignment and
        rejects later source changes before constructing the simulation scene.
        """
        indices = tuple(indices)
        variant_count = len(self._get_variant_spawn_configs())
        assert indices and all(
            type(index) is int and 0 <= index < variant_count for index in indices
        ), f"Object '{self.name}' has invalid variant indices."
        assert self._variant_indices_by_env in (
            None,
            indices,
        ), f"Object '{self.name}' already has a different variant assignment; construct a new object for a new scene."
        self._variant_indices_by_env = indices

    def as_variant(self) -> SpawnerCfg:
        """Copy this rigid singleton's native spawn configuration for Object(variants=...)."""
        assert self.object_type == ObjectType.RIGID, "Only rigid objects can supply a spawn variant"
        assert not self.has_variants, f"Select a concrete native variant of '{self.name}'"
        assert (
            self.bounding_box is None
        ), "Variants derive bounds from native geometry and cannot copy a bounds override"
        return deepcopy(self._get_variant_spawn_configs()[0])

    def _get_variant_spawn_configs(self) -> list[SpawnerCfg]:
        """Read alternatives from the authoritative native spawn configuration."""
        assert not isinstance(self.spawn_cfg, MultiUsdFileCfg), "Use Object(variants=...) instead of MultiUsdFileCfg"
        if isinstance(self.spawn_cfg, MultiAssetSpawnerCfg):
            assert self.object_type == ObjectType.RIGID, "Object variants support rigid objects only"
            assert self.spawn_cfg.assets_cfg, f"Object '{self.name}' requires at least one native variant"
            return self.spawn_cfg.assets_cfg
        return [self.spawn_cfg]

    def _get_geometry(self, variant_index: int = 0) -> ObjectGeometry:
        """Refresh a concrete variant's derived geometry after its native configuration changes."""
        spawn_cfg = self._get_variant_spawn_configs()[variant_index]
        geometry = self._geometry.get(variant_index)
        if geometry is None or not geometry.matches(spawn_cfg):
            geometry = ObjectGeometry(spawn_cfg, self.object_type)
            self._geometry[variant_index] = geometry
        return geometry

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return local bounds in the frame used to write this object's pose."""
        assert not self.has_variants, f"Object '{self.name}' requires per-environment bounding boxes"
        return self.bounding_box if self.bounding_box is not None else self._get_geometry().get_bounding_box()

    def get_bounding_box_for_env(self, env_id: int) -> AxisAlignedBoundingBox:
        """Return the assigned variant's local bounds for one environment."""
        assert env_id >= 0, "Environment index must be non-negative"
        if not self.has_variants:
            return self.get_bounding_box()
        indices = self.variant_indices_by_env
        assert indices is not None, f"Object '{self.name}' needs a variant assignment before geometry queries"
        assert env_id < len(indices), f"Object '{self.name}' has no assignment for environment {env_id}"
        return self._get_geometry(indices[env_id]).get_bounding_box()

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return assigned local bounds with one row per environment."""
        assert num_envs > 0, "Per-environment bounds require at least one environment"
        if not self.has_variants:
            bounds = self.get_bounding_box()
            return AxisAlignedBoundingBox(bounds.min_point.expand(num_envs, 3), bounds.max_point.expand(num_envs, 3))
        indices = self.variant_indices_by_env
        assert (
            indices is not None and len(indices) == num_envs
        ), f"Object '{self.name}' needs a variant assignment for {num_envs} environments before geometry queries"
        bounds = [
            self._get_geometry(index).get_bounding_box() for index in range(len(self._get_variant_spawn_configs()))
        ]
        return AxisAlignedBoundingBox(
            min_point=torch.stack([bounds[index].min_point[0] for index in indices]),
            max_point=torch.stack([bounds[index].max_point[0] for index in indices]),
        )

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return collision geometry in the same frame as placement bounds."""
        if self.has_variants:
            assert not excluded_prim_paths, "Prim exclusions require a concrete object variant"
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
        """Return the rigid-body path shared by every native spawn variant."""
        body_paths = {
            self._get_geometry(index).get_contact_body_path() for index in range(len(self._get_variant_spawn_configs()))
        }
        assert len(body_paths) == 1, f"Object '{self.name}' has incompatible rigid-body paths"
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
