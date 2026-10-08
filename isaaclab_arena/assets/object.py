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
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.assets.object_variant import ObjectVariant
from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_addons
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import has_light, open_stage
from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths


class Object(RootedObjectBase):
    """A named scene object with one or more asset variants."""

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
        variants: list[ObjectVariant] | None = None,
        random_choice: bool = False,
        **kwargs,
    ):
        # Pull out addons (and remove them from kwargs before passing to super)
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        source_count = sum(source is not None for source in (usd_path, spawner_cfg, variants))
        assert source_count == 1, "Provide exactly one of usd_path, spawner_cfg, or variants"
        if variants is not None:
            assert variants, "An object must have at least one variant"
            assert all(isinstance(variant, ObjectVariant) for variant in variants), "Expected ObjectVariant entries"
            assert all(variant.object_type == ObjectType.RIGID for variant in variants), "Object variants must be rigid"
            assert object_type in (None, ObjectType.RIGID), "Object variants must be rigid"
            assert not spawn_cfg_addon and scale == (1.0, 1.0, 1.0), "Configure spawn options on each variant"
            object_type = ObjectType.RIGID
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
        self.relations = list(relations or [])
        self.reset_pose = True
        # Keep nested addon settings independent when multiple objects reuse the same input mapping.
        self.spawn_cfg_addon = deepcopy(spawn_cfg_addon)
        self.asset_cfg_addon = asset_cfg_addon
        self.bounding_box = None
        self.random_choice = random_choice
        self._variant_indices_by_env: tuple[int, ...] | None = None
        if variants is None:
            variants = [ObjectVariant(spawn_cfg=self._get_single_source_spawn_cfg(), object_type=object_type)]
        self.variants = tuple(deepcopy(variants))
        self._bounding_box_overrides = tuple(deepcopy(variant.bounding_box) for variant in self.variants)
        self._configured_variants: list[ObjectVariant | None] = [None] * len(self.variants)
        self._single_variant: ObjectVariant | None = None
        self._prepared_spawn_cfgs: list[SpawnerCfg] | None = None
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()

    @property
    def has_variants(self) -> bool:
        """Whether this scene role can use different geometry across environments."""
        return len(self.variants) > 1

    @property
    def variant_indices_by_env(self) -> tuple[int, ...] | None:
        """The fixed environment-to-variant assignment established before placement."""
        return self._variant_indices_by_env

    def bind_variant_assignment(self, indices: tuple[int, ...]) -> None:
        """Bind the scene's variant indices without allowing reassignment.

        Args:
            indices: One valid variant index per environment.
        """
        assert indices and all(0 <= index < len(self.variants) for index in indices), "Invalid variant assignment"
        assert self._variant_indices_by_env in (
            None,
            indices,
        ), f"Object '{self.name}' already has a different variant assignment; construct a new object for a new scene"
        self._variant_indices_by_env = tuple(indices)

    def as_variant(self) -> ObjectVariant:
        """Return this single object's asset definition, without its scene pose or relations."""
        assert not self.has_variants, "Select a concrete variant instead of nesting object alternatives"
        return ObjectVariant(
            spawn_cfg=self.object_cfg.spawn,
            object_type=self.object_type,
            bounding_box=self.bounding_box if self.bounding_box is not None else self._bounding_box_overrides[0],
        )

    def _get_single_variant(self) -> ObjectVariant:
        """Keep singleton geometry aligned with the configured native spawner."""
        assert not self.has_variants, f"Object '{self.name}' requires per-environment geometry"
        if self._single_variant is None or self._single_variant.spawn_cfg.to_dict() != self.object_cfg.spawn.to_dict():
            self._single_variant = self.as_variant()
        return self._single_variant

    def _get_variant_spawn_configs(self) -> list[SpawnerCfg]:
        """Require the native configuration to retain this object's declared alternatives."""
        spawn_cfg = self.object_cfg.spawn
        assert isinstance(
            spawn_cfg, MultiAssetSpawnerCfg
        ), f"Object '{self.name}' must retain its MultiAssetSpawnerCfg; construct a new Object to change its variants"
        assert len(spawn_cfg.assets_cfg) == len(
            self.variants
        ), f"Object '{self.name}' variant count changed; construct a new Object to change its variants"
        return spawn_cfg.assets_cfg

    def _get_configured_variants(self) -> list[ObjectVariant]:
        """Keep per-variant geometry aligned with the configured native spawners."""
        configured_variants = []
        for variant_index, spawn_cfg in enumerate(self._get_variant_spawn_configs()):
            variant = self._configured_variants[variant_index]
            if variant is None or variant.spawn_cfg.to_dict() != spawn_cfg.to_dict():
                variant = ObjectVariant(
                    spawn_cfg=spawn_cfg,
                    object_type=self.object_type,
                    bounding_box=self._bounding_box_overrides[variant_index],
                )
                self._configured_variants[variant_index] = variant
            configured_variants.append(variant)
        return configured_variants

    def get_object_cfg(self) -> tuple[str, AssetBaseCfg]:
        """Return the scene configuration after checking the declared variant count."""
        if self.has_variants:
            self._get_variant_spawn_configs()
        return super().get_object_cfg()

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return bounds for a single-variant object in its root frame."""
        assert not self.has_variants, f"Object '{self.name}' requires per-environment bounding boxes"
        return self.bounding_box if self.bounding_box is not None else self._get_single_variant().get_bounding_box()

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return singleton collision geometry in the same frame as placement bounds."""
        if self.has_variants:
            assert not excluded_prim_paths, "Prim exclusions require a concrete object variant"
            return None
        return self._get_single_variant().get_collision_mesh(excluded_prim_paths)

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return the bounds of each environment's assigned variant."""
        if not self.has_variants:
            return super().get_bounding_box_per_env(num_envs)
        indices = self.variant_indices_by_env
        assert (
            indices is not None and len(indices) == num_envs
        ), f"Object '{self.name}' needs a scene variant assignment for {num_envs} environments before placement"
        bounds = [variant.get_bounding_box() for variant in self._get_configured_variants()]
        return AxisAlignedBoundingBox(
            min_point=torch.stack([bounds[index].min_point[0] for index in indices]),
            max_point=torch.stack([bounds[index].max_point[0] for index in indices]),
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
        """Return the body path shared by this object's prepared spawn configurations."""
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        spawn_cfg = self.object_cfg.spawn
        spawn_cfgs = spawn_cfg.assets_cfg if isinstance(spawn_cfg, MultiAssetSpawnerCfg) else [spawn_cfg]
        paths = {
            ObjectVariant(spawn_cfg=cfg, object_type=self.object_type).get_contact_body_path() for cfg in spawn_cfgs
        }
        assert len(paths) == 1, f"Object '{self.name}' has incompatible prepared contact-body paths"
        return self.prim_path + paths.pop()

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
            target_spawn_cfg = contact_against_object.object_cfg.spawn
            assert isinstance(target_spawn_cfg, UsdFileCfg), "BASE contact targets require a USD spawn configuration"
            target_variants = target_spawn_cfg.variants
            if target_variants is not None and not isinstance(target_variants, dict):
                target_variants = target_variants.to_dict()
            body_paths = read_asset_rigid_body_paths(
                target_spawn_cfg.usd_path,
                variants=target_variants,
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

    def _get_single_source_spawn_cfg(self) -> SpawnerCfg:
        """Copy current singleton source settings when subclasses rebuild their configuration."""
        if self.spawner_cfg is not None:
            assert not self.spawn_cfg_addon, "spawn_cfg_addon cannot be combined with spawner_cfg"
            return deepcopy(self.spawner_cfg)
        if self.usd_path is not None:
            return make_usd_spawn_cfg_with_addons(
                UsdFileCfg(usd_path=self.usd_path, scale=self.scale), self.spawn_cfg_addon
            )
        return deepcopy(self.variants[0].spawn_cfg)

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False) -> SpawnerCfg:
        """Build native spawners, caching hierarchy preparation only for multiple variants."""
        if self.has_variants:
            if self._prepared_spawn_cfgs is None:
                self._prepared_spawn_cfgs = prepare_rigid_object_variants(self.variants)
            spawn_cfgs = deepcopy(self._prepared_spawn_cfgs)
        else:
            spawn_cfgs = [self._get_single_source_spawn_cfg()]
        for spawn_cfg in spawn_cfgs:
            if activate_contact_sensors and hasattr(spawn_cfg, "activate_contact_sensors"):
                spawn_cfg.activate_contact_sensors = True
        if self.has_variants:
            return MultiAssetSpawnerCfg(
                assets_cfg=spawn_cfgs,
                random_choice=False,
                activate_contact_sensors=activate_contact_sensors,
            )
        return spawn_cfgs[0]

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
        if self.usd_path is not None:
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
