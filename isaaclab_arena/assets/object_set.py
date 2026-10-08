# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import trimesh
from collections.abc import Collection

from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_geometry import ObjectGeometry
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants


class RigidObjectSet(Object):
    """One rigid object per environment, selected from native asset configurations."""

    def __init__(
        self,
        name: str,
        objects: list[Object],
        prim_path: str | None = None,
        random_choice: bool = False,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        """Copy member spawn settings into one native scene object.

        Args:
            name: Scene object name shared by all alternatives.
            objects: Rigid objects supplying independent native spawn configurations.
                Member poses and relations belong to the members and are not copied.
            prim_path: Environment-scoped path shared by all alternatives.
            random_choice: Sample alternatives independently; otherwise repeat member order.
            initial_pose: Initial pose shared by all alternatives.
        """
        assert objects, f"Object set '{name}' requires at least one member."
        for obj in objects:
            assert (
                isinstance(obj, Object) and obj.object_type == ObjectType.RIGID
            ), f"Object set '{name}' accepts rigid Object members only."
            assert not isinstance(
                obj.spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), f"Object set '{name}' requires one native spawn configuration per member."
            assert (
                obj.bounding_box is None
            ), f"Object set '{name}' derives bounds from native geometry; member '{obj.name}' has a bounds override."
        spawn_configs = prepare_rigid_object_variants([obj.spawn_cfg for obj in objects])
        spawn_cfg = spawn_configs[0]
        if len(spawn_configs) > 1:
            spawn_cfg = MultiAssetSpawnerCfg(
                assets_cfg=spawn_configs,
                random_choice=False,
                # None keeps each member's native setting; the wrapper's False default replaces it.
                activate_contact_sensors=None,
            )
        self.random_choice = random_choice
        self._variant_indices_by_env: tuple[int, ...] | None = None
        self._variant_geometry: dict[int, ObjectGeometry] = {}
        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawn_cfg,
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )

    @property
    def has_variants(self) -> bool:
        """Whether this object can have different geometry across environments."""
        return isinstance(self.spawn_cfg, MultiAssetSpawnerCfg) and len(self.spawn_cfg.assets_cfg) > 1

    @property
    def variant_indices_by_env(self) -> tuple[int, ...] | None:
        """The fixed environment-to-variant assignment established before placement."""
        return self._variant_indices_by_env

    def bind_variant_assignment(self, indices: tuple[int, ...]) -> None:
        """Bind one valid variant index per environment without allowing reassignment."""
        indices = tuple(indices)
        variant_count = len(self._get_variant_spawn_configs())
        assert indices and all(
            type(index) is int and 0 <= index < variant_count for index in indices
        ), f"Object set '{self.name}' has invalid variant indices."
        assert self._variant_indices_by_env in (
            None,
            indices,
        ), f"Object set '{self.name}' already has a different variant assignment; construct a new set for a new scene."
        self._variant_indices_by_env = indices

    def _get_variant_spawn_configs(self) -> list[SpawnerCfg]:
        """Read the current alternatives from the authoritative native spawn configuration."""
        if isinstance(self.spawn_cfg, MultiAssetSpawnerCfg):
            assert self.spawn_cfg.assets_cfg, f"Object set '{self.name}' requires at least one native variant."
            return self.spawn_cfg.assets_cfg
        return [self.spawn_cfg]

    def _get_variant_geometry(self, variant_index: int) -> ObjectGeometry:
        """Refresh a member's derived geometry after its native configuration changes."""
        spawn_cfg = self._get_variant_spawn_configs()[variant_index]
        geometry = self._variant_geometry.get(variant_index)
        if geometry is None or not geometry.matches(spawn_cfg):
            geometry = ObjectGeometry(spawn_cfg, ObjectType.RIGID)
            self._variant_geometry[variant_index] = geometry
        return geometry

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return singleton bounds; heterogeneous sets require an environment selection."""
        assert not self.has_variants, f"Object set '{self.name}' requires per-environment bounding boxes."
        return super().get_bounding_box()

    def get_bounding_box_for_env(self, env_id: int) -> AxisAlignedBoundingBox:
        """Return the assigned member's local bounds for one environment."""
        assert env_id >= 0, "Environment index must be non-negative."
        if not self.has_variants:
            return self.get_bounding_box()
        indices = self.variant_indices_by_env
        assert indices is not None, f"Object set '{self.name}' needs a variant assignment before geometry queries."
        assert env_id < len(indices), f"Object set '{self.name}' has no assignment for environment {env_id}."
        return self._get_variant_geometry(indices[env_id]).get_bounding_box()

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return assigned local bounds with one row per environment."""
        assert num_envs > 0, "Per-environment bounds require at least one environment."
        if not self.has_variants:
            bounds = self.get_bounding_box()
            return AxisAlignedBoundingBox(bounds.min_point.expand(num_envs, 3), bounds.max_point.expand(num_envs, 3))
        indices = self.variant_indices_by_env
        assert (
            indices is not None and len(indices) == num_envs
        ), f"Object set '{self.name}' needs a variant assignment for {num_envs} environments before geometry queries."
        bounds = [
            self._get_variant_geometry(index).get_bounding_box()
            for index in range(len(self._get_variant_spawn_configs()))
        ]
        return AxisAlignedBoundingBox(
            min_point=torch.stack([bounds[index].min_point[0] for index in indices]),
            max_point=torch.stack([bounds[index].max_point[0] for index in indices]),
        )

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return singleton geometry; heterogeneous sets have no shared collision mesh."""
        if self.has_variants:
            assert not excluded_prim_paths, "Object set exclusions require a concrete member."
            return None
        return super().get_collision_mesh(excluded_prim_paths)

    def get_contact_sensor_prim_path(self) -> str:
        """Return the rigid-body path shared by all native alternatives."""
        if not self.has_variants:
            return super().get_contact_sensor_prim_path()
        body_paths = {
            self._get_variant_geometry(index).get_contact_body_path()
            for index in range(len(self._get_variant_spawn_configs()))
        }
        assert len(body_paths) == 1, f"Object set '{self.name}' has incompatible rigid-body paths."
        return self.prim_path + body_paths.pop()
