# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import trimesh
from collections.abc import Collection, Sequence
from typing import Literal

from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_geometry import ObjectGeometry
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants


class ObjectChoice(Object):
    """A scene object that selects one rigid asset for each environment."""

    def __init__(
        self,
        name: str,
        objects: Sequence[Object | SpawnerCfg],
        assign_to_environments: Literal["sequential", "random"] = "sequential",
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        **kwargs,
    ):
        """Copy asset settings and let the builder assign one choice before placement.

        Args:
            name: Scene name shared across environments.
            objects: Concrete rigid objects or native spawn configurations. Member names,
                poses, and relations are not copied; configure those on the choice.
            assign_to_environments: Cycle through objects in order ("sequential") or
                sample independently ("random"). Assignments remain fixed across resets.
            prim_path: Scene prim path; defaults to the environment namespace and name.
            initial_pose: Initial pose of the selected object in each environment.
            relations: Placement relations shared by all choices.
            **kwargs: Asset configuration and base-class options.
        """
        assert objects, "ObjectChoice requires at least one object"
        assert assign_to_environments in (
            "sequential",
            "random",
        ), "assign_to_environments must be 'sequential' or 'random'"
        spawn_configs = []
        for obj in objects:
            assert not isinstance(obj, ObjectChoice), "ObjectChoice cannot contain nested choices"
            if isinstance(obj, Object):
                assert obj.object_type == ObjectType.RIGID, "ObjectChoice supports rigid objects only"
                assert obj.bounding_box is None, "ObjectChoice cannot copy a bounds override"
                spawn_cfg = obj.spawn_cfg
            else:
                spawn_cfg = obj
            assert isinstance(
                spawn_cfg, SpawnerCfg
            ), "ObjectChoice requires rigid Object instances or native spawn configurations"
            assert not isinstance(
                spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), "ObjectChoice requires concrete assets; nested multi-spawners are not supported"
            spawn_configs.append(spawn_cfg)
        spawn_configs = prepare_rigid_object_variants(spawn_configs)
        spawn_cfg = spawn_configs[0]
        if len(spawn_configs) > 1:
            spawn_cfg = MultiAssetSpawnerCfg(
                assets_cfg=spawn_configs,
                # None preserves each member's contact-reporting setting.
                activate_contact_sensors=None,
            )
        self._initialize_object(name, prim_path, ObjectType.RIGID, spawn_cfg, initial_pose, relations, **kwargs)
        self.assign_to_environments = assign_to_environments
        self._variant_indices_by_env: tuple[int, ...] | None = None
        self._variant_geometry: dict[int, ObjectGeometry] = {}

    @Object.spawn_cfg.setter
    def spawn_cfg(self, value: SpawnerCfg) -> None:
        Object.spawn_cfg.fset(self, value)
        self._variant_geometry.clear()

    @property
    def has_variants(self) -> bool:
        """Whether more than one concrete asset is available across environments."""
        return len(self._get_variant_spawn_configs()) > 1

    @property
    def variant_indices_by_env(self) -> tuple[int, ...] | None:
        """The fixed environment-to-asset assignment established before placement."""
        return self._variant_indices_by_env

    def bind_variant_assignment(self, indices: tuple[int, ...]) -> None:
        """Bind one asset per environment without allowing reassignment."""
        indices = tuple(indices)
        variant_count = len(self._get_variant_spawn_configs())
        assert indices and all(
            type(index) is int and 0 <= index < variant_count for index in indices
        ), f"ObjectChoice '{self.name}' has invalid variant indices."
        assert self._variant_indices_by_env in (
            None,
            indices,
        ), (
            f"ObjectChoice '{self.name}' already has a different variant assignment; construct a new choice for a new"
            " scene."
        )
        self._variant_indices_by_env = indices

    def _get_variant_spawn_configs(self) -> list[SpawnerCfg]:
        """Read alternatives from the native spawn configuration."""
        assert not isinstance(self.spawn_cfg, MultiUsdFileCfg), "Use ObjectChoice with concrete asset configurations"
        if isinstance(self.spawn_cfg, MultiAssetSpawnerCfg):
            assert self.spawn_cfg.assets_cfg, f"ObjectChoice '{self.name}' requires at least one asset"
            return self.spawn_cfg.assets_cfg
        return [self.spawn_cfg]

    def _get_geometry(self, variant_index: int = 0) -> ObjectGeometry:
        """Refresh one alternative's geometry after native configuration changes."""
        spawn_cfg = self._get_variant_spawn_configs()[variant_index]
        geometry = self._variant_geometry.get(variant_index)
        if geometry is None or not geometry.matches(spawn_cfg):
            geometry = ObjectGeometry(spawn_cfg, ObjectType.RIGID)
            self._variant_geometry[variant_index] = geometry
        return geometry

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return local bounds when the choice contains only one asset."""
        assert not self.has_variants, f"ObjectChoice '{self.name}' requires per-environment bounding boxes"
        return super().get_bounding_box()

    def get_bounding_box_for_env(self, env_id: int) -> AxisAlignedBoundingBox:
        """Return the assigned asset's local bounds for one environment."""
        assert env_id >= 0, "Environment index must be non-negative"
        if not self.has_variants:
            return self.get_bounding_box()
        indices = self.variant_indices_by_env
        assert indices is not None, f"ObjectChoice '{self.name}' needs a variant assignment before geometry queries"
        assert env_id < len(indices), f"ObjectChoice '{self.name}' has no assignment for environment {env_id}"
        return self._get_geometry(indices[env_id]).get_bounding_box()

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return assigned local bounds with one row per environment."""
        assert num_envs > 0, "Per-environment bounds require at least one environment"
        if not self.has_variants:
            return super().get_bounding_box_per_env(num_envs)
        indices = self.variant_indices_by_env
        assert (
            indices is not None and len(indices) == num_envs
        ), f"ObjectChoice '{self.name}' needs a variant assignment for {num_envs} environments before geometry queries"
        bounds = [
            self._get_geometry(index).get_bounding_box() for index in range(len(self._get_variant_spawn_configs()))
        ]
        return AxisAlignedBoundingBox(
            min_point=torch.stack([bounds[index].min_point[0] for index in indices]),
            max_point=torch.stack([bounds[index].max_point[0] for index in indices]),
        )

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return a mesh for a single choice; heterogeneous choices use per-environment bounds."""
        if self.has_variants:
            assert not excluded_prim_paths, "Prim exclusions require a concrete object choice"
            return None
        return super().get_collision_mesh(excluded_prim_paths)

    def get_contact_sensor_prim_path(self) -> str:
        """Return the rigid-body path shared by every alternative."""
        body_paths = {
            self._get_geometry(index).get_contact_body_path() for index in range(len(self._get_variant_spawn_configs()))
        }
        assert len(body_paths) == 1, f"ObjectChoice '{self.name}' has incompatible rigid-body paths"
        return self.prim_path + body_paths.pop()
