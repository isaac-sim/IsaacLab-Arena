# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.relations import RelationBase
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
        relations: list[RelationBase] | None = None,
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
            relations: Placement relations shared across environments.
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
        self._initialize_object(name, prim_path, ObjectType.RIGID, spawn_cfg, initial_pose, relations, **kwargs)
        self.random_choice = random_choice
