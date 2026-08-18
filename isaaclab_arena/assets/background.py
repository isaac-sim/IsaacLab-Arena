# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_prim_tree import UsdPhysicsRootRecord, load_usd_physics_roots


class Background(Object):
    """
    Encapsulates the background scene for a environment.
    """

    def __init__(
        self,
        name: str,
        usd_path: str,
        object_min_z: float,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        reset_nested_physics: bool = False,
        **kwargs,
    ):
        self.reset_nested_physics = reset_nested_physics
        super().__init__(
            name=name,
            usd_path=usd_path,
            initial_pose=initial_pose,
            prim_path=prim_path,
            # Backgrounds don't have physics (at the moment)
            object_type=ObjectType.BASE,
            **kwargs,
        )
        # We use this to define reset terms for when objects are dropped.
        # NOTE(alexmillane, 2025.09.19): This is a global z height. If you shift the
        # background, by using initial_pose, this height doesn't shift with it.
        # TODO(alexmillane, 2025.09.19): Make this value relative to the background
        # prim origin.
        self.object_min_z = object_min_z
        self._nested_physics_records: list[UsdPhysicsRootRecord] | None = None

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Return a USD spawner that materializes nested instance-proxy physics."""
        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        if self.reset_nested_physics:
            from isaaclab_arena.assets.background_spawner import spawn_from_usd_with_resettable_nested_physics

            cfg.func = spawn_from_usd_with_resettable_nested_physics
        return cfg

    def get_nested_physics_cfgs(
        self,
        claimed_prim_paths: dict[str, ObjectType] | None = None,
    ) -> dict[str, ArticulationCfg | RigidObjectCfg]:
        """Return non-spawning configs for unclaimed nested physics roots.

        Args:
            claimed_prim_paths: Runtime paths already represented by explicit object references.

        Returns:
            Hidden scene configs keyed by deterministic private names.
        """
        if not self.reset_nested_physics:
            return {}
        if self._nested_physics_records is None:
            self._nested_physics_records = load_usd_physics_roots(self.usd_path)

        claimed_prim_paths = claimed_prim_paths or {}
        configs: dict[str, ArticulationCfg | RigidObjectCfg] = {}
        background_slug = re.sub(r"\W+", "_", self.name).strip("_")
        for index, record in enumerate(self._nested_physics_records):
            prim_path = f"{self.prim_path}/{record.relative_path}"
            is_claimed = prim_path in claimed_prim_paths or any(
                claimed_type == ObjectType.ARTICULATION and prim_path.startswith(f"{claimed_path}/")
                for claimed_path, claimed_type in claimed_prim_paths.items()
            )
            if is_claimed:
                continue
            path_slug = re.sub(r"\W+", "_", record.relative_path).strip("_")
            entity_name = f"_background_physics_{background_slug}_{index}_{path_slug}"
            assert entity_name not in configs, f"Duplicate hidden background entity name: {entity_name}"
            if record.object_type == ObjectType.ARTICULATION:
                configs[entity_name] = ArticulationCfg(
                    prim_path=prim_path,
                    actuators={},
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={}),
                )
            else:
                configs[entity_name] = RigidObjectCfg(prim_path=prim_path)
        return configs

    def get_viewer_cfg(self) -> ViewerCfg | None:
        """Return a custom viewer camera framing for this background, or None to auto-frame."""
