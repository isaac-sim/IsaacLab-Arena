# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd, has_light, open_stage


class Object(RootedObjectBase):
    """Pick-up object config for a pick-and-place environment."""

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
        self.spawn_cfg_addon = spawn_cfg_addon
        self.asset_cfg_addon = asset_cfg_addon
        self.bounding_box = None
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_local_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_local_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box.get_corners_at(pos)

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
        assert self.usd_path is not None, f"No USD path available for {self.name}. Can't add contact sensor."
        return self._get_contact_sensor_prim_path_from_usd(self.usd_path)

    def _get_contact_sensor_prim_path_from_usd(self, usd_path: str) -> str:
        """Return the contact-sensor prim path for the rigid body in a USD."""
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        rigid_body_relative_path = find_shallowest_rigid_body(
            usd_path,
            relative_to_root=True,
            variants=(self.spawn_cfg_addon or {}).get("variants"),
        )
        assert (
            rigid_body_relative_path is not None
        ), f"No rigid body found in {self.name} USD file: {usd_path}. Can't add contact sensor."
        return self.prim_path + rigid_body_relative_path

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        # We override this function from the parent class because some assets do not have their rigid body
        # at the root of the USD file. To be robust to this, we find the shallowest rigid body and add the
        # contact sensor to it.
        contact_sensor_prim_path = self.get_contact_sensor_prim_path()
        if isinstance(contact_against_object, Object):
            # Handles Object and its subclasses, including RigidObjectSet.
            # RigidObjectSet normalizes the USD paths for all members before spawning, so they have the same
            # relative structure and rigid-body name. We add the contact sensor to the normalized rigid body beneath the its scene prim.
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

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Return the spawn config to use: custom spawner_cfg if set, else a UsdFileCfg."""
        if self.spawner_cfg is not None:
            return self.spawner_cfg
        return UsdFileCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
            **self.spawn_cfg_addon,
        )

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
