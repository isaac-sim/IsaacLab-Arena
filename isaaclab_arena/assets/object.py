# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd, has_light, open_stage


class Object(ObjectBase):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType | None = None,
        usd_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_pose: Pose | None = None,
        relations: list[RelationBase] = [],
        spawner_cfg=None,
        **kwargs,
    ):
        # Pull out addons (and remove them from kwargs before passing to super)
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        self.spawner_cfg = spawner_cfg
        if usd_path is None and self.spawner_cfg is None:
            raise ValueError(f"Object '{name}' requires either usd_path or a spawner_cfg")
        # Detect object type if not provided
        if object_type is None:
            object_type = detect_object_type(usd_path=usd_path)
        super().__init__(name=name, prim_path=prim_path, object_type=object_type, **kwargs)
        self.usd_path = usd_path
        self.scale = scale
        self.initial_pose = initial_pose
        self.reset_pose = True
        self.spawn_cfg_addon = spawn_cfg_addon
        self.asset_cfg_addon = asset_cfg_addon
        self.bounding_box = None
        self.object_cfg = self._init_object_cfg()
        self.event_cfg = self._init_event_cfg()

    def add_relation(self, relation: RelationBase) -> None:
        """Add a relation to this object."""
        self.relations.append(relation)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_local_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get bounding box in world coordinates (local bbox rotated and translated).

        Only 90° rotations around Z axis are supported. An assertion error is raised
        for any other rotation. If initial_pose is a PoseRange (not a fixed Pose),
        returns the local bounding box without transformation.
        """
        local_bbox = self.get_bounding_box()
        if self.initial_pose is None or not isinstance(self.initial_pose, Pose):
            return local_bbox
        quarters = quaternion_to_90_deg_z_quarters(self.initial_pose.rotation_xyzw)
        return local_bbox.rotated_90_around_z(quarters).translated(self.initial_pose.position_xyz)

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_local_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box.get_corners_at(pos)

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self.event_cfg = self._init_event_cfg()

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self.event_cfg = self._init_event_cfg()

    def get_contact_sensor_cfg(
        self, contact_against_object: ObjectBase | None = None, usd_path: str | None = None
    ) -> ContactSensorCfg:
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        # We override this function from the parent class because in some assets, the rigid body
        # is not at the root of the USD file. To be robust to this, we find the shallowest rigid body
        # and add the contact sensor to it.
        # TODO(alexmillane, 2026.01.29): This capability to search for the correct place
        # to add the contact sensor is not yet supported for ObjectReferences and RigidObjectSet.
        # For these objects we just (try to) add the contact sensor to the root prim.
        usd_path = usd_path or self.usd_path
        rigid_body_relative_path = find_shallowest_rigid_body(usd_path, relative_to_root=True)
        assert (
            rigid_body_relative_path is not None
        ), f"No rigid body found in {self.name} USD file: {usd_path}. Can't add contact sensor."
        contact_sensor_prim_path = self.prim_path + rigid_body_relative_path
        # There are also cases where the contact against object does not have its rigid body at the root.
        # In that case, we also need to find the shallowest rigid body.
        # NOTE(alexmillane, 2026.04.10): For now we only support this for Object, but in the future we
        # could support this for ObjectReference and RigidObjectSet. For now, those object types
        # are assumed to have their rigid body at the their prim path.
        if isinstance(contact_against_object, Object):
            assert (
                contact_against_object.object_type == ObjectType.RIGID
            ), "Contact sensor is only supported for rigid objects"
            contact_against_relative_path = find_shallowest_rigid_body(
                contact_against_object.usd_path, relative_to_root=True
            )
            assert contact_against_relative_path is not None, (
                f"No rigid body found in {contact_against_object.name} USD file: {contact_against_object.usd_path}."
                " Can't add contact sensor."
            )
            filter_prim_paths = [contact_against_object.get_prim_path() + contact_against_relative_path]
        elif isinstance(contact_against_object, ObjectBase):
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
            prim_path="{ENV_REGEX_NS}/" + self.name,
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
