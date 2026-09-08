# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.shapes import CableCfg

from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.terms.events import reset_cable_to_default
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange
from isaaclab_arena.utils.velocity import Velocity


class Cable(ObjectBase):
    """A procedurally spawned Isaac Lab cable managed as an Arena scene asset.

    Cable simulation requires the Newton physics backend. Build environments containing this asset
    with a VBD solver. Relation-based placement and rigid-root operations are not yet supported.
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        spawn: CableCfg,
        initial_pose: Pose | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Configure a cable scene asset."""

        assert prim_path, "Cable prim_path must be non-empty."
        super().__init__(name=name, prim_path=prim_path, object_type=ObjectType.CABLE, tags=tags)
        self.spawn = spawn
        self.object_cfg = self._init_object_cfg()
        if initial_pose is not None:
            self.set_initial_pose(initial_pose, create_reset_event=False)
        self._pose_event_cfg = self._build_reset_event()

    def _init_object_cfg(self) -> CableObjectCfg:
        """Create the Isaac Lab cable configuration."""
        return CableObjectCfg(prim_path=self.prim_path, spawn=self.spawn)

    def _set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Set a fixed cable construction pose."""
        if not isinstance(pose, Pose):
            raise NotImplementedError(
                "Cable only supports a fixed initial Pose; ranged and per-environment poses are not supported."
            )
        super()._set_initial_pose(pose)

    def _build_reset_event(self) -> EventTermCfg:
        """Build the event that restores the cable's default segment state."""
        return EventTermCfg(
            func=reset_cable_to_default,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(self.name)},
        )

    def add_relation(self, relation: RelationBase) -> None:
        """Reject relation-based placement until cable segment placement is supported."""
        raise NotImplementedError("Cable does not yet support relation-based placement.")

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Reject bounding-box queries used by relation-based placement."""
        raise NotImplementedError("Cable bounding boxes are not yet supported by the object placer.")

    def layout_pose_to_scene_writes(self, layout_pose: Pose) -> list[tuple[str, Pose]]:
        """Reject root-pose writes produced by relation-based placement."""
        raise NotImplementedError("Cable placement requires segment-state writes, which are not yet supported.")

    def set_initial_velocity(self, velocity: Velocity) -> None:
        """Reject rigid-root initial velocity configuration."""
        raise NotImplementedError("Cable does not support a rigid-root initial velocity.")

    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Reject rigid-root runtime pose writes."""
        raise NotImplementedError("Cable runtime poses must be written per segment.")

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        """Reject rigid-object contact sensor configuration."""
        raise NotImplementedError("Cable does not support rigid-object contact sensors.")

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        raise NotImplementedError("Cable cannot generate a rigid-object configuration.")

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        raise NotImplementedError("Cable cannot generate an articulation configuration.")

    def _generate_base_cfg(self) -> AssetBaseCfg:
        raise NotImplementedError("Cable uses CableObjectCfg rather than a base-asset configuration.")
