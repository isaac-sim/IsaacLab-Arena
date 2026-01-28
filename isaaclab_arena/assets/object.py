# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose

from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.relations.relations import Relation, RelationBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PoseRange
from isaaclab_arena.utils.usd_helpers import compute_bounding_box_from_usd, has_light, open_stage


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
        **kwargs,
    ):
        # Pull out addons (and remove them from kwargs before passing to super)
        spawn_cfg_addon: dict[str, Any] = kwargs.pop("spawn_cfg_addon", {}) or {}
        asset_cfg_addon: dict[str, Any] = kwargs.pop("asset_cfg_addon", {}) or {}
        if object_type is not ObjectType.SPAWNER:
            assert usd_path is not None
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
        self.relations = []

    def add_relation(self, relation: RelationBase) -> None:
        self.relations.append(relation)

    def get_relations(self) -> list[RelationBase]:
        return self.relations

    def get_spatial_relations(self) -> list[Relation]:
        """Get only spatial relations (On, NextTo, etc.), excluding markers like IsAnchor."""
        return [r for r in self.relations if isinstance(r, Relation)]

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get bounding box in world coordinates (local bbox + position offset)."""
        local_bbox = self.get_bounding_box()
        pos = self.initial_pose.position_xyz if self.initial_pose else (0, 0, 0)
        return AxisAlignedBoundingBox(
            min_point=(
                local_bbox.min_point[0] + pos[0],
                local_bbox.min_point[1] + pos[1],
                local_bbox.min_point[2] + pos[2],
            ),
            max_point=(
                local_bbox.max_point[0] + pos[0],
                local_bbox.max_point[1] + pos[1],
                local_bbox.max_point[2] + pos[2],
            ),
        )

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box.get_corners_at(pos)

    def set_initial_pose(self, pose: Pose | PoseRange) -> None:
        """Set the initial pose of the object.

        Args:
            pose: The pose to set. Can be a single pose or a pose range.
                  In the case of a PoseRange, the object will be reset
                  to a random pose within the range on environment reset.
        """
        self.initial_pose = pose
        self.object_cfg = self._add_initial_pose_to_cfg(self.object_cfg)
        self.event_cfg = self._update_initial_pose_event_cfg(self.event_cfg)

    def get_initial_pose(self) -> Pose | PoseRange | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self.event_cfg = self._update_initial_pose_event_cfg(self.event_cfg)

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self.event_cfg = self._update_initial_pose_event_cfg(self.event_cfg)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
                **self.spawn_cfg_addon,
            ),
            **self.asset_cfg_addon,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
                **self.spawn_cfg_addon,
            ),
            **self.asset_cfg_addon,
            actuators={},
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_base_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.BASE
        with open_stage(self.usd_path) as stage:
            if has_light(stage):
                print("WARNING: Base object has lights, this may cause issues when using with multiple environments.")
        object_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                **self.spawn_cfg_addon,
            ),
            **self.asset_cfg_addon,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_spawner_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.SPAWNER
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=self.spawner_cfg,
            **self.asset_cfg_addon,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _add_initial_pose_to_cfg(
        self, object_cfg: RigidObjectCfg | ArticulationCfg | AssetBaseCfg
    ) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg:
        # Optionally specify initial pose
        if self.initial_pose is not None:
            if isinstance(self.initial_pose, Pose):
                initial_pose = self.initial_pose
            elif isinstance(self.initial_pose, PoseRange):
                initial_pose = self.initial_pose.get_midpoint()
            object_cfg.init_state.pos = initial_pose.position_xyz
            object_cfg.init_state.rot = initial_pose.rotation_wxyz
        return object_cfg

    def _requires_reset_pose_event(self) -> bool:
        return (
            self.initial_pose is not None
            and self.reset_pose
            and self.object_type in [ObjectType.RIGID, ObjectType.ARTICULATION]
        )

    def _init_event_cfg(self) -> EventTermCfg | None:
        if self._requires_reset_pose_event():
            # Two possible event types:
            # - initial pose is a Pose - reset to a single pose
            # - initial pose is a PoseRange - reset to a random pose within the range
            if isinstance(self.initial_pose, Pose):
                return EventTermCfg(
                    func=set_object_pose,
                    mode="reset",
                    params={
                        "pose": self.initial_pose,
                        "asset_cfg": SceneEntityCfg(self.name),
                    },
                )
            elif isinstance(self.initial_pose, PoseRange):
                return EventTermCfg(
                    func=randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": self.initial_pose.to_dict(),
                        "asset_cfgs": [SceneEntityCfg(self.name)],
                    },
                )
            else:
                raise ValueError(f"Initial pose {self.initial_pose} is not a Pose or PoseRange")
        else:
            return None

    def _needs_reinit_of_event_cfg(self):
        # If there is no event cfg, needs to be reinitialized
        if self.event_cfg is None:
            return True
        # Here we check if the event cfg is for the correct pose type.
        # If not, needs to be reinitialized.
        if (isinstance(self.initial_pose, Pose) and ("pose" not in self.event_cfg.params)) or (
            isinstance(self.initial_pose, PoseRange) and ("pose_range" not in self.event_cfg.params)
        ):
            return True
        return False

    def _update_initial_pose_event_cfg(self, event_cfg: EventTermCfg | None) -> EventTermCfg | None:
        if self._requires_reset_pose_event():
            # Create an event cfg if one does not yet exist
            if self._needs_reinit_of_event_cfg():
                event_cfg = self._init_event_cfg()
            if isinstance(self.initial_pose, Pose):
                event_cfg.params["pose"] = self.initial_pose
            elif isinstance(self.initial_pose, PoseRange):
                event_cfg.params["pose_range"] = self.initial_pose.to_dict()
            else:
                raise ValueError(f"Initial pose {self.initial_pose} is not a Pose or PoseRange")
        else:
            event_cfg = None
        return event_cfg
