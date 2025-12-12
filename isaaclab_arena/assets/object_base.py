# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaaclab_arena.assets.asset import Asset

if TYPE_CHECKING:
    from isaaclab_arena.utils.pose import Pose


class ObjectType(Enum):
    BASE = "base"
    RIGID = "rigid"
    ARTICULATION = "articulation"
    SPAWNER = "spawner"


class ObjectBase(Asset, ABC):
    """Parent class for (spawnable) Object and ObjectReference."""

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType = ObjectType.BASE,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if prim_path is None:
            prim_path = "{ENV_REGEX_NS}/" + self.name
        self.prim_path = prim_path
        self.object_type = object_type

    def set_prim_path(self, prim_path: str) -> None:
        self.prim_path = prim_path

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_object_cfg(self) -> dict[str, Any]:
        return {self.name: self.object_cfg}

    def _init_object_cfg(self) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg:
        if self.object_type == ObjectType.RIGID:
            object_cfg = self._generate_rigid_cfg()
        elif self.object_type == ObjectType.ARTICULATION:
            object_cfg = self._generate_articulation_cfg()
        elif self.object_type == ObjectType.BASE:
            object_cfg = self._generate_base_cfg()
        elif self.object_type == ObjectType.SPAWNER:
            object_cfg = self._generate_spawner_cfg()
        else:
            raise ValueError(f"Invalid object type: {self.object_type}")
        return object_cfg

    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Get the pose of the object in the environment.

        Args:
            env: The environment.
            is_relative: Whether to return the pose in the relative frame of the environment.

        Returns:
            The pose of the object in each environment. The shape is (num_envs, 7).
            The order is (x, y, z, qw, qx, qy, qz).
        """
        # We require that the asset has been added to the scene under its name.
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        if (self.object_type == ObjectType.RIGID) or (self.object_type == ObjectType.ARTICULATION):
            object_pose = env.scene[self.name].data.root_pose_w.clone()
        elif self.object_type == ObjectType.BASE:
            object_pose = torch.cat(env.scene[self.name].get_world_poses(), dim=-1)
        else:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")
        if is_relative:
            object_pose[:, :3] -= env.scene.env_origins
        return object_pose

    def get_contact_sensor_cfg(self, contact_against_prim_paths: list[str] | None = None) -> ContactSensorCfg:
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        if contact_against_prim_paths is None:
            contact_against_prim_paths = []
        return ContactSensorCfg(
            prim_path=self.prim_path,
            filter_prim_paths_expr=contact_against_prim_paths,
        )

    @abstractmethod
    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        # Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_articulation_cfg(self) -> ArticulationCfg:
        # Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_base_cfg(self) -> AssetBaseCfg:
        # Subclasses must implement this method
        pass

    def _generate_spawner_cfg(self) -> AssetBaseCfg:
        # Object Subclasses must implement this method
        pass

    # Spatial Relationship Methods
    def on_top_of(
        self,
        target: "ObjectBase",
        clearance: float = 0.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
    ) -> "ObjectBase":
        """
        Place this object on top of a target object.

        This method automatically computes the appropriate pose to place this object
        on top of the target object, accounting for both objects' geometries.

        Args:
            target: The target object to place this object on top of.
            clearance: Additional vertical clearance between objects (default: 0.0).
            x_offset: Horizontal offset in x direction from center (default: 0.0).
            y_offset: Horizontal offset in y direction from center (default: 0.0).

        Returns:
            Self, to allow method chaining.

        Example:
            ```python
            table = asset_registry.get_asset_by_name("table")()
            box = asset_registry.get_asset_by_name("cracker_box")()
            box.on_top_of(table)
            scene = Scene(assets=[table, box])
            ```
        """
        from isaaclab_arena.utils.spatial_relationships import compute_bounding_box_from_usd, compute_on_top_of_pose

        # Get bounding boxes for both objects
        # We need to access the usd_path and scale from the concrete Object class
        # For now, we'll use getattr to handle this polymorphically
        object_usd_path = getattr(self, "usd_path", None)
        object_scale = getattr(self, "scale", (1.0, 1.0, 1.0))
        target_usd_path = getattr(target, "usd_path", None)
        target_scale = getattr(target, "scale", (1.0, 1.0, 1.0))

        if object_usd_path is None:
            raise ValueError(f"Object {self.name} does not have a usd_path attribute")
        if target_usd_path is None:
            raise ValueError(f"Target object {target.name} does not have a usd_path attribute")

        # Get the target's current pose (if set)
        target_pose = getattr(target, "initial_pose", None)

        # Compute bounding boxes
        object_bbox = compute_bounding_box_from_usd(object_usd_path, scale=object_scale)
        target_bbox = compute_bounding_box_from_usd(target_usd_path, scale=target_scale, pose=target_pose)

        # Compute the placement pose
        placement_pose = compute_on_top_of_pose(
            object_bbox=object_bbox,
            target_bbox=target_bbox,
            clearance=clearance,
            x_offset=x_offset,
            y_offset=y_offset,
        )

        # Set the initial pose on this object
        self.set_initial_pose(placement_pose)

    def next_to(
        self,
        target: "ObjectBase",
        side: str = "right",
        clearance: float = 0.01,
        align_bottom: bool = True,
    ) -> "ObjectBase":
        """
        Place this object next to a target object.

        This method automatically computes the appropriate pose to place this object
        beside the target object, accounting for both objects' geometries.

        **Important**: Directions are in the world coordinate frame, not relative to
        the target object's orientation:
        - "right" = -Y world direction
        - "left" = +Y world direction
        - "front" = -X world direction
        - "back" = +X world direction

        Args:
            target: The target object to place this object next to.
            side: Which side to place the object ("left", "right", "front", "back").
                  These directions are in world frame, not relative to target's orientation.
            clearance: Horizontal clearance between objects (default: 0.01).
            align_bottom: If True, align bottoms; if False, center vertically (default: True).

        Returns:
            Self, to allow method chaining.

        Example:
            ```python
            laptop = asset_registry.get_asset_by_name("laptop")()
            mug = asset_registry.get_asset_by_name("mug")()
            # Places mug in -Y direction from laptop (world frame)
            mug.next_to(laptop, side="right")
            scene = Scene(assets=[laptop, mug])
            ```

        Note:
            This is a limitation of the MVP. Future versions may support
            placement relative to the target object's local coordinate frame.
        """
        from isaaclab_arena.utils.spatial_relationships import compute_bounding_box_from_usd, compute_next_to_pose

        # Get bounding boxes for both objects
        object_usd_path = getattr(self, "usd_path", None)
        object_scale = getattr(self, "scale", (1.0, 1.0, 1.0))
        target_usd_path = getattr(target, "usd_path", None)
        target_scale = getattr(target, "scale", (1.0, 1.0, 1.0))

        if object_usd_path is None:
            raise ValueError(f"Object {self.name} does not have a usd_path attribute")
        if target_usd_path is None:
            raise ValueError(f"Target object {target.name} does not have a usd_path attribute")

        # Get the target's current pose (if set)
        target_pose = getattr(target, "initial_pose", None)

        # Compute bounding boxes
        object_bbox = compute_bounding_box_from_usd(object_usd_path, scale=object_scale)
        target_bbox = compute_bounding_box_from_usd(target_usd_path, scale=target_scale, pose=target_pose)

        # Compute the placement pose
        placement_pose = compute_next_to_pose(
            object_bbox=object_bbox,
            target_bbox=target_bbox,
            side=side,
            clearance=clearance,
            align_bottom=align_bottom,
        )

        # Set the initial pose on this object
        self.set_initial_pose(placement_pose)

    @abstractmethod
    def set_initial_pose(self, pose: "Pose") -> None:
        """Set the initial pose of the object. Must be implemented by subclasses."""
        pass
