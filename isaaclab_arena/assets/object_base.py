# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import warp as wp
from abc import ABC, abstractmethod
from enum import Enum

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.relations.relations import AtPosition, Relation, RelationBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.pose import Pose, PoseRange
from isaaclab_arena.utils.velocity import Velocity


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
        self.initial_pose: Pose | PoseRange | None = None
        self.initial_velocity: Velocity | None = None
        self.object_cfg = None
        self.event_cfg = None
        self.relations: list[RelationBase] = []

    def get_initial_pose(self) -> Pose | PoseRange | None:
        """Return the current initial pose of this object.

        Subclasses may override to derive the pose from other sources
        (e.g. a parent asset), falling back to ``self.initial_pose``.
        """
        return self.initial_pose

    def _get_initial_pose_as_pose(self) -> Pose | None:
        """Return a single ``Pose`` suitable for *init_state* and bounding-box calculations.

        If the initial pose is a ``PoseRange``, its midpoint is returned.
        If the initial pose is ``None``, ``None`` is returned.
        """
        initial_pose = self.get_initial_pose()
        if initial_pose is None:
            return None
        if isinstance(initial_pose, PoseRange):
            return initial_pose.get_midpoint()
        return initial_pose

    def set_initial_pose(self, pose: Pose | PoseRange) -> None:
        """Set / override the initial pose and rebuild derived configs.

        Args:
            pose: A fixed ``Pose`` or a ``PoseRange`` (randomised on reset).
        """
        self.initial_pose = pose
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is not None and self.object_cfg is not None:
            self.object_cfg.init_state.pos = initial_pose.position_xyz
            self.object_cfg.init_state.rot = initial_pose.rotation_xyzw
        self.event_cfg = self._init_event_cfg()

    def set_initial_velocity(self, velocity: Velocity) -> None:
        """Set / override the initial velocity and rebuild derived configs.

        The velocity is applied as ``init_state.lin_vel`` and
        ``init_state.ang_vel`` on the underlying config
        (``RigidObjectCfg`` or ``ArticulationCfg``) and is also restored
        on every environment reset via the reset event.

        Args:
            velocity: A ``Velocity`` specifying linear and angular components.
        """
        self.initial_velocity = velocity
        if self.object_cfg is not None and hasattr(self.object_cfg.init_state, "lin_vel"):
            self.object_cfg.init_state.lin_vel = velocity.linear_xyz
        if self.object_cfg is not None and hasattr(self.object_cfg.init_state, "ang_vel"):
            self.object_cfg.init_state.ang_vel = velocity.angular_xyz
        self.event_cfg = self._init_event_cfg()

    def _requires_reset_pose_event(self) -> bool:
        """Whether a reset-event for the initial pose should be generated.

        Subclasses may override to add extra conditions (e.g. a ``reset_pose`` flag).
        """
        return self.get_initial_pose() is not None and self.object_type in (
            ObjectType.RIGID,
            ObjectType.ARTICULATION,
        )

    def _init_event_cfg(self) -> EventTermCfg | None:
        """Build the ``EventTermCfg`` for resetting this object's pose and velocity."""
        if not self._requires_reset_pose_event():
            return None

        initial_pose = self.get_initial_pose()
        if isinstance(initial_pose, PoseRange):
            return EventTermCfg(
                func=randomize_object_pose,
                mode="reset",
                params={
                    "pose_range": initial_pose.to_dict(),
                    "asset_cfgs": [SceneEntityCfg(self.name)],
                },
            )
        else:
            params: dict = {
                "pose": initial_pose,
                "asset_cfg": SceneEntityCfg(self.name),
                "velocity": self.initial_velocity,
            }
            return EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params=params,
            )

    def get_relations(self) -> list[RelationBase]:
        """Get all relations for this object."""
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        """Get only spatial relations (On, NextTo, AtPosition, etc.), excluding markers like IsAnchor."""
        return [r for r in self.relations if isinstance(r, (Relation, AtPosition))]

    def set_prim_path(self, prim_path: str) -> None:
        self.prim_path = prim_path

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_object_cfg(self) -> tuple[str, RigidObjectCfg | ArticulationCfg | AssetBaseCfg]:
        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, EventTermCfg | None]:
        return self.name, self.event_cfg

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
        assert self.name in env.unwrapped.scene.keys(), f"Asset {self.name} not found in scene"
        if (self.object_type == ObjectType.RIGID) or (self.object_type == ObjectType.ARTICULATION):
            object_pose = wp.to_torch(env.unwrapped.scene[self.name].data.root_pose_w).clone()
        elif self.object_type == ObjectType.BASE:
            object_pose = torch.cat(env.unwrapped.scene[self.name].get_world_poses(), dim=-1)
        else:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")
        if is_relative:
            object_pose[:, :3] -= env.unwrapped.scene.env_origins
        return object_pose

    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Set the pose of the object in the environment.

        Args:
            env: The environment.
            pose: The pose to set.
        """
        assert self.name in env.unwrapped.scene.keys(), f"Asset {self.name} not found in scene"
        if env_ids is None:
            env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
        # Grab the object
        asset = env.unwrapped.scene[self.name]
        num_envs = len(env_ids)
        # Convert the pose to the env frame
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.unwrapped.device).repeat(num_envs, 1)
        pose_t_xyz_q_xyzw[:, :3] += env.unwrapped.scene.env_origins[env_ids]
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(env.unwrapped.num_envs, 6, device=env.unwrapped.device), env_ids=env_ids)

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
