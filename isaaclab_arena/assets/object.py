# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from typing import Any

import warp as wp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose

from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.terms.events import set_object_pose, set_object_pose_per_env
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd, has_light, open_stage
from isaaclab_arena.utils.velocity import Velocity
from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation


class SpawnPrim:
    """USD or spawner-backed prim provenance for an Arena object."""

    def _init_spawn_prim(
        self,
        *,
        usd_path: str | None,
        spawner_cfg: SpawnerCfg | None,
        scale: tuple[float, float, float] | None,
        spawn_cfg_addon: dict[str, Any] | None = None,
        asset_cfg_addon: dict[str, Any] | None = None,
    ) -> None:
        """Store one exclusive source for a prim spawned by this object."""
        assert (usd_path is None) != (spawner_cfg is None), "Pass exactly one of usd_path or spawner_cfg"
        self.usd_path = usd_path
        self.spawner_cfg = spawner_cfg
        self.scale = scale
        self.spawn_cfg_addon = spawn_cfg_addon or {}
        self.asset_cfg_addon = asset_cfg_addon or {}
        self.bounding_box = None

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to object origin)."""
        assert self.usd_path is not None
        if self.bounding_box is None:
            self.bounding_box = compute_local_bounding_box_from_usd(self.usd_path, self.scale)
        return self.bounding_box

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        return self.get_bounding_box().get_corners_at(pos)

    def _get_contact_sensor_prim_path(self, usd_path: str | None = None) -> str:
        """Return the shallowest rigid-body prim for contact sensing."""
        usd_path = usd_path or self.usd_path
        if usd_path is None:
            return self.prim_path
        rigid_body_relative_path = find_shallowest_rigid_body(
            usd_path,
            relative_to_root=True,
            variants=(self.spawn_cfg_addon or {}).get("variants"),
        )
        assert (
            rigid_body_relative_path is not None
        ), f"No rigid body found in {self.name} USD file: {usd_path}. Can't add contact sensor."
        return self.prim_path + rigid_body_relative_path

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Return the custom spawner or a USD-file spawner."""
        if self.spawner_cfg is not None:
            return self.spawner_cfg
        return UsdFileCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
            **self.spawn_cfg_addon,
        )

    def _get_cfg_source_kwargs(self, activate_contact_sensors: bool = False) -> dict[str, Any]:
        """Return spawn-specific arguments for an Isaac Lab asset config."""
        return {
            "spawn": self._get_spawn_cfg(activate_contact_sensors),
            **self.asset_cfg_addon,
        }

    def _prepare_base_cfg_source(self) -> None:
        """Warn when a spawned static USD contains lights."""
        if self.spawner_cfg is None:
            with open_stage(self.usd_path) as stage:
                if has_light(stage):
                    print(
                        "WARNING: Base object has lights, this may cause issues when using with multiple environments."
                    )


class RootedTransform:
    """Root-transform state and reset behavior for rigid, articulated, and static objects."""

    def _init_rooted_transform(self) -> None:
        """Initialize state shared by root-transform physics representations."""
        self.reset_pose = True
        if self.object_type == ObjectType.RIGID:
            self.add_variation(ObjectMassVariation(self.name))

    def _set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Store the pose and write its construction values into the object config."""
        super()._set_initial_pose(pose)
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is not None and self.object_cfg is not None:
            self.object_cfg.init_state.pos = initial_pose.position_xyz
            self.object_cfg.init_state.rot = initial_pose.rotation_xyzw

    def set_initial_velocity(self, velocity: Velocity) -> None:
        """Set the initial velocity on the object config and its reset event."""
        self.initial_velocity = velocity
        if self.object_cfg is not None and hasattr(self.object_cfg.init_state, "lin_vel"):
            self.object_cfg.init_state.lin_vel = velocity.linear_xyz
        if self.object_cfg is not None and hasattr(self.object_cfg.init_state, "ang_vel"):
            self.object_cfg.init_state.ang_vel = velocity.angular_xyz
        self._pose_event_cfg = self._build_reset_event()

    def get_contact_sensor_cfg(
        self,
        contact_against_object: ObjectBase | None = None,
        usd_path: str | None = None,
    ) -> ContactSensorCfg:
        """Build a contact sensor config using provenance-specific prim-path hooks."""
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        return ContactSensorCfg(
            prim_path=self._get_contact_sensor_prim_path(usd_path),
            filter_prim_paths_expr=(
                [] if contact_against_object is None else [contact_against_object._get_contact_sensor_prim_path()]
            ),
        )

    def _requires_reset_pose_event(self) -> bool:
        """Return whether this object needs a root-pose reset event."""
        return self.get_initial_pose() is not None and self.object_type in (
            ObjectType.RIGID,
            ObjectType.ARTICULATION,
        )

    def _build_reset_event(self) -> EventTermCfg | None:
        """Build the event that restores this object's pose and velocity."""
        if not self._requires_reset_pose_event():
            return None

        initial_pose = self.get_initial_pose()
        if isinstance(initial_pose, PosePerEnv):
            return EventTermCfg(
                func=set_object_pose_per_env,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.name),
                    "pose_list": initial_pose.poses,
                },
            )
        if isinstance(initial_pose, PoseRange):
            return EventTermCfg(
                func=randomize_object_pose,
                mode="reset",
                params={
                    "pose_range": initial_pose.to_dict(),
                    "asset_cfgs": [SceneEntityCfg(self.name)],
                },
            )
        return EventTermCfg(
            func=set_object_pose,
            mode="reset",
            params={
                "pose": initial_pose,
                "asset_cfg": SceneEntityCfg(self.name),
                "velocity": self.initial_velocity,
            },
        )

    def _init_object_cfg(self) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg:
        if self.object_type == ObjectType.RIGID:
            return self._generate_rigid_cfg()
        if self.object_type == ObjectType.ARTICULATION:
            return self._generate_articulation_cfg()
        if self.object_type == ObjectType.BASE:
            return self._generate_base_cfg()
        raise ValueError(f"Invalid object type: {self.object_type}")

    def _get_cfg_source_kwargs(self, activate_contact_sensors: bool = False) -> dict[str, Any]:
        """Return provenance-specific arguments for an Isaac Lab asset config."""
        raise NotImplementedError

    def _prepare_base_cfg_source(self) -> None:
        """Perform provenance-specific checks before creating a base asset config."""

    def _add_initial_pose_to_cfg(
        self, object_cfg: RigidObjectCfg | ArticulationCfg | AssetBaseCfg
    ) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg:
        """Apply the resolved initial root pose to an Isaac Lab asset config."""
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is not None:
            object_cfg.init_state.pos = initial_pose.position_xyz
            object_cfg.init_state.rot = initial_pose.rotation_xyzw
        return object_cfg

    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Return the object's root pose in world or environment-relative coordinates."""
        assert self.name in env.unwrapped.scene.keys(), f"Asset {self.name} not found in scene"
        if self.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION):
            object_pose = wp.to_torch(env.unwrapped.scene[self.name].data.root_pose_w).clone()
        elif self.object_type == ObjectType.BASE:
            object_pose = torch.cat(env.unwrapped.scene[self.name].get_world_poses(), dim=-1)
        else:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")
        if is_relative:
            object_pose[:, :3] -= env.unwrapped.scene.env_origins
        return object_pose

    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Set the object's root pose and zero velocity in selected environments."""
        env = env.unwrapped
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        set_object_pose(env, env_ids, SceneEntityCfg(self.name), pose)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            **self._get_cfg_source_kwargs(activate_contact_sensors=True),
        )
        return self._add_initial_pose_to_cfg(object_cfg)

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            **self._get_cfg_source_kwargs(activate_contact_sensors=True),
            actuators={},
        )
        return self._add_initial_pose_to_cfg(object_cfg)

    def _generate_base_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.BASE
        self._prepare_base_cfg_source()
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            **self._get_cfg_source_kwargs(),
        )
        return self._add_initial_pose_to_cfg(object_cfg)


class Object(SpawnPrim, RootedTransform, ObjectBase):
    """Spawned rigid, articulated, or static Arena object."""

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
        self._init_spawn_prim(
            usd_path=usd_path,
            spawner_cfg=spawner_cfg,
            scale=scale,
            spawn_cfg_addon=spawn_cfg_addon,
            asset_cfg_addon=asset_cfg_addon,
        )
        self.initial_pose = initial_pose
        self.relations = list(relations)
        self._init_rooted_transform()
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self._pose_event_cfg = self._build_reset_event()

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self._pose_event_cfg = self._build_reset_event()

    def _requires_reset_pose_event(self) -> bool:
        return super()._requires_reset_pose_event() and self.reset_pose
