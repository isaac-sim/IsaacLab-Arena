# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena volume-deformable object."""

from __future__ import annotations

import torch

from isaaclab.assets import DeformableObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg

from isaaclab_arena.assets.deformable_spawn import (
    VolumeDeformableMaterial,
    build_newton_volume_spawn,
    build_physx_volume_spawn,
)
from isaaclab_arena.assets.object import SpawnPrim
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.terms.events import set_deformable_object_pose, set_deformable_object_pose_per_env
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange


class DeformableTransform:
    """Nodal state, reset, and backend resolution behavior for volume deformables."""

    def _init_deformable_transform(
        self,
        *,
        material: VolumeDeformableMaterial,
        initial_pose: Pose | PosePerEnv | None = None,
        visual_material: VisualMaterialCfg | None = None,
        local_bounding_box: AxisAlignedBoundingBox | None = None,
    ) -> None:
        """Initialize deformable state after the concrete object initializes its prim source."""
        self.material = material
        self.visual_material = visual_material
        self._local_bounding_box = local_bounding_box or self._bounding_box_from_spawner(self.spawner_cfg)
        self.initial_pose = initial_pose
        self.reset_pose = True
        self._pose_event_cfg = self._build_reset_event()

    @staticmethod
    def _bounding_box_from_spawner(
        spawner_cfg: DeformableObjectSpawnerCfg | None,
    ) -> AxisAlignedBoundingBox | None:
        if not isinstance(spawner_cfg, MeshCuboidCfg):
            return None
        half_size = tuple(size * 0.5 for size in spawner_cfg.size)
        return AxisAlignedBoundingBox(
            min_point=tuple(-value for value in half_size),
            max_point=half_size,
        )

    def resolve_object_cfg(self, physics_preset: object | None = None) -> DeformableObjectCfg:
        """Resolve this object for PhysX by default or an explicit Newton preset."""
        selected_preset = physics_preset or "physx"
        spawn_builder = {
            "default": build_physx_volume_spawn,
            "physx": build_physx_volume_spawn,
            "newton_mjwarp_vbd": build_newton_volume_spawn,
        }.get(selected_preset)
        if spawn_builder is None:
            raise ValueError(f"Physics preset {selected_preset!r} does not support volume deformables")
        source = self.usd_path if self.usd_path is not None else self.spawner_cfg
        spawn_cfg = spawn_builder(source, self.material, visual_material=self.visual_material, scale=self.scale)
        object_cfg = DeformableObjectCfg(prim_path=self.prim_path, spawn=spawn_cfg, **self.asset_cfg_addon)
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is not None:
            object_cfg.init_state.pos = initial_pose.position_xyz
            object_cfg.init_state.rot = initial_pose.rotation_xyzw
        self.object_cfg = object_cfg
        return object_cfg

    def _set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        assert isinstance(pose, (Pose, PosePerEnv)), "Deformables support fixed Pose or PosePerEnv only"
        super()._set_initial_pose(pose)
        self.object_cfg = None

    def set_initial_velocity(self, velocity) -> None:
        """Set the linear velocity restored by the deformable reset event."""
        self.initial_velocity = velocity
        self._pose_event_cfg = self._build_reset_event()

    def _build_reset_event(self) -> EventTermCfg | None:
        if not self.reset_pose or self.initial_pose is None:
            return None
        if isinstance(self.initial_pose, PosePerEnv):
            return EventTermCfg(
                func=set_deformable_object_pose_per_env,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(self.name), "pose_list": self.initial_pose.poses},
            )
        return EventTermCfg(
            func=set_deformable_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.name),
                "pose": self.initial_pose,
                "velocity": self.initial_velocity,
            },
        )

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self._pose_event_cfg = self._build_reset_event()

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self._pose_event_cfg = self._build_reset_event()

    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Return centroid position with identity orientation."""
        env = getattr(env, "unwrapped", env)
        asset = env.scene[self.name]
        object_pos = asset.data.root_pos_w.torch.clone()
        object_quat = torch.zeros((env.num_envs, 4), device=env.device)
        object_quat[:, 3] = 1.0
        object_pose = torch.cat((object_pos, object_quat), dim=-1)
        if is_relative:
            object_pose[:, :3] -= env.scene.env_origins
        return object_pose

    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Transform the default nodal state to the requested centroid pose."""
        env = getattr(env, "unwrapped", env)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        set_deformable_object_pose(env, env_ids, SceneEntityCfg(self.name), pose)

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None):
        raise NotImplementedError("Deformable objects do not support contact sensors")

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        assert self._local_bounding_box is not None, "A local bounding box is required for non-cuboid deformables"
        return self._local_bounding_box


class DeformableObject(DeformableTransform, SpawnPrim, ObjectBase):
    """Spawned volume deformable resolved to one concrete Lab config at build time."""

    def __init__(
        self,
        name: str,
        material: VolumeDeformableMaterial,
        usd_path: str | None = None,
        spawner_cfg: DeformableObjectSpawnerCfg | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | PosePerEnv | None = None,
        visual_material: VisualMaterialCfg | None = None,
        scale: tuple[float, float, float] | None = None,
        local_bounding_box: AxisAlignedBoundingBox | None = None,
        asset_cfg_addon: dict | None = None,
        **kwargs,
    ):
        super().__init__(name=name, prim_path=prim_path, object_type=ObjectType.DEFORMABLE, **kwargs)
        self._init_spawn_prim(
            usd_path=usd_path,
            spawner_cfg=spawner_cfg,
            scale=scale,
            asset_cfg_addon=asset_cfg_addon,
        )
        self._init_deformable_transform(
            material=material,
            initial_pose=initial_pose,
            visual_material=visual_material,
            local_bounding_box=local_bounding_box,
        )
