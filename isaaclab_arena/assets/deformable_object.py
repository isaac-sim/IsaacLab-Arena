# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from isaaclab.assets import DeformableObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg

from isaaclab_arena.assets.deformable_spawn import (
    DeformableMaterial,
    DeformableSource,
    SurfaceDeformableMaterial,
    backend_object_preset,
    build_deformable_spawn,
)
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.environments.physics_presets import SimulationBackend
from isaaclab_arena.relations.relations import RelationBase
from isaaclab_arena.terms.events import set_deformable_object_pose, set_deformable_object_pose_per_env
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class DeformableObject(ObjectBase):
    """A soft (FEM) object simulated as an Isaac Lab ``DeformableObject``.

    Deformables are a first-class scene category in Isaac Lab: a ``DeformableObjectCfg`` is routed
    into the scene's ``deformable_objects`` by ``InteractiveScene`` and its state is the nodal
    positions/velocities, not a rigid root pose. This class localizes all deformable-specific
    behavior (nodal-state pose get/set, nodal reset events) so that ``SpawnableObjectBase``/``Object``
    stay free of deformable branches.

    The object is declared backend-neutrally: either a ``usd_path`` or an Isaac Lab mesh spawner plus
    a backend-neutral deformable material. ``_init_object_cfg`` fans a single per-backend spawn
    builder across every soft-body physics preset via
    :func:`~isaaclab_arena.assets.deformable_spawn.backend_object_preset`, so the object names no
    physics backend or preset variant. The active preset is selected at build time by ``--presets``.
    """

    def __init__(
        self,
        name: str,
        material: DeformableMaterial | SurfaceDeformableMaterial,
        local_bounding_box: AxisAlignedBoundingBox,
        visual_material: VisualMaterialCfg,
        usd_path: str | None = None,
        spawner_cfg: DeformableObjectSpawnerCfg | None = None,
        scale: tuple[float, float, float] | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        asset_cfg_addon: dict | None = None,
        **kwargs,
    ):
        assert (usd_path is None) != (spawner_cfg is None), "Pass exactly one of usd_path or spawner_cfg."
        # NOTE: the config generators below read these attributes, so assign them before building the
        # object/event configs.
        self._source: DeformableSource = usd_path if usd_path is not None else spawner_cfg
        self._material = material
        self._visual_material = visual_material
        self._local_bounding_box = local_bounding_box
        self._scale = scale
        self.asset_cfg_addon = asset_cfg_addon or {}
        super().__init__(name=name, prim_path=prim_path, object_type=ObjectType.DEFORMABLE, **kwargs)
        self.initial_pose = initial_pose
        self.relations = list(relations or [])
        self.reset_pose = True
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()

    def requires_soft_body_solver(self) -> bool:
        return True

    def soft_body_kinds(self) -> frozenset[str]:
        return frozenset({self._material.kind.value})

    def add_relation(self, relation: RelationBase) -> None:
        """Add a relation to this object."""
        self.relations.append(relation)

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self._pose_event_cfg = self._build_reset_event()

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self._pose_event_cfg = self._build_reset_event()

    def _make_deformable_cfg(self, backend: SimulationBackend) -> DeformableObjectCfg:
        """Wrap the backend's deformable spawn into a ``DeformableObjectCfg`` with the initial pose."""
        spawn_cfg = build_deformable_spawn(
            self._source,
            self._material,
            backend,
            visual_material=self._visual_material,
            scale=self._scale,
        )
        object_cfg = DeformableObjectCfg(prim_path=self.prim_path, spawn=spawn_cfg, **self.asset_cfg_addon)
        return self._add_initial_pose_to_cfg(object_cfg)

    def _init_object_cfg(self):
        """Build a per-preset ``PresetCfg`` of ``DeformableObjectCfg`` across soft-body presets."""
        return backend_object_preset(self._make_deformable_cfg, soft_body_only=True)

    def _set_initial_pose(self, pose: Pose | PosePerEnv) -> None:
        """Store the pose and rebuild the per-backend cfg preset."""
        assert isinstance(pose, (Pose, PosePerEnv)), "Deformable reset currently supports Pose or PosePerEnv only."
        self.initial_pose = pose
        # A deformable's ``object_cfg`` is a ``PresetCfg`` bundle, so we regenerate it rather than
        # mutate a single ``init_state`` in place.
        if self.object_cfg is not None:
            self.object_cfg = self._init_object_cfg()

    def set_initial_velocity(self, velocity) -> None:
        """Store the initial (linear) velocity, applied to the nodal state on reset."""
        # ``DeformableObjectCfg`` has no ``init_state`` velocity field; the velocity is applied to the
        # nodal state by the reset event, so we only store it and refresh the event.
        self.initial_velocity = velocity
        self._pose_event_cfg = self._build_reset_event()

    def _requires_reset_pose_event(self) -> bool:
        return self.get_initial_pose() is not None and self.reset_pose

    def _build_reset_event(self) -> EventTermCfg | None:
        if not self._requires_reset_pose_event():
            return None
        initial_pose = self.get_initial_pose()
        if isinstance(initial_pose, PosePerEnv):
            return EventTermCfg(
                func=set_deformable_object_pose_per_env,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(self.name), "pose_list": initial_pose.poses},
            )
        assert isinstance(initial_pose, Pose), "Deformable reset currently supports Pose or PosePerEnv only."
        return EventTermCfg(
            func=set_deformable_object_pose,
            mode="reset",
            params={
                "pose": initial_pose,
                "asset_cfg": SceneEntityCfg(self.name),
                "velocity": self.initial_velocity,
            },
        )

    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Return the deformable centroid pose (nodal mean position, identity orientation).

        Deformables have no rigid root orientation, so the returned quaternion is identity. The order
        matches the rigid path: (x, y, z, qx, qy, qz, qw). Shape is (num_envs, 7).
        """
        env = getattr(env, "unwrapped", env)
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        asset = env.scene[self.name]
        object_pos = asset.data.root_pos_w.torch.clone()
        object_quat = torch.zeros((env.num_envs, 4), device=env.device)
        object_quat[:, 3] = 1.0
        object_pose = torch.cat([object_pos, object_quat], dim=-1)
        if is_relative:
            object_pose[:, :3] -= env.scene.env_origins
        return object_pose

    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Reset the deformable's nodal state so its centroid is at ``pose``."""
        env = getattr(env, "unwrapped", env)
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        set_deformable_object_pose(env, env_ids=env_ids, asset_cfg=SceneEntityCfg(self.name), pose=pose)

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None):
        raise NotImplementedError("Deformable objects carry no contact sensor.")

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the local (object-frame) bounding box derived from the object's shape."""
        return self._local_bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the world bounding box (translation only; deformables carry no root rotation)."""
        local_bbox = self.get_bounding_box()
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is None:
            return local_bbox
        return local_bbox.translated(initial_pose.position_xyz)
