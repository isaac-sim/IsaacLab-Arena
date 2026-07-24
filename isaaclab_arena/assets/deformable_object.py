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

from isaaclab_arena.assets.deformable_spawn import DeformableMaterial, backend_object_preset, build_deformable_spawn
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

    The object is declared backend-neutrally: a ``usd_path`` (a pre-tetrahedralized asset) and a
    :class:`~isaaclab_arena.assets.deformable_spawn.DeformableMaterial`. ``_init_object_cfg`` fans a
    single per-backend spawn builder across every soft-body physics preset via
    :func:`~isaaclab_arena.assets.deformable_spawn.backend_object_preset`, so the object names no
    physics backend or preset variant. The active preset is selected at build time by ``--presets``.
    """

    def __init__(
        self,
        name: str,
        usd_path: str,
        material: DeformableMaterial,
        local_bounding_box: AxisAlignedBoundingBox,
        visual_material: VisualMaterialCfg,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        relations: list[RelationBase] | None = None,
        asset_cfg_addon: dict | None = None,
        **kwargs,
    ):
        # NOTE: the config generators below read these attributes, so assign them before building the
        # object/event configs.
        self._usd_path = usd_path
        self._material = material
        self._visual_material = visual_material
        self._local_bounding_box = local_bounding_box
        self.asset_cfg_addon = asset_cfg_addon or {}
        super().__init__(name=name, prim_path=prim_path, object_type=ObjectType.DEFORMABLE, **kwargs)
        self.initial_pose = initial_pose
        self.relations = list(relations or [])
        self.reset_pose = True
        self.object_cfg = self._init_object_cfg()
        self.event_cfg = self._init_event_cfg()

    def requires_soft_body_solver(self) -> bool:
        return True

    def add_relation(self, relation: RelationBase) -> None:
        """Add a relation to this object."""
        self.relations.append(relation)

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def disable_reset_pose(self) -> None:
        self.reset_pose = False
        self.event_cfg = self._init_event_cfg()

    def enable_reset_pose(self) -> None:
        self.reset_pose = True
        self.event_cfg = self._init_event_cfg()

    def _make_deformable_cfg(self, backend: SimulationBackend) -> DeformableObjectCfg:
        """Wrap the backend's deformable spawn into a ``DeformableObjectCfg`` with the initial pose."""
        spawn_cfg = build_deformable_spawn(
            self._usd_path, self._material, backend, visual_material=self._visual_material
        )
        object_cfg = DeformableObjectCfg(prim_path=self.prim_path, spawn=spawn_cfg, **self.asset_cfg_addon)
        return self._add_initial_pose_to_cfg(object_cfg)

    def _init_object_cfg(self):
        """Build a per-preset ``PresetCfg`` of ``DeformableObjectCfg`` across soft-body presets."""
        return backend_object_preset(self._make_deformable_cfg, soft_body_only=True)

    def set_initial_pose(self, pose: Pose | PosePerEnv) -> None:
        """Set the initial pose and rebuild the per-backend cfg preset and reset event."""
        self.initial_pose = pose
        # A deformable's ``object_cfg`` is a ``PresetCfg`` bundle, so we regenerate it rather than
        # mutate a single ``init_state`` in place.
        self.object_cfg = self._init_object_cfg()
        self.event_cfg = self._init_event_cfg()

    def set_initial_velocity(self, velocity) -> None:
        """Store the initial (linear) velocity, applied to the nodal state on reset."""
        # ``DeformableObjectCfg`` has no ``init_state`` velocity field; the velocity is applied to the
        # nodal state by the reset event, so we only store it and refresh the event.
        self.initial_velocity = velocity
        self.event_cfg = self._init_event_cfg()

    def _requires_reset_pose_event(self) -> bool:
        return self.get_initial_pose() is not None and self.reset_pose

    def _init_event_cfg(self) -> EventTermCfg | None:
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
        assert self.name in env.unwrapped.scene.keys(), f"Asset {self.name} not found in scene"
        asset = env.unwrapped.scene[self.name]
        object_pos = asset.data.root_pos_w.torch.clone()
        object_quat = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        object_quat[:, 3] = 1.0
        object_pose = torch.cat([object_pos, object_quat], dim=-1)
        if is_relative:
            object_pose[:, :3] -= env.unwrapped.scene.env_origins
        return object_pose

    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Reset the deformable's nodal state so its centroid is at ``pose``."""
        assert self.name in env.unwrapped.scene.keys(), f"Asset {self.name} not found in scene"
        if env_ids is None:
            env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
        set_deformable_object_pose(env.unwrapped, env_ids=env_ids, asset_cfg=SceneEntityCfg(self.name), pose=pose)

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
