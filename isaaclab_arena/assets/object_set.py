# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.object_set_utils import rescale_rename_rigid_body_and_save_to_cache
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body


class RigidObjectSet(Object):
    """
    A set of rigid objects.
    """

    def __init__(
        self,
        name: str,
        objects: list[Object],
        prim_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        random_choice: bool = False,
        variant_indices_by_env: list[int] | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        """
        Args:
            name: The name of the object set.
            objects: The list of objects to be included in the object set.
            prim_path: The prim path of the object set. Note that for all environments, the object set
                prim path must be the same.
            scale: The scale of the object set. Note all objects can only have the same scale, if
                different scales are needed, considering scaling the object USD file.
            random_choice: Whether to randomly choose an object from the object set to spawn in
                each environment. If False, variants are assigned by repeating
                the member order across environments.
            variant_indices_by_env: Optional fixed variant index for each environment.
            initial_pose: The initial pose of the object from this object set.
        """
        if not self._are_all_objects_type_rigid(objects):
            raise ValueError(f"Object set {name} must contain only rigid objects.")

        # Isaac Lab support for MultiUsdFileCfg is limited. It applies the same scale and pose to all objects.
        # Furthermore it relies on the rigid body being at the root of the USD file, or at the same
        # path in all files. To expand our support in Arena, we modify the USDs to be compatible with each other.
        # In particular, we rescale the assets and rename the rigid bodies to have the same name.
        # We then save the resulting modified USDs to a cache.
        if self._is_asset_modification_needed(objects):
            if not self._asset_modification_possible(objects):
                depths = self._get_all_rigid_body_depths(objects)
                per_asset = [f"{obj.name}=depth_{d} ({obj.usd_path})" for obj, d in zip(objects, depths)]
                raise ValueError(
                    "Asset modification is not possible for object sets: all objects must have their shallowest "
                    "rigid body at the same depth so paths match after rename. "
                    f"Rigid body depths by asset: {per_asset}."
                )
            self.object_usd_paths = self._modify_assets(objects)
            print(f"Modified object USD paths: {self.object_usd_paths}")
        else:
            self.object_usd_paths = []
            for obj in objects:
                assert obj.usd_path is not None
                self.object_usd_paths.append(obj.usd_path)

        self.objects: list[Object] = objects
        self._member_object_usd_paths: list[str] = list(self.object_usd_paths)
        self.random_choice = random_choice
        self.variant_indices_by_env: list[int] | None = None

        if variant_indices_by_env is not None:
            self._set_variant_indices_by_env(variant_indices_by_env)

        # Set default prim_path if not provided
        if prim_path is None:
            prim_path = f"{{ENV_REGEX_NS}}/{name}"

        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            usd_path="",
            prim_path=prim_path,
            scale=(1.0, 1.0, 1.0),  # We rewrite the USDs to handle scaling
            initial_pose=initial_pose,
            **kwargs,
        )
        self.has_env_specific_bboxes = len(objects) > 1

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get the bounding box of the object set.

        Returns the bounding box with the greatest z-extent among all objects in the set.
        This is a heuristic to avoid objects spawning inside their support surfaces.
        """
        return max(self.objects, key=lambda obj: obj.get_bounding_box().size[0, 2].item()).get_bounding_box()

    def get_variant_indices(self, num_envs: int) -> list[int]:
        """Return which member object index is assigned to each environment.

        Multi-variant sets use one fixed assignment for the lifetime of the
        object set. When ``random_choice`` is True, each env independently
        samples one variant once. Otherwise, assignments repeat the member
        order across environments.

        Args:
            num_envs: Number of environments.

        Returns:
            List of length ``num_envs`` with indices into ``self.objects``.
        """
        if self.variant_indices_by_env is None:
            self._set_variant_indices_by_env(self._generate_variant_indices(num_envs))
        elif len(self.variant_indices_by_env) != num_envs:
            raise ValueError(
                f"RigidObjectSet '{self.name}' has variant assignments for "
                f"{len(self.variant_indices_by_env)} envs, got request for {num_envs}."
            )
        assert self.variant_indices_by_env is not None
        return self.variant_indices_by_env

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Get the actual bounding box for each env's variant.

        Unlike ``get_bounding_box()`` (which uses a max-z heuristic), this
        returns the real local bbox of the variant assigned to each env,
        enabling correct collision-free placement for heterogeneous scenes.

        Args:
            num_envs: Number of environments.

        Returns:
            ``AxisAlignedBoundingBox`` with ``min_point`` / ``max_point`` of
            shape ``(num_envs, 3)``.
        """
        variant_indices = self.get_variant_indices(num_envs)
        member_bboxes = [obj.get_bounding_box() for obj in self.objects]

        min_pts = torch.stack([member_bboxes[idx].min_point[0] for idx in variant_indices], dim=0)
        max_pts = torch.stack([member_bboxes[idx].max_point[0] for idx in variant_indices], dim=0)
        return AxisAlignedBoundingBox(min_point=min_pts, max_point=max_pts)

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        # We assume that by here, our USDs have been modified to be compatible with each other
        # and we can use the first USD path to find the shallowest rigid body.
        return super().get_contact_sensor_cfg(contact_against_object, usd_path=self.object_usd_paths[0])

    def _generate_variant_indices(self, num_envs: int) -> list[int]:
        n = len(self.objects)
        if n == 1:
            return [0 for _ in range(num_envs)]
        if not self.random_choice:
            return [env_idx % n for env_idx in range(num_envs)]
        return torch.randint(low=0, high=n, size=(num_envs,)).tolist()

    def _set_variant_indices_by_env(self, variant_indices_by_env: list[int]) -> None:
        n = len(self.objects)
        if any(idx < 0 or idx >= n for idx in variant_indices_by_env):
            raise ValueError(
                f"RigidObjectSet '{self.name}' variant indices must be in [0, {n}); got {variant_indices_by_env}."
            )

        self.variant_indices_by_env = list(variant_indices_by_env)
        if len(self.objects) > 1:
            self.object_usd_paths = [self._member_object_usd_paths[idx] for idx in self.variant_indices_by_env]
            spawn_cfg = self.object_cfg.spawn if getattr(self, "object_cfg", None) is not None else None
            if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
                spawn_cfg.usd_path = self.object_usd_paths
                spawn_cfg.random_choice = False

    def _are_all_objects_type_rigid(self, objects: list[Object]) -> bool:
        if objects is None or len(objects) == 0:
            raise ValueError(f"Object set {self.name} must contain at least 1 object.")
        for obj in objects:
            assert obj.usd_path is not None
            if detect_object_type(usd_path=obj.usd_path) != ObjectType.RIGID:
                return False
        return True

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=self.object_usd_paths,
                random_choice=self.random_choice if self.variant_indices_by_env is None else False,
                activate_contact_sensors=True,
            ),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        assert isinstance(object_cfg, RigidObjectCfg)
        return object_cfg

    def _generate_articulation_cfg(self):
        raise NotImplementedError("Articulation configuration is not supported for object sets")

    def _generate_base_cfg(self):
        raise NotImplementedError("Base configuration is not supported for object sets")

    def _generate_spawner_cfg(self):
        raise NotImplementedError("Spawner configuration is not supported for object sets")

    def _is_asset_modification_needed(self, objects: list[Object]) -> bool:
        # If any asset is scaled, we need to modify the assets
        for asset in objects:
            if asset.scale != (1.0, 1.0, 1.0):
                return True
        # If all assets have rigid bodies at the root, we don't need to modify the assets
        depths = self._get_all_rigid_body_depths(objects)
        if all(depth == 0 for depth in depths):
            return False
        # Otherwise, we need to modify the assets
        return True

    def _asset_modification_possible(self, objects: list[Object]) -> bool:
        # If all assets have their rigid bodies at the same depth,
        # we can modify the assets to be compatible with each other.
        depths = self._get_all_rigid_body_depths(objects)
        return all(depth == depths[0] for depth in depths)

    def _get_all_rigid_body_depths(self, objects: list[Object]) -> list[int]:
        depths = []
        for asset in objects:
            assert asset.usd_path is not None
            shallowest_rigid_body = find_shallowest_rigid_body(asset.usd_path)
            depth = shallowest_rigid_body.count("/") - 1 if shallowest_rigid_body else -1
            depths.append(depth)
        return depths

    def _modify_assets(self, objects: list[Object]) -> list[str]:
        new_usd_paths = []
        for asset in objects:
            new_usd_path = rescale_rename_rigid_body_and_save_to_cache(asset)
            new_usd_paths.append(new_usd_path)
        return new_usd_paths
