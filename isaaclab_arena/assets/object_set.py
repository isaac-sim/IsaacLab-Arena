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
    """A set of rigid objects with one member selected per environment."""

    def __init__(
        self,
        name: str,
        objects: list[Object],
        prim_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        random_choice: bool = False,
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
            initial_pose: The initial pose of the object from this object set.
        """
        assert len(objects) >= 1, f"Object set {name} must contain at least 1 object."
        assert self._are_all_objects_type_rigid(objects), f"Object set {name} must contain only rigid objects."

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
            self.member_usd_paths: list[str] = self._modify_assets(objects)
            print(f"Modified object USD paths: {self.member_usd_paths}")
        else:
            self.member_usd_paths = []
            for obj in objects:
                assert obj.usd_path is not None
                self.member_usd_paths.append(obj.usd_path)

        self.objects: list[Object] = objects
        self.random_choice = random_choice
        self.variant_indices_by_env: list[int] | None = None

        if prim_path is None:
            prim_path = f"{{ENV_REGEX_NS}}/{name}"

        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            usd_path="",
            prim_path=prim_path,
            scale=(1.0, 1.0, 1.0),  # We rewrite the USDs to handle scaling.
            initial_pose=initial_pose,
            **kwargs,
        )

    @property
    def object_usd_paths(self) -> list[str]:
        """USD paths passed to MultiUsdFileCfg.

        Before assignment this is the member USD list. After assignment this
        returns one USD path per environment based on variant_indices_by_env.
        """
        if self.variant_indices_by_env is not None:
            return [self.member_usd_paths[idx] for idx in self.variant_indices_by_env]
        return self.member_usd_paths

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return one local bbox for callers that cannot vary by env.

        The returned bbox has shape (1, 3) and uses the member with the
        greatest z-extent. Heterogeneous placement uses
        get_bounding_box_per_env() after assign_variants() so each env
        uses its actual variant geometry.
        """
        return max(self.objects, key=lambda obj: obj.get_bounding_box().size[0, 2].item()).get_bounding_box()

    def assign_variants(self, num_envs: int, variant_seed: int | None = None) -> None:
        """Fix one member-variant index per environment.

        The assignment is fixed for the lifetime of the object set so spawned
        USDs and per-env bboxes stay aligned across placement refills.
        Subsequent calls with the same num_envs are no-ops. A call with a
        different num_envs regenerates with a warning. When random_choice is True, each env
        independently samples one variant; otherwise assignments repeat the
        member order across environments.
        Regeneration is safe before the scene is spawned; afterwards, per-env
        bboxes can desync from the spawned USDs.

        Callers invoke this once num_envs is known, before reading
        variant_indices_by_env or get_bounding_box_per_env.

        Args:
            num_envs: Number of environments to assign variants for.
            variant_seed: Optional seed used when random_choice=True.
        """
        if self.variant_indices_by_env is not None:
            if len(self.variant_indices_by_env) == num_envs:
                return
            print(f"Warning: RigidObjectSet '{self.name}' regenerating variant assignments for {num_envs} envs.")
        self._set_variant_indices_by_env(self._generate_variant_indices(num_envs, variant_seed=variant_seed))

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return each env's actual variant bbox.

        Requires assign_variants(num_envs) to have been called first. The
        returned bbox has shape (num_envs, 3).

        Args:
            num_envs: Number of environments. Must match the assignment.

        Returns:
            AxisAlignedBoundingBox with min_point / max_point of
            shape (num_envs, 3).
        """
        assert self.variant_indices_by_env is not None, (
            f"RigidObjectSet '{self.name}' has no variant assignment; "
            "call assign_variants(num_envs) before get_bounding_box_per_env()."
        )
        assert len(self.variant_indices_by_env) == num_envs, (
            f"RigidObjectSet '{self.name}' got request for {num_envs} envs, "
            f"but is assigned for {len(self.variant_indices_by_env)} envs."
        )
        bounding_boxes = [obj.get_bounding_box() for obj in self.objects]

        min_pts = torch.stack([bounding_boxes[idx].min_point[0] for idx in self.variant_indices_by_env], dim=0)
        max_pts = torch.stack([bounding_boxes[idx].max_point[0] for idx in self.variant_indices_by_env], dim=0)
        return AxisAlignedBoundingBox(min_point=min_pts, max_point=max_pts)

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        # We assume that by here, our USDs have been modified to be compatible with each other
        # and we can use the canonical first member USD to find the shallowest rigid body.
        return super().get_contact_sensor_cfg(contact_against_object, usd_path=self.member_usd_paths[0])

    def _generate_variant_indices(self, num_envs: int, variant_seed: int | None = None) -> list[int]:
        """Return one member index per env.

        Ordered sets repeat member order. Random sets sample independently per
        env, using a local generator when variant_seed is set.
        """
        n = len(self.objects)
        if not self.random_choice:
            return [env_idx % n for env_idx in range(num_envs)]
        if variant_seed is None:
            return torch.randint(low=0, high=n, size=(num_envs,)).tolist()
        generator = torch.Generator()
        generator.manual_seed(variant_seed)
        return torch.randint(low=0, high=n, size=(num_envs,), generator=generator).tolist()

    def _set_variant_indices_by_env(self, variant_indices_by_env: list[int]) -> None:
        """Validate and store variant indices, then sync spawn config when it exists."""
        n = len(self.objects)
        assert all(
            0 <= idx < n for idx in variant_indices_by_env
        ), f"RigidObjectSet '{self.name}' variant indices must be in [0, {n}); got {variant_indices_by_env}."
        self.variant_indices_by_env = variant_indices_by_env
        # During __init__, Object.object_cfg has not been built yet; _generate_rigid_cfg()
        # reads object_usd_paths after this assignment.
        spawn_cfg = self.object_cfg.spawn if getattr(self, "object_cfg", None) is not None else None
        if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
            spawn_cfg.usd_path = self.object_usd_paths

    def _are_all_objects_type_rigid(self, objects: list[Object]) -> bool:
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
                # Arena owns per-env variant assignment so bbox selection and
                # spawned USDs stay aligned.
                random_choice=False,
                activate_contact_sensors=True,
            ),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
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
