# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from copy import deepcopy

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_body_path_alignment import align_rigid_body_paths


class RigidObjectSet(Object):
    """A set of rigid objects with one member selected per environment."""

    def __init__(
        self,
        name: str,
        objects: list[Object],
        prim_path: str | None = None,
        random_choice: bool = False,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        """
        Args:
            name: The name of the object set.
            objects: The list of objects to be included in the object set. Each member spawns from a copy
                of its own object_cfg.spawn, taken when the set is built, so configure each member before
                building the set. Set-wide settings can be changed on the set's object_cfg.spawn, but
                per-environment assignment rebuilds the entries in object_cfg.spawn.assets_cfg from these
                copies.
            prim_path: The prim path of the object set. Note that for all environments, the object set
                prim path must be the same.
            random_choice: Whether to randomly choose an object from the object set to spawn in
                each environment. If False, members are assigned by repeating
                the member order across environments.
            initial_pose: The initial pose of the object from this object set.
        """
        assert len(objects) >= 1, f"Object set {name} must contain at least 1 object."
        assert "scale" not in kwargs, (
            f"Object set {name} does not take a scale: each member keeps the scale of its own spawn configuration. "
            "Set the scale on the member objects instead."
        )
        for member in objects:
            assert not isinstance(
                member, RigidObjectSet
            ), f"Object set {name} cannot contain another object set ({member.name}); add its members instead."
            assert member.object_type == ObjectType.RIGID, f"Object set {name} must contain only rigid objects."

        # Isaac Lab finds the rigid body in one environment and uses its path in all of them,
        # so the members' spawn configurations are aligned to one rigid-body path.
        member_spawn_cfgs = align_rigid_body_paths([member.object_cfg.spawn for member in objects])
        if len(member_spawn_cfgs) == 1:
            spawn_cfg = member_spawn_cfgs[0]
        else:
            spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=member_spawn_cfgs,
                # Arena assigns members to environments. Isaac Lab ignores this flag and warns when it is True.
                random_choice=False,
                # A value here would replace every member's own setting.
                activate_contact_sensors=None,
            )

        self.objects: list[Object] = objects
        self.random_choice = random_choice
        # TODO(cvolk, 2026.10.13): assign_variants() and variant_indices_by_env mean the member chosen for each
        # environment, not USD variants. Both go away when per-environment assignment moves to Object.
        self.variant_indices_by_env: list[int] | None = None

        if prim_path is None:
            prim_path = f"{{ENV_REGEX_NS}}/{name}"

        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawn_cfg,
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )
        # Object gives every rigid object an asset_selection variation. A set spawns its own members, which a
        # selected asset would replace, so the set does not offer it.
        del self.variations["asset_selection"]

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return one local bbox for callers that cannot vary by env.

        The returned bbox has shape (1, 3) and uses the member with the
        greatest z-extent. Heterogeneous placement uses
        get_bounding_box_per_env() after assign_variants() so each env
        uses the geometry of its assigned member.
        """
        return max(self.objects, key=lambda obj: obj.get_bounding_box().size[0, 2].item()).get_bounding_box()

    def assign_variants(self, num_envs: int, variant_seed: int | None = None) -> None:
        """Assign one member index to each environment and keep it.

        The assignment is fixed for the lifetime of the object set so spawned
        members and per-env bboxes stay aligned across placement refills.
        Subsequent calls with the same num_envs are no-ops. A call with a
        different num_envs regenerates with a warning. When random_choice is True, each env
        independently samples one member; otherwise assignments repeat the
        member order across environments.
        Regeneration is safe before the scene is spawned; afterwards, per-env
        bboxes can desync from the spawned members.

        Callers invoke this once num_envs is known, before reading
        variant_indices_by_env or get_bounding_box_per_env.

        Args:
            num_envs: Number of environments to assign members to.
            variant_seed: Optional seed used when random_choice=True.
        """
        if self.variant_indices_by_env is not None:
            if len(self.variant_indices_by_env) == num_envs:
                return
            print(f"Warning: RigidObjectSet '{self.name}' regenerating member assignments for {num_envs} envs.")
        self._apply_member_assignment(self._choose_member_indices(num_envs, seed=variant_seed))

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return the bounding box of each environment's assigned member.

        Requires assign_variants(num_envs) to have been called first. The
        returned bbox has shape (num_envs, 3).

        Args:
            num_envs: Number of environments. Must match the assignment.

        Returns:
            AxisAlignedBoundingBox with min_point / max_point of
            shape (num_envs, 3).
        """
        assert self.variant_indices_by_env is not None, (
            f"RigidObjectSet '{self.name}' has no member assignment; "
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

    def get_contact_sensor_prim_path(self) -> str:
        """Return the contact-sensor path shared by all spawn entries of the set."""
        spawn_cfg = self.object_cfg.spawn
        # One entry per member, or one per environment once assign_variants() has expanded the live config.
        if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
            spawn_entries = spawn_cfg.assets_cfg
        else:
            spawn_entries = [spawn_cfg]
        # Isaac Lab finds the rigid body in one environment and uses its path in every environment.
        body_paths = {self._get_spawn_cfg_geometry(entry).get_contact_body_path() for entry in spawn_entries}
        assert len(body_paths) == 1, (
            f"RigidObjectSet '{self.name}' spawn entries must have the same contact-sensor prim path; "
            f"got rigid-body paths {sorted(body_paths)}."
        )
        return self.prim_path + body_paths.pop()

    def _choose_member_indices(self, num_envs: int, seed: int | None = None) -> list[int]:
        """Return one member index per env.

        Ordered sets repeat member order. Random sets sample independently per
        env, using a local generator when seed is set.
        """
        n = len(self.objects)
        if not self.random_choice:
            return [env_idx % n for env_idx in range(num_envs)]
        if seed is None:
            return torch.randint(low=0, high=n, size=(num_envs,)).tolist()
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randint(low=0, high=n, size=(num_envs,), generator=generator).tolist()

    def _apply_member_assignment(self, member_index_by_env: list[int]) -> None:
        """Validate and store each environment's member index, then give each environment its member's spawn entry."""
        n = len(self.objects)
        assert all(
            0 <= idx < n for idx in member_index_by_env
        ), f"RigidObjectSet '{self.name}' member indices must be in [0, {n}); got {member_index_by_env}."
        self.variant_indices_by_env = member_index_by_env
        if isinstance(self.spawner_cfg, sim_utils.MultiAssetSpawnerCfg):
            # When this set is the scene's only multi-entry spawner, Isaac Lab's default sequential cloning
            # gives environment i the entry i. With several such sets the cloner enumerates the product of their
            # entries, so this mapping does not hold until the assignment is pinned through the clone strategy.
            # Entries come from the member copies taken when the set was built, so a second assignment does not
            # compound and edits to entries of the live config are replaced; set-wide settings on it are kept.
            member_spawn_cfgs = self.spawner_cfg.assets_cfg
            spawn_cfgs_by_env = [deepcopy(member_spawn_cfgs[idx]) for idx in member_index_by_env]
            self.object_cfg.spawn = self.object_cfg.spawn.replace(assets_cfg=spawn_cfgs_by_env)
