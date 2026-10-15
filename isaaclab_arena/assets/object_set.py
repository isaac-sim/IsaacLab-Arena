# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg
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
                building the set. Set-wide settings can be changed on the set's object_cfg.spawn.
            prim_path: The prim path of the object set. Note that for all environments, the object set
                prim path must be the same.
            random_choice: If True, each environment draws its member at random, seeded by the build's
                placement seed. If False, members repeat in order across environments.
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
            assert member.bounding_box is None, (
                f"Object set {name} member {member.name} sets bounding_box; "
                "set bounds come from the members' spawn configurations."
            )

        # Isaac Lab finds the rigid body in one environment and uses its path in all of them,
        # so the members' spawn configurations are aligned to one rigid-body path.
        member_spawn_cfgs = align_rigid_body_paths([member.object_cfg.spawn for member in objects])
        if len(member_spawn_cfgs) == 1:
            spawn_cfg = member_spawn_cfgs[0]
        else:
            spawn_cfg = PinnedMultiAssetSpawnerCfg(
                assets_cfg=member_spawn_cfgs,
                # Arena assigns members to environments. Isaac Lab ignores this flag and warns when it is True.
                random_choice=False,
                # A value here would replace every member's own setting.
                activate_contact_sensors=None,
            )

        self.objects: list[Object] = objects
        self.random_choice = random_choice

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
