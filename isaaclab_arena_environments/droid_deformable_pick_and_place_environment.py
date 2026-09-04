# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""DROID PhysX deformable-object pick-and-place environment.

Launch a short headless rollout from the development container:

    /isaac-sim/python.sh -m isaaclab_arena.evaluation.policy_runner \
        --headless --policy_type zero_action --num_steps 10 \
        droid_deformable_pick_and_place

Select another deformable object with ``--pick_object teddy_bear`` or
``--pick_object surface``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

_PICK_OBJECT_ALIASES = {
    "cube": "deformable_cube",
    "surface": "deformable_surface",
    "teddy_bear": "deformable_teddy_bear",
}


@dataclass
class DroidDeformablePickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID deformable-object pick-and-place environment."""

    pick_object: str = "cube"
    """Deformable object alias or asset registry name, exposed as ``--pick_object``."""

    embodiment: str = "droid_abs_joint_pos"
    """DROID embodiment registry name, exposed as ``--embodiment``."""


@register_environment
class DroidDeformablePickAndPlaceEnvironment(ArenaEnvironmentFactory[DroidDeformablePickAndPlaceEnvironmentCfg]):
    """Build the fixed-pose PhysX deformable-object example."""

    name = "droid_deformable_pick_and_place"
    _legacy_argparse_cfg_type = DroidDeformablePickAndPlaceEnvironmentCfg

    def build(self, cfg: DroidDeformablePickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.deformable_object import DeformableObject
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        table = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=table,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())
        destination = self.asset_registry.get_asset_by_name("plate_large_vomp_robolab")(instance_name="plate")

        pick_object_registry_name = _PICK_OBJECT_ALIASES.get(cfg.pick_object, cfg.pick_object)
        pick_object = self.asset_registry.get_asset_by_name(pick_object_registry_name)(instance_name="pick_object")
        assert isinstance(
            pick_object, DeformableObject
        ), f"Pick object {cfg.pick_object!r} resolves to {pick_object_registry_name!r}, which is not deformable."
        destination.add_relation(On(table_reference))
        pick_object.add_relation(On(table_reference))
        pick_object.add_relation(NextTo(destination, side=Side.POSITIVE_Y))

        assert cfg.embodiment in {"droid_abs_joint_pos", "droid_differential_ik"}, (
            "The deformable pick-and-place example supports droid_abs_joint_pos and droid_differential_ik, "
            f"got {cfg.embodiment!r}."
        )
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_location=destination,
            background_scene=table,
            episode_length_s=30.0,
            task_description=f"Pick up the {cfg.pick_object.replace('_', ' ')} and place it on the plate.",
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[table, table_reference, light, directional_light, destination, pick_object]),
            task=task,
        )
