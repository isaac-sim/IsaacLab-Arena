# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID Maple-table environment for homogeneous and heterogeneous placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


HOMOGENEOUS_OBJECTS = [
    "orange_01_fruits_veggies_robolab",
    "ketchup_bottle_hope_robolab",
    "alphabet_soup_can_hope_robolab",
    "spoon_handal_robolab",
    "sugar_box_ycb_robolab",
]

HETEROGENEOUS_VARIANT_SETS = {
    "fruits": [
        "orange_01_fruits_veggies_robolab",
        "lemon_02_fruits_veggies_robolab",
        "lemon_01_fruits_veggies_robolab",
        "banana_ycb_robolab",
    ],
    "bottles": [
        "ketchup_bottle_hope_robolab",
        "mustard_bottle_hope_robolab",
        "mayonnaise_bottle_hope_robolab",
        "bbq_sauce_bottle_hope_robolab",
    ],
    "cans": [
        "alphabet_soup_can_hope_robolab",
        "canned_peaches_hope_robolab",
        "corn_can_hope_robolab",
        "tomato_sauce_can_hope_robolab",
    ],
    "tools": [
        "spoon_handal_robolab",
        "spoon_1_handal_robolab",
        "spoon_handal_robolab",
        "measuring_spoon_handal_robolab",
    ],
    "boxes": [
        "popcorn_box_hope_robolab",
        "chocolate_pudding_mix_hope_robolab",
    ],
}


@dataclass
class DroidTableMultiObjectPlacementEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID Maple-table placement environment."""

    embodiment: str = "droid_abs_joint_pos"
    light_intensity: float = 500.0
    episode_length_s: float | None = None
    mode: str = "homogeneous"
    """Placement mode, either ``homogeneous`` or ``heterogeneous``."""

    def __post_init__(self) -> None:
        assert self.mode in {"homogeneous", "heterogeneous"}, f"Unsupported placement mode: {self.mode}"


@register_environment
class DroidTableMultiObjectPlacementEnvironment(ArenaEnvironmentFactory[DroidTableMultiObjectPlacementEnvironmentCfg]):
    """Place homogeneous or heterogeneous objects on a Maple table."""

    name: str = "droid_table_multi_object_placement"
    _legacy_argparse_cfg_type = DroidTableMultiObjectPlacementEnvironmentCfg

    def build(self, cfg: DroidTableMultiObjectPlacementEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        if cfg.mode == "heterogeneous":
            placeable_assets = self._build_heterogeneous_objects(table_reference)
        else:
            placeable_assets = self._build_homogeneous_objects(table_reference)

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )

        scene = Scene(
            assets=[
                background,
                light,
                directional_light,
                table_reference,
                *placeable_assets,
            ]
        )

        episode_length_s = cfg.episode_length_s

        def _configure_viewer_and_timeout(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.9, 0.0, 1.6), lookat=(0.35, 0.0, 0.0))
            if episode_length_s is not None:
                import isaaclab.envs.mdp as mdp_isaac_lab
                from isaaclab.managers import TerminationTermCfg

                env_cfg.episode_length_s = episode_length_s
                env_cfg.terminations.time_out = TerminationTermCfg(
                    func=mdp_isaac_lab.time_out,
                    time_out=True,
                )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            env_cfg_callback=_configure_viewer_and_timeout,
        )

    def _build_homogeneous_objects(self, table_reference: ObjectReference) -> list[Object]:
        """Build the same set of registered objects in every environment."""
        from isaaclab_arena.relations.relations import On

        objects = self._build_registered_objects(HOMOGENEOUS_OBJECTS)
        for obj in objects:
            obj.add_relation(On(table_reference))
        return objects

    def _build_heterogeneous_objects(self, table_reference: ObjectReference) -> list[Object]:
        """Build per-environment object sets."""
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.relations.relations import On

        placeable_assets = []
        for set_name, variant_names in HETEROGENEOUS_VARIANT_SETS.items():
            members = self._build_registered_objects(variant_names)
            object_set = RigidObjectSet(name=set_name, objects=members)
            object_set.add_relation(On(table_reference))
            placeable_assets.append(object_set)
        return placeable_assets

    def _build_registered_objects(self, names: list[str]) -> list[Object]:
        """Build registered objects by name."""
        return [self.asset_registry.get_asset_by_name(name)() for name in names]
