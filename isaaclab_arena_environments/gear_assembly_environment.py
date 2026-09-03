# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton DROID environment for gear insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


TABLE_TOP_Z = 0.071
"""World-space height of the maple tabletop."""

MAPLE_TABLE_TOP_Z = 0.003000684082508087
"""Top of the table geometry in the maple-table USD's local frame."""

GEAR_HALF_HEIGHT = 0.01875
GEAR_INITIAL_POSITION = (0.41, 0.17, TABLE_TOP_Z + GEAR_HALF_HEIGHT + 0.001)
GEAR_BASE_POSITION = (0.55, -0.08, TABLE_TOP_Z + GEAR_HALF_HEIGHT)


@dataclass
class GearAssemblyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Newton DROID gear-assembly environment."""

    teleop_device: str | None = None
    """Arena teleoperation device, such as ``keyboard`` or ``spacemouse``."""

    episode_length_s: float = 70.0
    """Maximum episode duration."""

    def __post_init__(self) -> None:
        assert self.episode_length_s > 0.0, "episode_length_s must be positive"


@register_environment
class GearAssemblyEnvironment(ArenaEnvironmentFactory[GearAssemblyEnvironmentCfg]):
    """Build the DROID task for inserting a medium gear onto its matching peg."""

    name = "gear_assembly"
    _legacy_argparse_cfg_type = GearAssemblyEnvironmentCfg

    def build(self, cfg: GearAssemblyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from registered Arena components."""
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.gear_assembly_task import GearAssemblyTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp
        from isaaclab_arena_environments.gear_assembly_asset_library import (
            make_gear_assembly_base,
            make_gear_assembly_medium_gear,
        )

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        background.set_initial_pose(Pose(position_xyz=(0.0, 0.0, TABLE_TOP_Z - MAPLE_TABLE_TOP_Z)))
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        gear_base = make_gear_assembly_base(Pose(position_xyz=GEAR_BASE_POSITION))
        medium_gear = make_gear_assembly_medium_gear(Pose(position_xyz=GEAR_INITIAL_POSITION))
        insertion_target = ObjectReference(
            name="medium_gear_target",
            prim_path=f"{gear_base.get_prim_path()}/medium_gear_target",
            parent_asset=gear_base,
            object_type=ObjectType.BASE,
        )

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(1800.0)
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        embodiment = self.asset_registry.get_asset_by_name("droid_differential_ik")(enable_cameras=cfg.enable_cameras)
        mdp.configure_droid_for_newton_gear_assembly(embodiment)

        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )
        scene = Scene(
            assets=[
                background,
                table_reference,
                gear_base,
                insertion_target,
                medium_gear,
                light,
                directional_light,
            ]
        )
        task = GearAssemblyTask(
            fixed_asset=gear_base,
            held_asset=medium_gear,
            insertion_target=insertion_target,
            background_scene=background,
            episode_length_s=cfg.episode_length_s,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.gear_assembly_newton_env_cfg_callback,
        )
