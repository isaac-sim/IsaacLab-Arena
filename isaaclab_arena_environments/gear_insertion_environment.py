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


MAPLE_TABLE_ALIGNMENT_Z = 0.068
"""Maple-table root height aligned with the fixed DROID workcell."""

GEAR_INITIAL_XY = (0.41, 0.17)
GEAR_BASE_XY = (0.55, -0.08)
DROID_GEAR_APPROACH_JOINT_POSITIONS = {
    "panda_joint1": 1.4302369,
    "panda_joint2": -0.3727887,
    "panda_joint3": -1.0313725,
    "panda_joint4": -2.5031810,
    "panda_joint5": -0.4169911,
    "panda_joint6": 2.2608662,
    "panda_joint7": 0.4437634,
}
"""DROID arm pose above the source gear."""


@dataclass
class GearInsertionEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Newton DROID gear insertion environment."""

    hdr: str | None = "empty_warehouse_robolab"
    """Registered HDR background, or None for uniform dome lighting."""

    teleop_device: str | None = None
    """Arena teleoperation device, such as ``keyboard`` or ``spacemouse``."""

    episode_length_s: float = 70.0
    """Maximum episode duration."""

    def __post_init__(self) -> None:
        assert self.episode_length_s > 0.0, "episode_length_s must be positive"


@register_environment
class GearInsertionEnvironment(ArenaEnvironmentFactory[GearInsertionEnvironmentCfg]):
    """Build the DROID task for inserting a medium gear onto its matching peg."""

    name = "gear_insertion"
    _legacy_argparse_cfg_type = GearInsertionEnvironmentCfg

    def build(self, cfg: GearInsertionEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from registered Arena components."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On
        from isaaclab_arena.relations.validation.types import PlacementCheck
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.gear_insertion_task import GearInsertionTask
        from isaaclab_arena.utils.physics_backend import PhysicsBackend
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.gear_insertion import gear_insertion_newton_env_cfg_callback

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        background.set_initial_pose(Pose(position_xyz=(0.0, 0.0, MAPLE_TABLE_ALIGNMENT_Z)))
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        gear_base = self.asset_registry.get_asset_by_name("gear_insertion_base")()
        gear_base.add_relation(On(table_reference, clearance_m=0.0))
        gear_base.add_relation(AtPosition(x=GEAR_BASE_XY[0], y=GEAR_BASE_XY[1]))

        medium_gear = self.asset_registry.get_asset_by_name("gear_insertion_medium_gear")()
        medium_gear.add_relation(On(table_reference, clearance_m=0.001))
        medium_gear.add_relation(AtPosition(x=GEAR_INITIAL_XY[0], y=GEAR_INITIAL_XY[1]))
        insertion_target = ObjectReference(
            name="medium_gear_target",
            prim_path=f"{gear_base.get_prim_path()}/medium_gear_target",
            parent_asset=gear_base,
            object_type=ObjectType.BASE,
        )

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(1800.0)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        embodiment = self.asset_registry.get_asset_by_name("droid_differential_ik")(enable_cameras=cfg.enable_cameras)
        embodiment.set_joint_initial_pos(DROID_GEAR_APPROACH_JOINT_POSITIONS)

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
        task = GearInsertionTask(
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
            env_cfg_callback=gear_insertion_newton_env_cfg_callback,
            default_physics_backend=PhysicsBackend.NEWTON,
            placer_params=ObjectPlacerParams(
                enabled_checks={
                    PlacementCheck.NO_OVERLAP,
                    PlacementCheck.ON_RELATION,
                },
            ),
        )
