# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST assembled board gear-mesh environment with operational-space torque control."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

_FRANKA_NIST_GEAR_INSERTION_OSC_EMBODIMENT = "franka_nist_gear_insertion_osc"
_PEG_TIP_OFFSET = (0.02025, 0.0, 0.025)
_PEG_BASE_OFFSET = (0.02025, 0.0, 0.0)
_GEAR_PEG_HEIGHT = 0.02
_SUCCESS_Z_FRACTION = 0.20
_XY_THRESHOLD = 0.0025
_EPISODE_LENGTH_S = 15.0


@register_environment
class NISTAssembledGearMeshOSCEnvironment(ExampleEnvironmentBase):
    """NIST gear insertion using OSC torque control and assembly-style observations."""

    name: str = "nist_assembled_gear_mesh_osc"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        import isaaclab_arena_environments.mdp as mdp
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.nist_gear_insertion.task import GearInsertionGeometryCfg, NistGearInsertionRLTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.nist_gear_insertion.franka_osc_cfg import (
            FrankaNistGearInsertionObservationsCfg,
            FrankaNistGearInsertionOscActionsCfg,
        )
        from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_rewards import NistGearInsertionOscRewardsCfg

        table = self.asset_registry.get_asset_by_name("table")()
        assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
        gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        embodiment = self.asset_registry.get_asset_by_name(_FRANKA_NIST_GEAR_INSERTION_OSC_EMBODIMENT)(
            enable_cameras=args_cli.enable_cameras,
            concatenate_observation_terms=True,
        )
        embodiment.action_config = FrankaNistGearInsertionOscActionsCfg(
            fixed_asset_name=gears_and_base.name,
            peg_offset=_PEG_TIP_OFFSET,
        )
        embodiment.observation_config = FrankaNistGearInsertionObservationsCfg(
            fixed_asset_name=gears_and_base.name,
            peg_offset=_PEG_TIP_OFFSET,
            fingertip_body_name=embodiment.get_command_body_name(),
            concatenate_observation_terms=embodiment.concatenate_observation_terms,
        )
        embodiment.reward_config = NistGearInsertionOscRewardsCfg(
            gear_name=medium_gear.name,
            board_name=gears_and_base.name,
            peg_offset=_PEG_BASE_OFFSET,
            held_gear_base_offset=_PEG_BASE_OFFSET,
            gear_peg_height=_GEAR_PEG_HEIGHT,
            success_z_fraction=_SUCCESS_Z_FRACTION,
            xy_threshold=_XY_THRESHOLD,
        )
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
        assembled_board.set_initial_pose(
            Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
        )
        medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        gears_and_base.set_initial_pose(
            Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
        )
        scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

        geometry_cfg = GearInsertionGeometryCfg(
            peg_offset_from_board=list(_PEG_BASE_OFFSET),
            peg_offset_for_obs=list(_PEG_TIP_OFFSET),
            success_z_fraction=_SUCCESS_Z_FRACTION,
            xy_threshold=_XY_THRESHOLD,
        )

        task = NistGearInsertionRLTask(
            assembled_board=assembled_board,
            held_gear=medium_gear,
            background_scene=table,
            gear_base_asset=gears_and_base,
            geometry_cfg=geometry_cfg,
            episode_length_s=_EPISODE_LENGTH_S,
            grasp_cfg=embodiment.get_gear_insertion_grasp_config(),
            fingertip_body_name=embodiment.get_command_body_name(),
            enable_randomization=True,
            disable_success_termination=args_cli.disable_success_termination,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )
        parser.add_argument(
            "--disable_success_termination",
            action="store_true",
            help="Disable success termination during training.",
        )
        parser.add_argument(
            "--rl_training_mode",
            dest="disable_success_termination",
            action="store_true",
            help="Alias for --disable_success_termination.",
        )
