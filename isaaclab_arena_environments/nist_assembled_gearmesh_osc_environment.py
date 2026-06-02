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

        embodiment = self.asset_registry.get_asset_by_name("franka_nist_gear_insertion_osc")(
            enable_cameras=args_cli.enable_cameras,
            concatenate_observation_terms=True,
        )

        # GearInsertionGeometryCfg owns the canonical insertion geometry for the NIST board.
        geometry_cfg = GearInsertionGeometryCfg()

        embodiment.action_config = FrankaNistGearInsertionOscActionsCfg(
            fixed_asset_name=gears_and_base.name,
            peg_offset=geometry_cfg.peg_tip_offset,
        )
        embodiment.observation_config = FrankaNistGearInsertionObservationsCfg(
            fixed_asset_name=gears_and_base.name,
            peg_offset=geometry_cfg.peg_tip_offset,
            fingertip_body_name=embodiment.get_command_body_name(),
            concatenate_observation_terms=embodiment.concatenate_observation_terms,
        )
        embodiment.reward_config = NistGearInsertionOscRewardsCfg(
            gear_name=medium_gear.name,
            board_name=gears_and_base.name,
            peg_offset=geometry_cfg.peg_base_offset,
            held_gear_base_offset=geometry_cfg.held_gear_base_offset,
            gear_peg_height=geometry_cfg.gear_peg_height,
            success_z_fraction=geometry_cfg.success_z_fraction,
            xy_threshold=geometry_cfg.xy_threshold,
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

        task = NistGearInsertionRLTask(
            assembled_board=assembled_board,
            held_gear=medium_gear,
            background_scene=table,
            gear_base_asset=gears_and_base,
            geometry_cfg=geometry_cfg,
            episode_length_s=15.0,
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
