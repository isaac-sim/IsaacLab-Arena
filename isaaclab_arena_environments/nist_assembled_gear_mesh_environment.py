# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST gear insertion environment for DROID policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


DROID_EMBODIMENTS = (
    "droid_abs_joint_pos",
    "droid_rel_joint_pos",
    "droid_differential_ik",
)


@dataclass
class NistAssembledGearMeshEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the NIST gear insertion environment."""

    embodiment: str = "droid_abs_joint_pos"
    episode_length_s: float = 15.0


@register_environment
class NistAssembledGearMeshEnvironment(ArenaEnvironmentFactory[NistAssembledGearMeshEnvironmentCfg]):
    """NIST gear insertion environment using existing DROID embodiments."""

    name: str = "nist_assembled_gear_mesh"
    _legacy_argparse_cfg_type = NistAssembledGearMeshEnvironmentCfg

    def build(self, cfg: NistAssembledGearMeshEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        assert (
            cfg.embodiment in DROID_EMBODIMENTS
        ), f"{self.name} only supports existing DROID embodiments {DROID_EMBODIMENTS}, got '{cfg.embodiment}'."

        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.nist_gear_insertion.task import GearInsertionGeometryCfg, NistGearInsertionTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        table = self.asset_registry.get_asset_by_name("table")()
        gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
        gears_and_base.set_initial_pose(
            Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
        )
        medium_gear.set_initial_pose(Pose(position_xyz=(0.50, -0.24, 0.02), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        scene = Scene(assets=[table, medium_gear, gears_and_base, light])
        geometry_cfg = GearInsertionGeometryCfg()
        task = NistGearInsertionTask(
            held_gear=medium_gear,
            background_scene=table,
            gear_base_asset=gears_and_base,
            geometry_cfg=geometry_cfg,
            episode_length_s=cfg.episode_length_s,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=None,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
        )
