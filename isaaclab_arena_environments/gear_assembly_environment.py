# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gear Assembly scene using Arena's existing Droid embodiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.tasks.gear_assembly.specs import (
    DROID_GEAR_ASSEMBLY_EMBODIMENTS,
    DroidGearAssemblyEmbodiment,
    GearAssemblyMode,
    PhysicsBackend,
    gear_pose_for_mode,
    get_droid_robot_spec,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.tasks.gear_assembly.task import GearAssemblyTask


@dataclass
class GearAssemblyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Arena Droid Gear Assembly environment."""

    embodiment: DroidGearAssemblyEmbodiment = "droid_abs_joint_pos"
    mode: GearAssemblyMode = "play"
    physics_backend: PhysicsBackend = "newton"

    def __post_init__(self) -> None:
        assert (
            self.embodiment in DROID_GEAR_ASSEMBLY_EMBODIMENTS
        ), f"Gear Assembly is Droid-only; got embodiment={self.embodiment!r}"


@register_environment
class GearAssemblyEnvironment(ArenaEnvironmentFactory[GearAssemblyEnvironmentCfg]):
    """Arena Gear Assembly factory using the existing Droid robot."""

    name = "gear_assembly"
    _legacy_argparse_cfg_type = GearAssemblyEnvironmentCfg

    def build(self, cfg: GearAssemblyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object_library import DomeLight
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.gear_assembly.assets import (
            make_factory_gear_base,
            make_factory_gear_large,
            make_factory_gear_medium,
            make_factory_gear_small,
            make_ground,
        )
        from isaaclab_arena.tasks.gear_assembly.task import GearAssemblyTask

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.observation_config = None
        robot_spec = get_droid_robot_spec()
        gear_pose = gear_pose_for_mode(cfg.mode)

        assets = [
            make_ground(),
            make_factory_gear_base(gear_pose),
            make_factory_gear_small(gear_pose),
            make_factory_gear_medium(gear_pose),
            make_factory_gear_large(gear_pose),
            DomeLight(
                instance_name="light",
                prim_path="/World/light",
                spawner_cfg=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
            ),
        ]

        task = GearAssemblyTask(robot_spec=robot_spec, mode=cfg.mode)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=assets),
            task=task,
            env_cfg_callback=_make_env_cfg_callback(cfg, task),
        )


def _make_env_cfg_callback(cfg: GearAssemblyEnvironmentCfg, task: GearAssemblyTask):
    def gear_assembly_env_cfg_callback(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        from isaaclab_physx.physics import PhysxCfg

        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

        env_cfg.episode_length_s = 6.66
        env_cfg.viewer.eye = (3.5, 3.5, 3.5)
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        env_cfg.sim.dt = 1.0 / 120.0

        if cfg.physics_backend == "newton":
            env_cfg.sim.physics = ArenaPhysicsCfg().newton
            env_cfg.scene.replicate_physics = True
        elif cfg.physics_backend == "physx":
            env_cfg.sim.physics = PhysxCfg(
                gpu_collision_stack_size=2**30,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
            )
            env_cfg.scene.replicate_physics = False
        else:
            raise ValueError(f"Unsupported Gear Assembly physics backend: {cfg.physics_backend}")

        for attr_name, value in task.runtime_env_attrs().items():
            setattr(env_cfg, attr_name, value)
        return env_cfg

    return gear_assembly_env_cfg_callback
