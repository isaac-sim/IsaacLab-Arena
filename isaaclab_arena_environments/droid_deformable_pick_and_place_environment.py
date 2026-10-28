# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""DROID deformable-object pick-and-place environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def _configure_newton_deformable_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Couple Newton rigid bodies to the VBD deformable solver."""
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
    from isaaclab_newton.physics import NewtonCfg, NewtonSoftContactCfg, VBDSolverCfg

    physics_cfg = env_cfg.sim.physics
    assert isinstance(physics_cfg, NewtonCfg), "Newton deformable objects require '--presets newton'."
    rigid_solver_cfg = physics_cfg.solver_cfg
    all_environment_bodies = [r"/World/envs/env_[^/]+/.*"]
    deformable_contact_bodies = [
        r"/World/envs/env_[^/]+/Robot/Gripper/Robotiq_2F_85/.*",
        r"/World/envs/env_[^/]+/maple_table_robolab/table(?:/.*)?",
        r"/World/envs/env_[^/]+/plate(?:/.*)?",
    ]
    soft_solver_cfg = VBDSolverCfg(
        iterations=15,
        rigid_body_contact_buffer_size=1024,
        rigid_body_particle_contact_buffer_size=8192,
    )
    rigid_solver_cfg.iterations = 200
    rigid_solver_cfg.ls_iterations = 50
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=rigid_solver_cfg,
                    bodies=all_environment_bodies,
                    include_static_shapes=True,
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=soft_solver_cfg,
                    all_particles=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=deformable_contact_bodies,
                    collide_interval=1,
                ),
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=1.0e3,
            soft_contact_kd=10.0,
            soft_contact_mu=3.0,
        ),
        num_substeps=8,
    )
    return env_cfg


@dataclass
class DroidDeformablePickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID deformable-object pick-and-place environment."""

    pick_object: str = "deformable_cube_newton"
    """Deformable object registry name, exposed as ``--pick_object``."""

    embodiment: str = "droid_abs_joint_pos"
    """DROID embodiment registry name, exposed as ``--embodiment``."""

    teleop_device: str | None = None
    """Teleoperation device registry name, exposed as ``--teleop_device``."""


@register_environment
class DroidDeformablePickAndPlaceEnvironment(ArenaEnvironmentFactory[DroidDeformablePickAndPlaceEnvironmentCfg]):
    """Build the fixed-pose deformable-object example."""

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

        assert cfg.embodiment in {"droid_abs_joint_pos", "droid_differential_ik"}, (
            "The deformable pick-and-place example supports droid_abs_joint_pos and droid_differential_ik, "
            f"got {cfg.embodiment!r}."
        )

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
        destination = self.asset_registry.get_asset_by_name("plate_large_vomp_robolab")(name="plate")
        destination.add_relation(On(table_reference))

        pick_object = self.asset_registry.get_asset_by_name(cfg.pick_object)(name="pick_object")
        assert isinstance(pick_object, DeformableObject), f"Pick object {cfg.pick_object!r} is not a deformable asset."
        pick_object.add_relation(On(table_reference))
        pick_object.add_relation(NextTo(destination, side=Side.POSITIVE_Y))

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )
        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )

        task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_location=destination,
            background_scene=table,
            episode_length_s=30.0,
            placement_consecutive_steps=3,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[table, table_reference, light, directional_light, destination, pick_object]),
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=(
                _configure_newton_deformable_physics if pick_object.physics_preset is PhysicsBackend.NEWTON else None
            ),
            default_physics_backend=pick_object.physics_preset,
        )
