# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Franka volume-deformable cube pick-and-place environment.

Launch a short headless rollout from the development container:

    /isaac-sim/python.sh -m isaaclab_arena.evaluation.policy_runner \
        --headless --policy_type zero_action --num_steps 10 \
        franka_deformable_cube_pick_and_place

The scene uses fixed initial poses and the Newton MJWarp/VBD preset. Deformable Mimic,
policy training, randomized placement, and interactive PhysX mouse manipulation are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class FrankaDeformableCubePickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Franka deformable-cube pick-and-place environment."""


@register_environment
class FrankaDeformableCubePickAndPlaceEnvironment(
    ArenaEnvironmentFactory[FrankaDeformableCubePickAndPlaceEnvironmentCfg]
):
    """Build the fixed-pose Newton deformable-cube example."""

    name = "franka_deformable_cube_pick_and_place"
    _legacy_argparse_cfg_type = FrankaDeformableCubePickAndPlaceEnvironmentCfg

    def build(self, cfg: FrankaDeformableCubePickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
        from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg

        from isaaclab_arena.assets.deformable_object import DeformableObject
        from isaaclab_arena.assets.deformable_spawn import VolumeDeformableMaterial
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.pose import Pose

        table = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        plate_radius = 0.12
        plate_height = 0.02
        destination = Object(
            name="plate",
            object_type=ObjectType.RIGID,
            spawner_cfg=sim_utils.CylinderCfg(
                radius=plate_radius,
                height=plate_height,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.82)),
            ),
            initial_pose=Pose(
                position_xyz=(0.55, -0.25, 0.81),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        destination.usd_path = ""
        destination.bounding_box = AxisAlignedBoundingBox(
            min_point=(-plate_radius, -plate_radius, -plate_height * 0.5),
            max_point=(plate_radius, plate_radius, plate_height * 0.5),
        )

        cube = DeformableObject(
            name="deformable_cube",
            spawner_cfg=MeshCuboidCfg(size=(0.08, 0.08, 0.08)),
            material=VolumeDeformableMaterial(
                youngs_modulus=1.0e5,
                poissons_ratio=0.3,
                density=500.0,
                damping=0.01,
                particle_radius=0.008,
            ),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.15)),
            initial_pose=Pose(position_xyz=(0.48, 0.18, 0.85), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )

        # TODO(qianl): Switch to DROID after a validated Newton-compatible DROID asset exists.
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        task = PickAndPlaceTask(
            pick_up_object=cube,
            destination_location=destination,
            background_scene=table,
            episode_length_s=30.0,
            task_description="Pick up the deformable cube and place it on the plate.",
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[table, light, directional_light, destination, cube]),
            task=task,
            default_physics_preset="newton_mjwarp_vbd",
        )
