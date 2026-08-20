# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""DROID PhysX deformable-cube pick-and-place environment.

Launch a short headless rollout from the development container:

    /isaac-sim/python.sh -m isaaclab_arena.evaluation.policy_runner \
        --headless --policy_type zero_action --num_steps 10 \
        droid_deformable_cube_pick_and_place
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class DroidDeformableCubePickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the DROID deformable-cube pick-and-place environment."""

    embodiment: str = "droid_abs_joint_pos"
    """DROID embodiment registry name, exposed as ``--embodiment`` on the legacy CLI."""


@register_environment
class DroidDeformableCubePickAndPlaceEnvironment(
    ArenaEnvironmentFactory[DroidDeformableCubePickAndPlaceEnvironmentCfg]
):
    """Build the fixed-pose PhysX deformable-cube example."""

    name = "droid_deformable_cube_pick_and_place"
    _legacy_argparse_cfg_type = DroidDeformableCubePickAndPlaceEnvironmentCfg

    def build(self, cfg: DroidDeformableCubePickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
        from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg

        from isaaclab_arena.assets.deformable_object import DeformableObject
        from isaaclab_arena.assets.deformable_spawn import VolumeDeformableMaterial
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        table = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        destination = self.asset_registry.get_asset_by_name("plate_large_vomp_robolab")(
            instance_name="plate",
            initial_pose=Pose(position_xyz=(0.55, -0.25, 0.02), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )

        cube = DeformableObject(
            name="deformable_cube",
            spawner_cfg=MeshCuboidCfg(size=(0.15, 0.04, 0.04)),
            material=VolumeDeformableMaterial(
                youngs_modulus=8.0e4,
                poissons_ratio=0.25,
                density=300.0,
                particle_radius=0.01,
            ),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            initial_pose=Pose(position_xyz=(0.48, 0.18, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )

        assert cfg.embodiment in {"droid_abs_joint_pos", "droid_differential_ik"}, (
            "The deformable-cube example supports droid_abs_joint_pos and droid_differential_ik, "
            f"got {cfg.embodiment!r}."
        )
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )
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
        )
