# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Cached after first call to :func:`procedural_asset_classes` (Object/pxr stay unloaded until then).
_procedural_table_cls: type[Object] | None = None
_procedural_cube_cls: type[Object] | None = None


def procedural_asset_classes() -> tuple[type[Object], type[Object]]:
    """Return ProceduralTable/ProceduralCube, defining them on first use.

    Kept off the module import path so ``isaaclab_arena_environments`` package init
    (which imports every ``*_environment`` module) does not pull ``Object`` → ``pxr``
    before ``SimulationApp`` starts.
    """
    global _procedural_table_cls, _procedural_cube_cls
    if _procedural_table_cls is not None and _procedural_cube_cls is not None:
        return _procedural_table_cls, _procedural_cube_cls

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.pose import Pose

    table_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        visible=False,
    )

    class ProceduralTable(Object):
        """Kinematic cuboid table (invisible collision surface). Newton-safe, single geometry."""

        tags = ["background", "procedural"]
        object_min_z: float = 0.0

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = None,
            initial_pose: Pose | None = None,
        ):
            resolved_name = instance_name if instance_name is not None else "table"
            resolved_prim = prim_path if prim_path is not None else "{ENV_REGEX_NS}/table"
            super().__init__(
                name=resolved_name,
                prim_path=resolved_prim,
                object_type=ObjectType.RIGID,
                usd_path="",
                initial_pose=initial_pose,
            )

        def _generate_rigid_cfg(self) -> RigidObjectCfg:
            cfg = RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=table_spawn_cfg,
                **self.asset_cfg_addon,
            )
            return self._add_initial_pose_to_cfg(cfg)

    cube_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.05, 0.1, 0.1),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    )

    class ProceduralCube(Object):
        """Rigid cuboid manipuland (0.2 kg, 5x10x10 cm). Newton-safe, single geometry."""

        tags = ["object", "procedural"]

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = None,
            initial_pose: Pose | None = None,
        ):
            resolved_name = instance_name if instance_name is not None else "object"
            resolved_prim = prim_path if prim_path is not None else "{ENV_REGEX_NS}/Object"
            super().__init__(
                name=resolved_name,
                prim_path=resolved_prim,
                object_type=ObjectType.RIGID,
                usd_path="",
                initial_pose=initial_pose,
            )

        def _generate_rigid_cfg(self) -> RigidObjectCfg:
            cfg = RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=cube_spawn_cfg,
                **self.asset_cfg_addon,
            )
            return self._add_initial_pose_to_cfg(cfg)

    _procedural_table_cls = ProceduralTable
    _procedural_cube_cls = ProceduralCube
    return ProceduralTable, ProceduralCube


@dataclass
class DexsuiteLiftEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Dexsuite lift environment."""


@register_environment
class DexsuiteLiftEnvironment(ArenaEnvironmentFactory[DexsuiteLiftEnvironmentCfg]):
    """
    Dexsuite Kuka Allegro lift task; RSL-RL config ``DexsuiteKukaAllegroPPORunnerCfg``.
    The robot picks up a cube and lifts it to a target position.
    """

    name: str = "dexsuite_lift"
    _legacy_argparse_cfg_type = DexsuiteLiftEnvironmentCfg

    def build(self, cfg: DexsuiteLiftEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import math

        import isaaclab_tasks.manager_based.manipulation.dexsuite  # noqa: F401

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import DexsuiteLiftTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        procedural_table_cls, procedural_cube_cls = procedural_asset_classes()

        dexsuite_table = procedural_table_cls()
        dexsuite_table.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.235)))

        manip_object = procedural_cube_cls()
        manip_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(-0.75, -0.1, 0.35),
                position_xyz_max=(-0.35, 0.3, 0.75),
                rpy_min=(-math.pi, -math.pi, -math.pi),
                rpy_max=(math.pi, math.pi, math.pi),
            )
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        embodiment = self.asset_registry.get_asset_by_name("kuka_allegro")(enable_cameras=cfg.enable_cameras)

        scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
        task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

        dexsuite_rl_cfg_entry = (
            "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents."
            "rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg"
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=None,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=dexsuite_rl_cfg_entry,
        )
