# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@functools.lru_cache(maxsize=1)
def procedural_asset_classes() -> tuple[type[Object], type[Object]]:
    """Return ProceduralTable/ProceduralCube, defining them on first use.

    Kept off the module import path so ``isaaclab_arena_environments`` package init
    does not pull ``Object`` → ``pxr`` before ``SimulationApp`` starts.
    """
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.pose import Pose

    class _ProceduralCuboid(Object):
        """Rigid cuboid spawned from a CuboidCfg (no USD)."""

        tags: list[str]
        _default_name: str
        _default_prim: str
        _spawn_cfg: sim_utils.CuboidCfg

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = None,
            initial_pose: Pose | None = None,
        ):
            super().__init__(
                name=instance_name or self._default_name,
                prim_path=prim_path or self._default_prim,
                object_type=ObjectType.RIGID,
                tags=self.tags,
                initial_pose=initial_pose,
                spawner_cfg=self._spawn_cfg,
            )

    class ProceduralTable(_ProceduralCuboid):
        """Kinematic cuboid table (invisible collision surface). Newton-safe, single geometry."""

        tags = ["background", "procedural"]
        object_min_z: float = 0.0
        _default_name = "table"
        _default_prim = "{ENV_REGEX_NS}/table"
        _spawn_cfg = sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visible=False,
        )

    class ProceduralCube(_ProceduralCuboid):
        """Rigid cuboid manipuland (0.2 kg, 5x10x10 cm). Newton-safe, single geometry."""

        tags = ["object", "procedural"]
        _default_name = "object"
        _default_prim = "{ENV_REGEX_NS}/Object"
        _spawn_cfg = sim_utils.CuboidCfg(
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
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
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
            # 50 Hz control, the rate the RL policies for this task were trained at.
            env_cfg_callback=set_control_rate_50hz,
        )
