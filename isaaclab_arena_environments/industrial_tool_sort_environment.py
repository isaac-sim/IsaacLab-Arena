# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Industrial FR3 tool-sorting environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def configure_industrial_tool_sort_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply the benchmark's 50 Hz, ten-substep Newton configuration."""
    env_cfg.sim.dt = 1.0 / 50.0
    env_cfg.sim.render_interval = 1
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.sim.use_newton_actuators = True
    env_cfg.decimation = 1
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            enable_multiccd=True,
            solver="newton",
            integrator="euler",
            nconmax=5000,
            njmax=5000,
            iterations=100,
            ls_iterations=50,
            impratio=20.0,
            cone="elliptic",
            use_mujoco_contacts=True,
        ),
        default_shape_cfg=NewtonShapeCfg(ke=60000.0, kd=500.0, gap=0.002),
        num_substeps=10,
        use_cuda_graph=True,
        debug_mode=False,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.scene.replicate_physics = False
    return env_cfg


@dataclass
class IndustrialToolSortEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the industrial FR3 tool-sorting environment."""

    embodiment: str = "industrial_fr3_robotiq_2f85"
    teleop_device: str | None = None
    top_camera_position: list[float] | None = None
    top_camera_rotation_wxyz: list[float] | None = None


@register_environment
class IndustrialToolSortEnvironment(ArenaEnvironmentFactory[IndustrialToolSortEnvironmentCfg]):
    """Sort four industrial tools into the matching destination-bin regions."""

    name = "vabar_tool_sort__sort_all_newton"
    _legacy_argparse_cfg_type = IndustrialToolSortEnvironmentCfg

    def build(self, cfg: IndustrialToolSortEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the workcell, fixed tool layout, task, and Newton profile."""
        from isaaclab_arena.assets.hdr_image_library import EmptyWarehouseHDRRobolab
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.objects_in_regions_task import ObjectsInRegionsTask
        from isaaclab_arena.utils.pose import Pose

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            initial_pose=Pose(position_xyz=(-0.5, -0.1, 0.912)),
        )
        self._configure_top_camera(embodiment, cfg)

        background = self.asset_registry.get_asset_by_name("industrial__fr3_workcell_table")()
        background.set_initial_pose(Pose(position_xyz=(-0.5, -0.1, 0.912)))
        table = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/industrial__fr3_workcell_table/placement_surface",
            parent_asset=background,
            object_type=ObjectType.BASE,
        )

        source_bin = self.asset_registry.get_asset_by_name("industrial__tool_sort_bin")(
            instance_name="bin1",
            side="source",
            initial_pose=Pose(position_xyz=(0.1, -0.25, 0.8)),
        )
        destination_bin = self.asset_registry.get_asset_by_name("industrial__tool_sort_bin")(
            instance_name="bin2",
            side="destination",
            initial_pose=Pose(position_xyz=(0.1, 0.28, 0.8)),
        )

        tool_specs = (
            (
                "hammer_0",
                "vabar_tool_sort__hammer",
                (0.055071812123060226, -0.2918435335159302, 0.8555317521095276),
                (-0.4180590510368347, -0.4858432114124298, -0.44771748781204224, -0.6234837770462036),
            ),
            (
                "drill_0",
                "vabar_tool_sort__drill",
                (0.22463268041610718, -0.13386666774749756, 0.905187974),
                (-0.1799403727054596, -0.17290718853473663, 0.7333788275718689, 0.6323607563972473),
            ),
            (
                "round_nut_0",
                "vabar_tool_sort__round_nut",
                (0.1308548003435135, -0.18529225885868073, 0.8305647373199463),
                (0.006598883308470249, 0.009126810356974602, 0.9815715551376343, 0.1907627433538437),
            ),
            (
                "clamp_0",
                "vabar_tool_sort__clamp",
                (0.15202291309833527, -0.336628258228302, 0.8376063704490662),
                (-0.014119562692940235, -0.08092429488897324, 0.5865128636360168, -0.805763304233551),
            ),
        )
        tools = []
        for instance_name, registry_name, position, rotation in tool_specs:
            tool = self.asset_registry.get_asset_by_name(registry_name)(
                instance_name=instance_name,
                initial_pose=Pose(position_xyz=position, rotation_xyzw=rotation),
            )
            tools.append(tool)

        shadow_receiver = self.asset_registry.get_asset_by_name("industrial__hdr_shadow_receiver")()
        light = self.asset_registry.get_asset_by_name("light")(hdr=EmptyWarehouseHDRRobolab())
        scene = Scene(
            assets=[
                background,
                shadow_receiver,
                source_bin,
                destination_bin,
                *tools,
                light,
                table,
            ]
        )
        task = ObjectsInRegionsTask(
            object_list=tools,
            region_list=[destination_bin] * len(tools),
            bounds_xyzxyz=[
                (-0.095, 0.035, 0.8, 0.1, 0.28, 0.95),
                (0.1, 0.035, 0.8, 0.295, 0.28, 0.95),
                (-0.095, 0.28, 0.8, 0.1, 0.525, 0.95),
                (0.1, 0.28, 0.8, 0.295, 0.525, 0.95),
            ],
            episode_length_s=192.0,
            task_description="Sort all tools from bin1 into the compartment of bin2 labelled for their type.",
        )
        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=configure_industrial_tool_sort_physics,
        )

    # TODO(qianl, 2026-09-01): Can we use variation instead?
    @staticmethod
    def _configure_top_camera(embodiment, cfg: IndustrialToolSortEnvironmentCfg) -> None:
        """Apply an optional finite top-camera pose override."""
        position_values = cfg.top_camera_position
        rotation_values = cfg.top_camera_rotation_wxyz
        assert (position_values is None) == (
            rotation_values is None
        ), "Top camera position and rotation must be set together."
        if position_values is None or rotation_values is None:
            return
        position = tuple(float(value) for value in position_values)
        rotation = tuple(float(value) for value in rotation_values)
        assert (
            len(position) == 3 and len(rotation) == 4
        ), "Top camera pose must contain three position and four rotation values."
        assert all(math.isfinite(value) for value in (*position, *rotation)), "Top camera pose values must be finite."
        top_camera = embodiment.camera_config.top_camera
        top_camera.offset.pos = position
        top_camera.offset.rot = rotation
