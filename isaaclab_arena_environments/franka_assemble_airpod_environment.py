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
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class FrankaAssembleAirpodEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the franka assemble airpod environment."""

    enable_cameras: bool = False
    object: str = "rubiks_cube_hot3d_robolab"
    destination: str = "bowl_ycb_robolab"
    embodiment: str = "franka_ik"
    teleop_device: str | None = None


@register_environment
class FrankaAssembleAirpodEnvironment(ArenaEnvironmentFactory[FrankaAssembleAirpodEnvironmentCfg]):

    name: str = "pick_and_place_airpod_maple_table"
    _legacy_argparse_cfg_type = FrankaAssembleAirpodEnvironmentCfg

    def build(self, cfg: FrankaAssembleAirpodEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()

        # 桌子是环境中的锚点
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
        )
        table_reference.add_relation(IsAnchor())

        # 定义机器人、机械臂
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        # 定义待抓取的物体和目标放置位置
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
        pick_up_object.add_relation(On(table_reference, clearance_m=0.02))
        pick_up_object.add_relation(AtPosition(x=0.4, y=0.0))

        # 定义目标放置位置
        destination_location = self.asset_registry.get_asset_by_name(cfg.destination)()
        destination_location.add_relation(On(table_reference, clearance_m=0.02))
        destination_location.add_relation(AtPosition(x=-0.4, y=0.0))

        # 遥操作设备
        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)()
            if cfg.teleop_device is not None
            else None
        )

        # 构建场景
        scene = Scene(assets=[background, table_reference, pick_up_object, destination_location])

        # 定义任务
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
        )

        # 把场景、任务、机器人、遥操作设备组合成环境
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
