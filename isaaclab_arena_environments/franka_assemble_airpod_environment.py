# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""运行示例:

python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 200 \
  pick_and_place_airpod

在指定环境中用指定策略运行给定的步数。
"""


from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class AirPodEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the airpod Maple-table pick-and-place environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_abs_joint_pos"
    hdr: str | None = None
    light_intensity: float = 500.0
    # 待抓取的物体
    pick_up_object: str = "rubiks_cube_hot3d_robolab"   # 需要改成 airpod
    # 目标放置位置
    destination_location: str = "bowl_ycb_robolab" # 需要改成 airpod 盒子
    # 额外干扰物体，放在桌子上，增加环境复杂性，需要改成 airpod 盒盖
    additional_table_objects: list[str] = field(
        default_factory=lambda: [
            "banana_ycb_robolab",
            "chocolate_pudding_ycb_robolab",
            "mug_ycb_robolab",
        ]
    )


@register_environment
class AirPodEnvironment(ArenaEnvironmentFactory[AirPodEnvironmentCfg]):
    """Registered provider for the airpod pick-and-place environment."""

    name: str = "pick_and_place_airpod"
    _legacy_argparse_cfg_type = AirPodEnvironmentCfg

    def build(self, cfg: AirPodEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.pick_up_object)()
        destination_location = self.asset_registry.get_asset_by_name(cfg.destination_location)()

        # Step 2: 桌子为场景中的固定物体
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        # 桌子上放两个物体
        pick_up_object.add_relation(On(table_reference))
        destination_location.add_relation(On(table_reference))

        # 桌子上的额外物体
        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in cfg.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        # Step 4: 设置机械臂
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )

        # Step 5: 组装场景（不包括机械臂）
        scene = Scene(
            assets=[
                background,
                light,
                directional_light,
                pick_up_object,
                destination_location,
                table_reference,
                *additional_table_objects,
            ]
        )

        # Step 6: 定义任务
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=20.0,
        )

        # 设置视口摄像机以匹配 robolab droid 视图
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        # Step 7: 组装环境，包括机械臂、场景、任务和视口配置
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

    # TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this CLI-only option when teleoperation runners
    # receive typed configuration instead of the environment subparser namespace.
    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        # Consumed directly by teleop.py and record_demos.py, not by build(cfg).
        parser.add_argument("--teleop_device", type=str, default=None)
