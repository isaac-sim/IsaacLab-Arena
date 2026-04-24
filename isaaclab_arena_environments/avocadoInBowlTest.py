# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hand-authored test for the `In` relation.

Scene: maple table, bowl on table, avocado **inside bowl** (initial state).
No task — we only care that the relation solver can satisfy ``In(bowl)``
for the avocado. Inspect the printed init_state positions to confirm
that the avocado's XY lands within the bowl's XY footprint; Z is left
free so gravity can drop the avocado into the bowl on the first tick.

Run:
    /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
        --policy_type zero_action --num_steps 5 --num_envs 1 \
        avocadoInBowlTest --embodiment no_embodiment
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class AvocadoInBowlTestEnvironment(ExampleEnvironmentBase):
    """Bowl on the table, avocado initially *in* the bowl."""

    name: str = "avocadoInBowlTest"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import In, IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        background.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068)))
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        light = self.asset_registry.get_asset_by_name("light")()

        tabletop_anchor = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        tabletop_anchor.add_relation(IsAnchor())

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

        bowl_obj = self.asset_registry.get_asset_by_name("bowl_ycb_robolab")()
        avocado_obj = self.asset_registry.get_asset_by_name("avocado01_fruits_veggies_robolab")()

        _tbl_bbox = tabletop_anchor.get_world_bounding_box()
        _tbl_min_xyz = [float(_tbl_bbox.min_point[0, i]) for i in range(3)]
        _tbl_max_xyz = [float(_tbl_bbox.max_point[0, i]) for i in range(3)]
        _tbl_margin = 0.05

        # Bowl sits on the tabletop.
        bowl_obj.add_relation(On(tabletop_anchor, clearance_m=0.02))
        bowl_obj.add_relation(
            PositionLimits(
                x_min=_tbl_min_xyz[0] + _tbl_margin,
                x_max=_tbl_max_xyz[0] - _tbl_margin,
                y_min=_tbl_min_xyz[1] + _tbl_margin,
                y_max=_tbl_max_xyz[1] - _tbl_margin,
            )
        )

        # Avocado initially inside the bowl — In clamps XY to the bowl's
        # footprint, Z is unconstrained so gravity drops it on the first
        # physics tick.
        avocado_obj.add_relation(In(bowl_obj))

        scene = Scene(assets=[background, ground_plane, light, tabletop_anchor, bowl_obj, avocado_obj])

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="no_embodiment")
