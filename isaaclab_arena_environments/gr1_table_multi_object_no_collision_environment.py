# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Table + multi-object no-overlap environment with a robot (e.g. GR1).
No task -- suitable for policy_runner with zero_action or any policy.

Supports two placement modes via ``--mode``:

* **homogeneous** (default): each object is a regular Object — same in all envs.
* **heterogeneous**: objects are wrapped in ``RigidObjectSet`` for per-env variance.

Both modes use the office table by default. Use ``--objects`` to override object
lists for controlled experiments.

Example (--viz kit enables the Kit visualizer, --episode_length_s triggers periodic resets):

  # Homogeneous (default) — YCB objects
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py --viz kit --policy_type zero_action --num_steps 500 \\
    --num_envs 16 --env_spacing 4.0 --enable_cameras \\
    gr1_table_multi_object_no_collision --embodiment gr1_joint --episode_length_s 4.0

  # Heterogeneous — robolab objects in RigidObjectSet
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py --viz kit --policy_type zero_action --num_steps 500 \\
    --num_envs 16 --env_spacing 4.0 --enable_cameras \\
    gr1_table_multi_object_no_collision --embodiment gr1_joint --episode_length_s 4.0 --mode heterogeneous
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

DEFAULT_TABLE_OBJECTS = [
    "cracker_box",
    "sugar_box",
    "tomato_soup_can",
    "dex_cube",
    "power_drill",
    "red_container",
]
# NOTE: The gradient-based solver does not guarantee collision-free placement for all
# objects. Better initialization strategies and constraining unchanged pose dimensions
# are needed in the near future.

# -- Heterogeneous mode default object sets ----------------------------------
# Each entry is a multi-variant RigidObjectSet — each env gets a different
# variant. Objects sourced from het-viz branch gif capture script.
HETERO_VARIANT_SETS = {
    "bottles": [
        "mustard_bottle_hope_robolab",
        "milk_carton_hope_robolab",
        "orange_juice_carton_hope_robolab",
        "parmesan_cheese_canister_hope_robolab",
    ],
    "cans": [
        "alphabet_soup_can_hope_robolab",
        "canned_peaches_hope_robolab",
        "corn_can_hope_robolab",
        "tomato_sauce_can_hope_robolab",
        "pineapple_slices_can_hope_robolab",
        "green_beans_can_hope_robolab",
    ],
    "tools": [
        "spoon_handal_robolab",
        "spoon_1_handal_robolab",
        "spoon_2_handal_robolab",
        "measuring_spoon_handal_robolab",
    ],
    "boxes": [
        "popcorn_box_hope_robolab",
        "chocolate_pudding_mix_hope_robolab",
        "macaroni_and_cheese_hope_robolab",
        "granola_bars_hope_robolab",
    ],
}

HETERO_FIXED_OBJECTS = [
    ("banana_ycb_robolab", 0.5, -0.15),
    ("lime01_fruits_veggies_robolab", 0.5, 0.15),
]


@register_environment
class GR1TableMultiObjectNoCollisionEnvironment(ExampleEnvironmentBase):
    """
    Table-based scene with multiple objects (On(table) + built-in no-overlap) and a robot.
    Layout is solved by ArenaEnvBuilder default relation solving; reset uses asset events.

    Supports ``--mode homogeneous`` (default) and ``--mode heterogeneous`` for
    inter-environment object variance via ``RigidObjectSet``.
    """

    name: str = "gr1_table_multi_object_no_collision"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        mode = getattr(args_cli, "mode", "homogeneous")
        enable_cameras = getattr(args_cli, "enable_cameras", False)
        camera_offset = Pose(
            position_xyz=(0.12515, 0.0, 0.06776),
            rotation_xyzw=(0.11204, -0.17712, -0.79108, 0.57469),
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=enable_cameras,
            camera_offset=camera_offset,
            use_tiled_camera=(getattr(args_cli, "num_envs", 1) > 1),
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(1.2, 0.0, 0.995),
                rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
            )
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        table_background = self.asset_registry.get_asset_by_name("office_table")()
        tabletop_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
            parent_asset=table_background,
        )
        tabletop_reference.add_relation(IsAnchor())

        object_names = getattr(args_cli, "objects", None)
        if mode == "heterogeneous":
            placeable_assets = self._build_heterogeneous_objects(tabletop_reference, object_names)
        else:
            placeable_assets = self._build_homogeneous_objects(tabletop_reference, object_names)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(
            assets=[
                ground_plane,
                table_background,
                tabletop_reference,
                *placeable_assets,
                light,
            ]
        )

        episode_length_s = args_cli.episode_length_s
        env_cfg_callback = None
        if episode_length_s is not None:

            def _enable_periodic_reset(cfg):
                import isaaclab.envs.mdp as mdp_isaac_lab
                from isaaclab.managers import TerminationTermCfg

                cfg.episode_length_s = episode_length_s
                cfg.terminations.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
                return cfg

            env_cfg_callback = _enable_periodic_reset

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
            force_convex_hull=(mode == "heterogeneous"),
        )
        return isaaclab_arena_environment

    def _build_homogeneous_objects(self, tabletop_reference, object_names=None):
        """Build placeable objects for homogeneous mode (same objects in all envs).

        Each object is a regular Object instance — identical across environments.
        """
        from isaaclab_arena.relations.relations import On

        names = object_names or DEFAULT_TABLE_OBJECTS
        placeable_assets = []
        for name in names:
            obj = self.asset_registry.get_asset_by_name(name)()
            obj.add_relation(On(tabletop_reference, clearance_m=0.001))
            placeable_assets.append(obj)
        return placeable_assets

    def _build_heterogeneous_objects(self, tabletop_reference, object_names=None):
        """Build placeable objects for heterogeneous mode.

        When --objects is provided, each object becomes a single-variant RigidObjectSet.
        Otherwise, uses HETERO_FIXED_OBJECTS (pinned fruits) + HETERO_VARIANT_SETS
        (multi-variant sets from het-viz branch).
        """
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.relations.relations import AtPosition, On

        if object_names:
            print(
                "Warning: --objects with --mode heterogeneous wraps each object as a "
                "single-variant set (no per-env variance). Use default sets for true heterogeneity."
            )
            placeable_assets = []
            for name in object_names:
                obj = self.asset_registry.get_asset_by_name(name)()
                obj_set = RigidObjectSet(name=name, objects=[obj])
                obj_set.add_relation(On(tabletop_reference, clearance_m=0.001))
                placeable_assets.append(obj_set)
        else:
            placeable_assets = []
            for name, x, y in HETERO_FIXED_OBJECTS:
                obj = self.asset_registry.get_asset_by_name(name)()
                obj.add_relation(On(tabletop_reference, clearance_m=0.001))
                obj.add_relation(AtPosition(x=x, y=y))
                placeable_assets.append(obj)

            for set_name, variant_names in HETERO_VARIANT_SETS.items():
                members = [self.asset_registry.get_asset_by_name(n)() for n in variant_names]
                obj_set = RigidObjectSet(name=set_name, objects=members)
                obj_set.add_relation(On(tabletop_reference, clearance_m=0.001))
                placeable_assets.append(obj_set)

        return placeable_assets

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--objects",
            nargs="*",
            type=str,
            default=None,
            help=(
                "Object names (works in both modes). "
                f"Homo default: {' '.join(DEFAULT_TABLE_OBJECTS)}; "
                f"Hetero default: {', '.join(HETERO_VARIANT_SETS.keys())} variant sets"
            ),
        )
        parser.add_argument("--embodiment", type=str, default="gr1_joint", help="Robot embodiment to use")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
        parser.add_argument(
            "--episode_length_s",
            type=float,
            default=None,
            help="Episode length in seconds. Enables time_out termination so objects are re-placed on reset.",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="homogeneous",
            choices=["homogeneous", "heterogeneous"],
            help="Placement mode: 'homogeneous' (same objects everywhere) or 'heterogeneous' (per-env variants).",
        )
