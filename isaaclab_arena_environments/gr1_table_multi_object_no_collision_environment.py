# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Table + multi-object NoCollision environment.

Uses an office table as the support surface, spawns multiple objects with On(table)
and pairwise NoCollision (relation solver places them). Includes a robot (e.g. GR1)
in the scene. No task — suitable for policy_runner with zero_action or any policy.

Run via policy_runner, e.g.:

  python isaaclab_arena/evaluation/policy_runner.py \\
    --policy_type zero_action \\
    --num_steps 2000 \\
    --num_envs 4 \\
    --env_spacing 5.0 \\
    --enable_cameras \\
    --environment isaaclab_arena_environments.gr1_table_multi_object_no_collision_environment:GR1TableMultiObjectNoCollisionEnvironment \\
    gr1_table_multi_object_no_collision \\
    --embodiment gr1_joint
"""

import argparse
from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING
from isaaclab.managers import EventTermCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.apply_layout import apply_layout_per_env
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Small objects get a larger NoCollision margin to reduce bumping from contact resolution.
SMALL_TABLE_OBJECTS = frozenset({"dex_cube", "red_cube", "green_cube"})

# Default: 7 table-suitable objects (On table + pairwise NoCollision); fewer objects ease layout optimization.
DEFAULT_TABLE_OBJECTS = [
    "cracker_box",
    "mustard_bottle",
    "sugar_box",
    "tomato_soup_can",
    "mug",
    "brown_box",
    "dex_cube",
]


@configclass
class TableLayoutEventCfg:
    """Event that applies one precomputed layout per env at reset (On table + NoCollision)."""

    apply_table_layout: EventTermCfg = MISSING

    def __init__(self, layouts: list[dict[str, tuple[float, float, float]]], object_names: list[str]):
        self.apply_table_layout = EventTermCfg(
            func=apply_layout_per_env,
            mode="reset",
            params={"layouts": layouts, "object_names": object_names},
        )


class TableMultiObjectLayoutTask(TaskBase):
    """Task that precomputes one layout per env (On + NoCollision) and applies it at reset.

    NoCollision uses rotation-invariant XY (bounding-circle distance) so random yaw
    at apply time stays safe. One layout per env (different seeds) for variety.
    """

    name = "table_multi_object_layout"

    def __init__(
        self,
        objects_with_relations: list,
        placeable_assets: list,
        num_envs: int,
    ):
        super().__init__()
        self._object_names = [obj.name for obj in placeable_assets]
        solver_params = RelationSolverParams(verbose=True, max_iters=600)
        self._layouts = []
        print(f"[TableMultiObjectLayoutTask] Precomputing {num_envs} table layouts (rotation-invariant NoCollision)...")
        # Accept layout when loss <= this. Solver stops early at good_enough_loss=1.0,
        # so accept placements that meet that (constraints "good enough").
        max_acceptable_loss = 1.0
        for seed in range(num_envs):
            params_seed = ObjectPlacerParams(
                apply_positions_to_objects=False,
                placement_seed=seed,
                max_placement_attempts=20,
                solver_params=solver_params,
                verbose=False,
                max_acceptable_loss=max_acceptable_loss,
            )
            placer_seed = ObjectPlacer(params=params_seed)
            result = placer_seed.place(objects=objects_with_relations)
            layout = {obj.name: result.positions[obj] for obj in placeable_assets}
            self._layouts.append(layout)
            msg = f"  Layout {seed + 1}/{num_envs} done (attempts={result.attempts}, success={result.success}"
            if not result.success:
                msg += f", best_loss={result.final_loss:.2e}"
            print(msg + ")")
        print("[TableMultiObjectLayoutTask] Precompute finished.")
        self._events_cfg = TableLayoutEventCfg(layouts=self._layouts, object_names=self._object_names)

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> Any:
        return None

    def get_events_cfg(self) -> TableLayoutEventCfg:
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))


class GR1TableMultiObjectNoCollisionEnvironment(ExampleEnvironmentBase):
    """
    Table-based scene with multiple objects (On(table) + NoCollision) and a robot.

    Register and run via:
      --environment isaaclab_arena_environments.gr1_table_multi_object_no_collision_environment:GR1TableMultiObjectNoCollisionEnvironment
      gr1_table_multi_object_no_collision --objects cracker_box mug tomato_soup_can --embodiment gr1_joint
    """

    name = "gr1_table_multi_object_no_collision"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, NoCollision, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        enable_cameras = getattr(args_cli, "enable_cameras", False)
        camera_offset = Pose(
            position_xyz=(0.12515, 0.0, 0.06776),
            rotation_wxyz=(0.57469, 0.11204, -0.17712, -0.79108),
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=enable_cameras,
            camera_offset=camera_offset,
            use_tiled_camera=(getattr(args_cli, "num_envs", 1) > 1),
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(1.2, 0.0, 0.995),
                rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068),
            )
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        table_background = self.asset_registry.get_asset_by_name("office_table")()
        light = self.asset_registry.get_asset_by_name("light")()

        tabletop_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
            parent_asset=table_background,
        )
        tabletop_reference.add_relation(IsAnchor())

        object_names = getattr(args_cli, "objects", None) or DEFAULT_TABLE_OBJECTS
        placeable_assets = []
        for name in object_names:
            obj = self.asset_registry.get_asset_by_name(name)()
            obj.add_relation(On(tabletop_reference))
            placeable_assets.append(obj)
        # Horizontal gap (X,Y only) so collision meshes don't touch; too small and physics
        # resolves overlap with impulses and objects "fly away". Z is no-overlap only.
        # Use a larger margin for pairs involving small objects to reduce bumping.
        min_separation_m = 0.05
        min_separation_small_m = 0.07
        for i, obj_a in enumerate(placeable_assets):
            for obj_b in placeable_assets[i + 1 :]:
                margin = (
                    min_separation_small_m
                    if (obj_a.name in SMALL_TABLE_OBJECTS or obj_b.name in SMALL_TABLE_OBJECTS)
                    else min_separation_m
                )
                obj_a.add_relation(NoCollision(obj_b, min_separation_m=margin))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        num_envs = getattr(args_cli, "num_envs", 1)
        objects_with_relations = [tabletop_reference, *placeable_assets]
        task = TableMultiObjectLayoutTask(
            objects_with_relations=objects_with_relations,
            placeable_assets=placeable_assets,
            num_envs=num_envs,
        )

        scene = Scene(
            assets=[
                ground_plane,
                table_background,
                tabletop_reference,
                *placeable_assets,
                light,
            ]
        )
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            scene=scene,
            embodiment=embodiment,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--objects",
            nargs="*",
            type=str,
            default=None,
            help=f"Object names to spawn on the table (On + NoCollision). Default: {' '.join(DEFAULT_TABLE_OBJECTS)}",
        )
        parser.add_argument("--embodiment", type=str, default="gr1_joint", help="Robot embodiment to use")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
