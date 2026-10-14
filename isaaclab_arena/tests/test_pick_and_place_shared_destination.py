# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import traceback
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 50
HEADLESS = True


class _AssetDouble:
    """Asset whose contact sensor filters contacts against other assets by name."""

    def __init__(self, name, object_type):
        self.name = name
        self.object_type = object_type
        self.object_min_z = -1.0

    def get_contact_sensor_cfg(self, contact_against_objects=()):
        filter_prim_paths = [
            f"/World/{contact_against_object.name}" for contact_against_object in contact_against_objects
        ]
        return SimpleNamespace(prim_path=f"/World/{self.name}", filter_prim_paths_expr=filter_prim_paths)


def _get_contact_filter_paths(pick_and_place_task) -> list[str]:
    contact_sensor_cfg = getattr(pick_and_place_task.get_scene_cfg(), pick_and_place_task.contact_sensor_name)
    return contact_sensor_cfg.filter_prim_paths_expr


def _test_composite_filters_contacts_with_objects_on_same_destination(simulation_app) -> bool:
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    background = _AssetDouble("background", ObjectType.BASE)
    bowl = _AssetDouble("bowl", ObjectType.RIGID)
    plate = _AssetDouble("plate", ObjectType.RIGID)
    banana = _AssetDouble("banana", ObjectType.RIGID)
    cube = _AssetDouble("cube", ObjectType.RIGID)
    apple = _AssetDouble("apple", ObjectType.RIGID)
    towel = _AssetDouble("towel", ObjectType.DEFORMABLE)

    banana_task = PickAndPlaceTask(banana, bowl, background)
    cube_task = PickAndPlaceTask(cube, bowl, background)
    apple_task = PickAndPlaceTask(apple, plate, background)
    towel_task = PickAndPlaceTask(towel, bowl, background)
    assert _get_contact_filter_paths(cube_task) == ["/World/bowl"]

    CompositeTaskBase([banana_task, cube_task, apple_task, towel_task])

    # Contact sensors cannot filter against deformable objects, so the towel is not a support candidate.
    assert _get_contact_filter_paths(banana_task) == ["/World/bowl", "/World/cube"]
    assert _get_contact_filter_paths(cube_task) == ["/World/bowl", "/World/banana"]
    assert _get_contact_filter_paths(apple_task) == ["/World/plate"]
    assert towel_task.get_scene_cfg() is None
    return True


def _test_object_resting_on_another_object_counts_as_placed(simulation_app) -> bool:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("maple_table_robolab")()
    bowl = asset_registry.get_asset_by_name("bowl_ycb_robolab")(
        initial_pose=Pose(position_xyz=(0.55, 0.0, 0.03), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    )
    # The lower cube rests on the bowl floor and the upper cube rests only on the lower cube.
    lower_cube = asset_registry.get_asset_by_name("dex_cube")(
        instance_name="lower_cube",
        initial_pose=Pose(position_xyz=(0.55, 0.0, 0.035), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    upper_cube = asset_registry.get_asset_by_name("dex_cube")(
        instance_name="upper_cube",
        initial_pose=Pose(position_xyz=(0.55, 0.0, 0.085), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )

    lower_cube_task = PickAndPlaceTask(lower_cube, bowl, background)
    upper_cube_task = PickAndPlaceTask(upper_cube, bowl, background)
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="stacked_cubes_in_bowl",
        embodiment=FrankaIKEmbodiment(),
        scene=Scene(assets=[background, bowl, lower_cube, upper_cube]),
        task=CompositeTaskBase([lower_cube_task, upper_cube_task]),
    )

    builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = builder.make_registered()
    env.reset()

    try:
        with torch.inference_mode():
            for _ in range(NUM_STEPS):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

            lower_cube_z = env.unwrapped.arena_world.get_position_w(lower_cube.name)[:, 2]
            upper_cube_z = env.unwrapped.arena_world.get_position_w(upper_cube.name)[:, 2]
            assert torch.all(upper_cube_z > lower_cube_z + 0.04), "The upper cube no longer rests on the lower cube."

            for task in (lower_cube_task, upper_cube_task):
                placement_predicate = task.get_termination_cfg().success[0].predicate_sequence[-1].predicate
                is_placed = placement_predicate(env.unwrapped)
                assert torch.all(is_placed), f"{task.pick_up_object.name} does not count as placed in the bowl."

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def test_composite_filters_contacts_with_objects_on_same_destination():
    assert run_function_with_persistent_simulation_app(
        _test_composite_filters_contacts_with_objects_on_same_destination
    )


def test_object_resting_on_another_object_counts_as_placed():
    assert run_function_with_persistent_simulation_app(
        _test_object_resting_on_another_object_counts_as_placed,
        headless=HEADLESS,
    )


if __name__ == "__main__":
    test_composite_filters_contacts_with_objects_on_same_destination()
    test_object_resting_on_another_object_counts_as_placed()
