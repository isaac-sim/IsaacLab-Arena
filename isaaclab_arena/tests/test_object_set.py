# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
NUM_ENVS = 10
OBJECT_SET_1_PRIM_PATH = "/World/envs/env_.*/ObjectSet_1"
OBJECT_SET_2_PRIM_PATH = "/World/envs/env_.*/ObjectSet_2"


def _test_empty_object_set(simulation_app):
    from isaaclab_arena.assets.object_set import RigidObjectSet

    try:
        RigidObjectSet(name="empty_object_set", objects=[])
    except Exception:
        return True
    return False


def _test_articulation_object_set(simulation_app):
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet

    asset_registry = AssetRegistry()
    microwave = asset_registry.get_asset_by_name("microwave")()
    try:
        RigidObjectSet(name="articulation_object_set", objects=[microwave])
    except Exception:
        return True
    return False


def _test_single_object_in_one_object_set(simulation_app):
    from isaacsim.core.utils.stage import get_current_stage

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    obj_set = RigidObjectSet(
        name="single_object_set", objects=[cracker_box, cracker_box], prim_path=OBJECT_SET_1_PRIM_PATH
    )
    obj_set.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    scene = Scene(assets=[background, obj_set])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="single_object_set_test",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(
            pick_up_object=obj_set, destination_location=destination_location, background_scene=background
        ),
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    args_cli.headless = HEADLESS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    try:
        for i in range(NUM_ENVS):
            # Construct the actual prim path for this environment
            path = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            assert "cracker_box.usd" in path, "Path does not contain cracker_box.usd"
            assert obj_set.get_initial_pose() is not None, "Initial pose is None"

        assert env.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
        assert (
            env.scene.sensors["pick_up_object_contact_sensor"].data.force_matrix_w is not None
        ), "Contact sensor data is None"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_multi_objects_in_one_object_set(simulation_app):
    from isaacsim.core.utils.stage import get_current_stage

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    obj_set = RigidObjectSet(
        name="multi_object_sets", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    scene = Scene(assets=[background, obj_set])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multi_objects_in_one_object_set_test",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(
            pick_up_object=obj_set, destination_location=destination_location, background_scene=background
        ),
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    args_cli.headless = HEADLESS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    assert env.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
    assert (
        env.scene.sensors["pick_up_object_contact_sensor"].data.force_matrix_w is not None
    ), "Contact sensor data is None"

    # replace * in OBJECT_SET_PRIM_PATH with env_index
    object_paths = []
    try:
        for i in range(NUM_ENVS):

            path = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            object_paths.append(path)
        assert len(object_paths) == NUM_ENVS, "Object_paths length is not equal to NUM_ENVS"
        assert cracker_box.usd_path in object_paths, "Cracker box USD path is not in Object_paths"
        assert sugar_box.usd_path in object_paths, "Sugar box USD path is not in Object_paths"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_multi_object_sets(simulation_app):
    from isaacsim.core.utils.stage import get_current_stage

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()

    obj_set_1 = RigidObjectSet(
        name="multi_object_sets_1", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_1_PRIM_PATH
    )
    obj_set_2 = RigidObjectSet(
        name="multi_object_sets_2", objects=[sugar_box, mustard_bottle], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    scene = Scene(assets=[background, obj_set_1, obj_set_2])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multi_object_sets_test",
        embodiment=embodiment,
        scene=scene,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    args_cli.headless = HEADLESS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    try:
        object_1_paths = []
        object_2_paths = []
        for i in range(NUM_ENVS):

            path_1 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            path_2 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            object_1_paths.append(path_1)
            object_2_paths.append(path_2)
            assert path_1 is not None, (
                "Path_1 from Prim Path " + OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
            assert path_2 is not None, (
                "Path_2 from Prim Path " + OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
        assert len(object_1_paths) == NUM_ENVS, "Object_1_paths length is not equal to NUM_ENVS"
        assert len(object_2_paths) == NUM_ENVS, "Object_2_paths length is not equal to NUM_ENVS"
        # Check that each object in the set turns up in one of the environments
        # NOTE(alexmillane): If we get really unlucky, this can fail because every environment
        # gets the same object. The chance of this is 0.5^NUM_ENVS. So with 20 envs this is very small.
        assert cracker_box.usd_path in object_1_paths, "Cracker box USD path is not in Object_1_paths"
        assert sugar_box.usd_path in object_1_paths, "Sugar box USD path is not in Object_1_paths"
        assert sugar_box.usd_path in object_2_paths, "Sugar box USD path is not in Object_2_paths"
        assert mustard_bottle.usd_path in object_2_paths, "Mustard bottle USD path is not in Object_2_paths"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def test_empty_object_set():
    result = run_simulation_app_function(
        _test_empty_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_empty_object_set.__name__} failed"


def test_articulation_object_set():
    result = run_simulation_app_function(
        _test_articulation_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_articulation_object_set.__name__} failed"


def test_single_object_in_one_object_set():
    result = run_simulation_app_function(
        _test_single_object_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_single_object_in_one_object_set.__name__} failed"


def test_multi_objects_in_one_object_set():
    result = run_simulation_app_function(
        _test_multi_objects_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_objects_in_one_object_set.__name__} failed"


def test_multi_object_sets():
    result = run_simulation_app_function(
        _test_multi_object_sets,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_object_sets.__name__} failed"


if __name__ == "__main__":
    test_empty_object_set()
    test_articulation_object_set()
    test_single_object_in_one_object_set()
    test_multi_objects_in_one_object_set()
    test_multi_object_sets()
